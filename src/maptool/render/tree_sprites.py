"""Tree-sprite scatter — overlays loaded RGBA sprite PNGs onto the painted map.

Sprites are loaded from ``<cache>/sprites/trees/*.png`` (RGBA). Where the
landcover mask says ``forest`` (WorldCover class 10), we sample sprite
positions on a jittered grid and alpha-composite each sprite onto the
canvas. No tinting — the sprite's own colours dictate the look.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

log = logging.getLogger(__name__)

FOREST_CLASS = 10
SPRITE_DIR_NAME = "sprites/trees"

DEFAULT_PLACEMENT_CELL = 5          # px between scatter candidates (very dense)
DEFAULT_SPRITE_PX_MIN = 6           # smallest rendered sprite size
DEFAULT_SPRITE_PX_MAX = 14          # largest rendered sprite size
DEFAULT_DENSITY_FLOOR = 0.10        # below this no sprite
SPRITE_DARKEN = 0.82                # multiplier on sprite RGB at composite

# Per-tree shadow drop (NW sun -> shadow goes SE).
SHADOW_DY = 1                       # pixels south
SHADOW_DX = 1                       # pixels east
SHADOW_STRENGTH = 0.35              # how much darker the underlying canvas gets

# Density falls off with elevation (denser at low altitude, sparser high).
DENSITY_ELEV_LO = 0.20              # full density below this normalized elev
DENSITY_ELEV_HI = 0.55              # zero density above this

# Bushes on shrubland (WorldCover class 20).
SHRUB_CLASS = 20
BUSH_PX_MIN = 3
BUSH_PX_MAX = 6
BUSH_CELL = 4
BUSH_DARKEN = 0.78
DENSITY_BLUR_SIGMA = 2.5
NOISE_SIGMA = 22.0                  # low-freq noise to break polygon edges
NOISE_AMOUNT = 0.32

# Exclude scatter on cliffs + steep slopes — match biome_compositor thresholds.
TREE_NORM_MAX = 0.55
TREE_SLOPE_MAX = 0.30

SIZE_BUCKETS = (3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18)


# Latitude-based species. Each band picks ONE species so a whole forest patch
# (and a whole small region like Sicily) reads as a single tree type.
LAT_SPECIES_BANDS: tuple[tuple[float, str], ...] = (
    (38.0, "olive"),         # < 38°N: dry mediterranean
    (43.0, "oak"),           # 38-43°N: mid mediterranean / Italy
    (46.0, "pine"),          # 43-46°N: alpine south
    (90.0, "dense_cluster"), # >= 46°N: northern boreal
)


def species_for_lat(mid_lat: float) -> str:
    """Pick the tree slug for a region centred at *mid_lat*."""
    for upper, slug in LAT_SPECIES_BANDS:
        if mid_lat < upper:
            return slug
    return LAT_SPECIES_BANDS[-1][1]


def _list_sprites(cache_root: Path) -> list[Path]:
    folder = cache_root / SPRITE_DIR_NAME
    if not folder.exists():
        return []
    return sorted(folder.glob("*.png"))


def _load_sprite(path: Path) -> np.ndarray:
    """Load a PNG as (H, W, 4) uint8 — RGB premultiplied with alpha externally."""
    img = Image.open(path).convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def load_sized_pool(cache_root: Path, *, slug: str | None = None) -> dict[int, list[np.ndarray]]:
    """Pre-resize sprite PNG(s) to every SIZE_BUCKETS entry.

    If *slug* is given (e.g. "oak"), only that species is loaded — used so a
    whole forest reads as a single tree type. Without slug, all sprites mix.
    """
    sprite_paths = _list_sprites(cache_root)
    if slug is not None:
        sprite_paths = [p for p in sprite_paths if p.stem == slug]
    if not sprite_paths:
        log.warning("No tree sprites found in %s (slug=%s)",
                    cache_root / SPRITE_DIR_NAME, slug)
        return {}
    sized: dict[int, list[np.ndarray]] = {sz: [] for sz in SIZE_BUCKETS}
    for p in sprite_paths:
        img = Image.open(p).convert("RGBA")
        for sz in SIZE_BUCKETS:
            r = img.resize((sz, sz), Image.Resampling.LANCZOS)
            sized[sz].append(np.asarray(r, dtype=np.uint8))
    return sized


def _build_density(codes: np.ndarray, seed: int, *, target_class: int = FOREST_CLASS) -> np.ndarray:
    """Soft density mask in [0,1] from the WorldCover class grid."""
    raw = (codes == target_class).astype(np.float32)
    soft = gaussian_filter(raw, sigma=DENSITY_BLUR_SIGMA)
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(
        rng.normal(0.0, 1.0, codes.shape).astype(np.float32),
        sigma=NOISE_SIGMA,
    )
    nmin, nmax = float(noise.min()), float(noise.max())
    if nmax > nmin:
        noise = (noise - nmin) / (nmax - nmin)
    density = soft * (1.0 - NOISE_AMOUNT + NOISE_AMOUNT * noise)
    return np.clip(density, 0.0, 1.0)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def scatter_sprites(
    canvas_rgb: np.ndarray,
    codes: np.ndarray,
    cache_root: Path,
    *,
    norm: np.ndarray | None = None,
    slope: np.ndarray | None = None,
    species_slug: str | None = None,
    target_class: int = FOREST_CLASS,
    seed: int = 0xF0E5,
    cell: int = DEFAULT_PLACEMENT_CELL,
    sprite_px_min: int = DEFAULT_SPRITE_PX_MIN,
    sprite_px_max: int = DEFAULT_SPRITE_PX_MAX,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
    rgb_darken: float = SPRITE_DARKEN,
    shadow: bool = True,
) -> tuple[np.ndarray, int]:
    """Composite tree sprites onto *canvas_rgb* in place. Returns (canvas, n).

    *target_class* picks which WorldCover class drives the density mask
    (10 = forest for trees, 20 = shrubland for bushes). *species_slug*
    restricts the sprite pool to one tree type so a forest reads as a
    single species. *norm* / *slope* keep sprites off cliffs/steep slopes,
    plus elevation-falloff thins the forest going up the mountain.
    """
    h, w = canvas_rgb.shape[:2]
    if codes.shape != (h, w):
        # Resize landcover to canvas resolution with nearest neighbour.
        codes = np.asarray(
            Image.fromarray(codes, mode="L").resize((w, h), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )

    sized = load_sized_pool(cache_root, slug=species_slug)
    if not sized:
        return canvas_rgb, 0

    density = _build_density(codes, seed=seed, target_class=target_class)
    # Mask out cliffs + steep slopes if those layers are available — matches
    # the biome_compositor's classification so trees never appear on rock.
    if norm is not None and norm.shape == (h, w):
        density = density * (norm < TREE_NORM_MAX).astype(np.float32)
        # Elevation-driven density falloff: full at low elev, fades up the
        # mountain so forests thin out as you climb.
        elev_factor = np.float32(1.0) - _smoothstep(DENSITY_ELEV_LO, DENSITY_ELEV_HI, norm)
        density = density * elev_factor
    if slope is not None and slope.shape == (h, w):
        density = density * (slope < TREE_SLOPE_MAX).astype(np.float32)
    if density.max() < density_floor:
        return canvas_rgb, 0

    rng = np.random.default_rng(seed ^ 0x9C42)
    half = cell // 2
    cy_grid = np.arange(half, h - half, cell, dtype=np.int32)
    cx_grid = np.arange(half, w - half, cell, dtype=np.int32)
    yy, xx = np.meshgrid(cy_grid, cx_grid, indexing="ij")
    d_at = density[yy, xx]
    accept = rng.random(d_at.shape) < d_at ** 0.55
    ay, ax = np.where(accept)
    n = ay.size
    if n == 0:
        return canvas_rgb, 0

    # Pre-roll all randomisation in one batch.
    pos_y = yy[ay, ax] + rng.integers(-half, half + 1, size=n)
    pos_x = xx[ay, ax] + rng.integers(-half, half + 1, size=n)
    sizes = rng.integers(sprite_px_min, sprite_px_max + 1, size=n)
    bucket_arr = np.array(SIZE_BUCKETS, dtype=np.int32)
    snapped = bucket_arr[np.argmin(np.abs(sizes[:, None] - bucket_arr[None, :]), axis=1)]
    sprite_idx = rng.integers(0, len(next(iter(sized.values()))), size=n)
    alpha_scale = rng.uniform(0.85, 1.0, size=n).astype(np.float32)

    darken = np.float32(rgb_darken)
    shadow_strength = np.float32(SHADOW_STRENGTH)

    n_placed = 0
    for i in range(n):
        sz = int(snapped[i])
        spr = sized[sz][int(sprite_idx[i])]   # (sz, sz, 4) uint8
        py, px = int(pos_y[i]), int(pos_x[i])
        y0 = py - sz // 2
        x0 = px - sz // 2
        y1, x1 = y0 + sz, x0 + sz
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue

        alpha_full = spr[..., 3].astype(np.float32) / 255.0

        # Drop a soft shadow southeast of the sprite (NW sun).
        if shadow:
            sy0 = y0 + SHADOW_DY
            sx0 = x0 + SHADOW_DX
            sy1 = sy0 + sz
            sx1 = sx0 + sz
            if 0 <= sy0 and 0 <= sx0 and sy1 <= h and sx1 <= w:
                shadow_a = (alpha_full * shadow_strength)[..., None]
                shadow_slice = canvas_rgb[sy0:sy1, sx0:sx1].astype(np.float32)
                shadow_slice *= np.float32(1.0) - shadow_a
                canvas_rgb[sy0:sy1, sx0:sx1] = np.clip(shadow_slice, 0, 255).astype(np.uint8)

        rgb = spr[..., :3].astype(np.float32) * darken
        alpha = alpha_full * alpha_scale[i]
        a3 = alpha[..., None]
        slice_rgb = canvas_rgb[y0:y1, x0:x1].astype(np.float32)
        blended = slice_rgb * (np.float32(1.0) - a3) + rgb * a3
        canvas_rgb[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        n_placed += 1

    return canvas_rgb, n_placed


def render_tree_layer(
    color_png: Path,
    codes_png: Path,
    out_png: Path,
    cache_root: Path,
    *,
    height_png: Path | None = None,
    meta_path: Path | None = None,
    seed: int = 0xF0E5,
) -> Path:
    """Read a painted color PNG + landcover, scatter sprites, write out_png.

    If *height_png* + *meta_path* are provided, we derive a slope mask and
    pick a species per the bbox's mid-latitude so the whole region uses one
    tree type and trees stay off cliffs.
    """
    import json

    canvas = np.array(Image.open(color_png).convert("RGB"), dtype=np.uint8)
    codes = np.asarray(Image.open(codes_png), dtype=np.uint8)

    norm = None
    slope = None
    species = None
    if height_png is not None and meta_path is not None:
        with Image.open(height_png) as hi:
            arr = np.asarray(hi, dtype=np.uint16)
        mn, mx = int(arr.min()), int(arr.max())
        span = max(1, mx - mn)
        norm = ((arr.astype(np.float32) - mn) / span).astype(np.float32, copy=False)
        # Resize norm to canvas if needed.
        if norm.shape != canvas.shape[:2]:
            norm = np.asarray(
                Image.fromarray((norm * 65535.0).astype(np.uint16))
                .resize((canvas.shape[1], canvas.shape[0]), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 65535.0
        # Slope from gradient on a coarse downsample (cheap).
        from maptool.render.climate_paint import compute_slope
        slope = compute_slope(norm, return_uint8=False)
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        bbox = meta["bbox"]
        mid_lat = 0.5 * (bbox["south"] + bbox["north"])
        species = species_for_lat(mid_lat)
        log.info("  trees: lat=%.1f° -> species=%s", mid_lat, species)

    # Pass 1: bushes on shrubland — small sprites between potential big trees.
    canvas, n_bush = scatter_sprites(
        canvas, codes, cache_root,
        norm=norm, slope=slope, species_slug=species,
        target_class=SHRUB_CLASS,
        seed=seed ^ 0xB05,
        cell=BUSH_CELL,
        sprite_px_min=BUSH_PX_MIN,
        sprite_px_max=BUSH_PX_MAX,
        rgb_darken=BUSH_DARKEN,
        shadow=False,  # bushes are tiny -> shadow not worth the noise
    )

    # Pass 2: trees on forest — main canopy.
    canvas, n_tree = scatter_sprites(
        canvas, codes, cache_root,
        norm=norm, slope=slope, species_slug=species,
        target_class=FOREST_CLASS,
        seed=seed,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGB").save(out_png, "PNG", optimize=True)
    log.info("  trees %d sprites  +  bushes %d  -> %s", n_tree, n_bush, out_png.name)
    return out_png
