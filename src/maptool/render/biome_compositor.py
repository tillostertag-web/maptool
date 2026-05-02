"""Biome-texture compositor — paints the map by tiling per-biome textures.

For each output pixel:

1. classify_biome(elev, slope, landcover, lat) picks one of the 7 biomes
2. sample the biome's tileable texture at (px % tile_px, py % tile_px)
3. multiply by hillshade

Tile-size in pixels controls how *small* each painted patch reads on the
map. At ``tile_px=96`` and a 16k-wide Sicily, each tile covers ≈ 2.5 km of
real terrain — Carthage-Bellum-Punicum-style painterly granularity.

Memory bounded at ultra resolution by stripe-processing: textures are
small (~3 MB each at 1024² uint8), masks pre-derived as uint8, and the
inner loop allocates only stripe-sized float32 buffers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from maptool.assets.biomes import CATALOG_STANDARD
from maptool.assets.generator import asset_paths
from maptool.assets.procedural import DEFAULT_PARAMS, BiomeParams, render_biome
from maptool.regions import BBox
from maptool.render.climate_paint import (
    HILLSHADE_RANGE,
    SEA_LEVEL_NORM,
    compute_beach_blend,
    compute_coast_edge,
    compute_hillshade,
    compute_slope,
    lat_grid_for,
)
from maptool.render.paint_noise import build_low_tier, sample_chunk

log = logging.getLogger(__name__)


# Biome priority order. classify_biome walks this list and assigns the FIRST
# matching biome per pixel — so put more-specific biomes first.
BIOME_ORDER: tuple[str, ...] = (
    "sea",
    "snow",
    "alpine_rock",
    "cliffs",
    "temperate_forest",
    "temperate_grass",
    "coast",
    "arid",
)
BIOME_INDEX: dict[str, int] = {slug: i for i, slug in enumerate(BIOME_ORDER)}

# Classification thresholds — tuned heuristically. These are over the same
# normalized [0,1] slope/elev signals climate_paint uses.
SLOPE_ALPINE_LO = 0.40        # very steep -> bare alpine rock face
CLIFFS_NORM_LO = 0.55         # high band below the snow line -> cliffs
CLIFFS_NORM_HI = 0.78         # above this -> snow
SNOW_NORM_LO = 0.78
SNOW_SLOPE_MAX = 0.20
COAST_DIST_PX = 2              # land within N px of sea -> coast (thin sand line)
DRY_LAT_MAX = 38.5             # latitudes south of this favour arid over grass

# Default tile size in canvas pixels (controls painterly granularity).
DEFAULT_TILE_PX = 96
COMPOSITE_STRIPE_ROWS = 256
PAINT_NOISE_AMPLITUDE = 0.10   # ±10 % global hand-paint grain on the final RGB


def load_biome_params(cache_root: Path) -> dict[str, BiomeParams]:
    """Load edited params from biome_params.json, falling back to defaults."""
    state_path = cache_root / "biome_params.json"
    out = {slug: DEFAULT_PARAMS[slug] for slug in BIOME_ORDER}
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            for slug, p_dict in data.get("params", {}).items():
                if slug in out:
                    out[slug] = BiomeParams.from_dict(p_dict)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("biome_params.json unreadable: %s", e)
    return out


def load_or_render_textures(
    cache_root: Path,
    *,
    tile_px: int,
    seed: int = 20260501,
) -> dict[str, np.ndarray]:
    """Return ``{slug: (tile_px, tile_px, 3) uint8}`` ready for sampling.

    Uses cached PNGs in ``<cache>/biomes/<slug>.png`` if present; otherwise
    renders them on the fly from the editable params.
    """
    params = load_biome_params(cache_root)
    out: dict[str, np.ndarray] = {}
    for slug in BIOME_ORDER:
        biome = next(b for b in CATALOG_STANDARD if b.slug == slug)
        asset = asset_paths(cache_root, biome)
        if asset.image_path.exists():
            img = Image.open(asset.image_path).convert("RGB")
        else:
            img = render_biome(params[slug], size=1024, seed=seed)
        img = img.resize((tile_px, tile_px), Image.Resampling.LANCZOS)
        out[slug] = np.asarray(img, dtype=np.uint8)
    return out


def _classify_chunk(
    norm_chunk: np.ndarray,
    slope_chunk: np.ndarray,
    codes_chunk: np.ndarray,
    lat_chunk: np.ndarray,
    coast_chunk: np.ndarray,
) -> np.ndarray:
    """Return a (H, W) uint8 of biome indices following BIOME_ORDER."""
    h, w = norm_chunk.shape
    out = np.full((h, w), fill_value=BIOME_INDEX["arid"], dtype=np.uint8)

    # WorldCover class 80 = permanent water bodies, 0 = nodata (outside
    # coverage, typically open sea for our bboxes). Combine with elevation
    # threshold so we catch lakes too.
    sea_mask = (codes_chunk == 80) | (codes_chunk == 0) | (norm_chunk < SEA_LEVEL_NORM)
    snow_mask = (norm_chunk >= SNOW_NORM_LO) & (slope_chunk < SNOW_SLOPE_MAX) & ~sea_mask
    rock_mask = (slope_chunk >= SLOPE_ALPINE_LO) & ~sea_mask & ~snow_mask
    # Cliffs: high-altitude weathered band below the snowline. Slope-independent
    # so it covers the whole upper mountain shoulder, not just the cliff face.
    cliffs_mask = (
        (norm_chunk >= CLIFFS_NORM_LO) & (norm_chunk < CLIFFS_NORM_HI)
        & ~sea_mask & ~snow_mask & ~rock_mask
    )
    # No dedicated forest base layer — forest reads from the tree sprites on
    # top, not from a green underground. WorldCover tree-cover (10) gets
    # painted as grass; the sprites add the canopy.
    grass_mask = (
        ((codes_chunk == 10) | (codes_chunk == 20) | (codes_chunk == 30)
         | (codes_chunk == 40) | (codes_chunk == 90))
        & ~sea_mask & ~snow_mask & ~rock_mask & ~cliffs_mask
    )
    coast_mask = (
        (coast_chunk > 0)
        & ~sea_mask & ~snow_mask & ~rock_mask & ~cliffs_mask & ~grass_mask
    )

    out[sea_mask] = BIOME_INDEX["sea"]
    out[snow_mask] = BIOME_INDEX["snow"]
    out[rock_mask] = BIOME_INDEX["alpine_rock"]
    out[cliffs_mask] = BIOME_INDEX["cliffs"]
    out[grass_mask] = BIOME_INDEX["temperate_grass"]
    out[coast_mask] = BIOME_INDEX["coast"]
    return out


def _bbox_from_meta(meta: dict) -> BBox:
    b = meta["bbox"]
    return BBox(b["south"], b["north"], b["west"], b["east"])


def render_with_biomes(
    heightmap_path: Path,
    codes_path: Path,
    meta_path: Path,
    out_path: Path,
    cache_root: Path,
    *,
    tile_px: int = DEFAULT_TILE_PX,
) -> Path:
    """Compose the map by tiling biome textures + applying hillshade."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    bbox = _bbox_from_meta(meta)

    # Load heightmap as uint16 for low-memory norm decoding (200 MP / 400 MB).
    with Image.open(heightmap_path) as img:
        arr_u16 = np.asarray(img, dtype=np.uint16)
    h, w = arr_u16.shape
    mn_u16, mx_u16 = int(arr_u16.min()), int(arr_u16.max())
    span = max(1, mx_u16 - mn_u16)
    norm_scale = np.float32(1.0 / span)
    norm_offset = np.float32(mn_u16)

    # Pre-derive modulators (same approach as climate_paint).
    norm_full = (arr_u16.astype(np.float32) - norm_offset) * norm_scale
    hs_u8 = compute_hillshade(norm_full, return_uint8=True)
    slope_u8 = compute_slope(norm_full, return_uint8=True)
    coast_u8 = compute_coast_edge(norm_full, ring_px=COAST_DIST_PX)
    del norm_full

    # Landcover.
    with Image.open(codes_path) as ci:
        if ci.size != (w, h):
            ci = ci.resize((w, h), Image.Resampling.NEAREST)
        codes = np.asarray(ci, dtype=np.uint8)

    textures = load_or_render_textures(cache_root, tile_px=tile_px)
    log.info("biome compositor  tile_px=%d  textures=%d", tile_px, len(textures))

    final = np.empty((h, w, 3), dtype=np.uint8)
    lat_grid = lat_grid_for(h, w, bbox)
    lo, hi = HILLSHADE_RANGE
    hs_scale = np.float32((hi - lo) / 255.0)
    hs_offset = np.float32(lo)
    inv255 = np.float32(1.0 / 255.0)

    # Pre-compute the pixel->tile remainder for x; y is per-stripe.
    x_idx = (np.arange(w, dtype=np.int64) % tile_px).astype(np.int32)

    # Global hand-paint noise (low-freq pre-baked + per-stripe high-freq).
    paint_low = build_low_tier()
    paint_rng = np.random.default_rng(0xBADC0DE)
    paint_amp = np.float32(PAINT_NOISE_AMPLITUDE)

    for y0 in range(0, h, COMPOSITE_STRIPE_ROWS):
        y1 = min(y0 + COMPOSITE_STRIPE_ROWS, h)
        norm_chunk = (arr_u16[y0:y1].astype(np.float32) - norm_offset) * norm_scale
        slope_chunk = slope_u8[y0:y1].astype(np.float32) * inv255
        coast_chunk = coast_u8[y0:y1]
        codes_chunk = codes[y0:y1]
        lat_chunk = lat_grid[y0:y1]

        biome_idx = _classify_chunk(
            norm_chunk, slope_chunk, codes_chunk, lat_chunk, coast_chunk
        )

        y_idx = (np.arange(y0, y1, dtype=np.int64) % tile_px).astype(np.int32)

        # Hard per-pixel biome assignment. classify_biome already gives one
        # biome per pixel; here we just sample that biome's tileable texture.
        chunk_rgb = np.empty((y1 - y0, w, 3), dtype=np.uint8)
        for slug, idx in BIOME_INDEX.items():
            mask = biome_idx == idx
            if not mask.any():
                continue
            tex = textures[slug]
            sampled = tex[y_idx[:, None], x_idx[None, :]]
            chunk_rgb[mask] = sampled[mask]

        # Apply hillshade + global paint-noise modulation. The noise gives
        # every pixel hand-painted brush-stroke granularity (the look of the
        # reference: visible texture on every surface, not flat colour).
        hs_chunk = hs_u8[y0:y1].astype(np.float32) * hs_scale + hs_offset
        noise_chunk = sample_chunk(paint_low, h, w, y0, y1, paint_rng)
        chunk_f = chunk_rgb.astype(np.float32)
        chunk_f *= hs_chunk[..., None]
        chunk_f *= np.float32(1.0) + noise_chunk[..., None] * paint_amp
        np.clip(chunk_f, 0, 255, out=chunk_f)
        final[y0:y1] = chunk_f.astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final, "RGB").save(out_path, "PNG", optimize=True)
    log.info("  biome composite %dx%d  -> %s", w, h, out_path.name)
    return out_path
