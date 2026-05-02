"""Procedural top-down forest canopy layer.

Visual target: dense dark canopy with individual crowns visible from directly
above, warm earth clearings where coverage is partial. Total War Attila
campaign-map look — never side-view, never stems, never AI textures.

Pipeline:
  1. Read the WorldCover landcover_codes.png. Forest density = (class == 10),
     gaussian-blurred and broken up with low-frequency noise so polygon edges
     dissolve into organic clearings instead of reading as a clean polygon.
  2. Pre-generate a small pool of procedural canopy-mark alpha masks
     (round / lobed / small filler), all soft dark blobs with no stem.
  3. Jittered-grid placement: tighter spacing in dense areas, density-weighted
     rejection sampling. Each placement picks a random sprite from the pool
     and tints it with a climate-aware dark green (south = warm olive,
     north = cold pine).
  4. Composite via numpy alpha blend onto the climate-painted preview.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from maptool.regions import BBox

log = logging.getLogger(__name__)


FOREST_CLASS = 10  # ESA WorldCover 2021 v200: tree cover

# Canopy tint by climate zone — Total-War campaign-map dark green, slightly
# warmer in the south. Used both for the continuous mass underlay AND for the
# individual sprite dots on top, so they always read as "the same forest".
TINT_DRY = np.array([72, 84, 44], dtype=np.float32)        # < 38 N: olive-green
TINT_MOIST = np.array([55, 78, 40], dtype=np.float32)      # 38-44 N: mid green
TINT_TEMPERATE = np.array([42, 64, 36], dtype=np.float32)  # > 44 N: dark green

SPRITE_SOURCE_SIZE = 32    # bigger source -> Lanczos downsample has more info
                           # for clean anti-aliased edges (no hard pixel look)
SPRITE_POOL_SIZE = 32      # pre-rendered variants
PLACEMENT_CELL = 3         # less dense than cell=2; trees stay distinct
DENSITY_FLOOR = 0.07       # slightly higher floor -> fewer fringe sprites
DENSITY_BLUR_SIGMA = 2.0
NOISE_SIGMA = 32.0         # low-freq noise breaks polygon edges into clearings
NOISE_AMOUNT = 0.40        # higher noise amount -> more organic forest edges
SPRITE_SCALE_MIN = 0.09    # 32 * 0.09 = ~3 px effective sprite
SPRITE_SCALE_MAX = 0.28    # 32 * 0.28 = ~9 px for cluster variation
SPRITE_ALPHA_MIN = 0.85    # opaque sprites read as distinct stamps, not blur
SPRITE_ALPHA_MAX = 1.00
DENSITY_MAX_DIM = 4096     # cap density-mask resolution; saves 800 MB at ultra

# Phase 4 — forest as layered mass + crown-highlight sprites:
# - Continuous dark-olive base fills gaps where density is above MASS_FLOOR.
# - Sprites paint LIGHTER on top of that mass, so each sprite reads as a
#   sun-lit crown top instead of a shadow dot.
MASS_FLOOR = 0.08          # density below this -> no mass underlay
MASS_FULL = 0.42           # density above this -> max mass alpha
MASS_ALPHA_MAX = 0.62      # underlay strength (sprites still dominant)
SPRITE_BRIGHTEN = 1.55     # sprite_color = mass_tint * SPRITE_BRIGHTEN
MASS_STRIPE_ROWS = 512     # stripe height for the mass underlay loop


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def climate_tint_per_row(h: int, bbox: BBox) -> np.ndarray:
    """Return (h, 3) float32 RGB tint, one row per pixel-row of the canvas."""
    lats = np.linspace(bbox.north, bbox.south, h, dtype=np.float32)
    t_dm = _smoothstep(38.0, 42.0, lats)[:, None]
    t_mt = _smoothstep(42.0, 45.5, lats)[:, None]
    dry_moist = TINT_DRY[None, :] * (1.0 - t_dm) + TINT_MOIST[None, :] * t_dm
    return dry_moist * (1.0 - t_mt) + TINT_TEMPERATE[None, :] * t_mt


def _ellipse_alpha(rng: np.random.Generator, cx: float, cy: float, rx: float, ry: float,
                   size: int, falloff: float = 0.7) -> np.ndarray:
    """Soft ellipse alpha in [0,1]."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    d2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return np.clip(1.0 - d2, 0.0, 1.0) ** falloff


def _make_canopy(rng: np.random.Generator, kind: str) -> np.ndarray:
    """Build one (size, size) alpha mask. No stem, no separate shadow.

    All radii are expressed as fractions of *size* so changing
    SPRITE_SOURCE_SIZE rescales sprites without retuning the params.
    """
    size = SPRITE_SOURCE_SIZE
    s = float(size)
    # _ellipse_alpha = (1 - d^2)^falloff. Low falloff (<0.5) keeps the body
    # at full opacity and drops sharply at the rim — crisp sprites with a
    # thin anti-alias band, no Gaussian halo.
    if kind == "round":
        cx = s / 2 + rng.uniform(-0.04, 0.04) * s
        cy = s / 2 + rng.uniform(-0.04, 0.04) * s
        rx = rng.uniform(0.35, 0.42) * s
        ry = rng.uniform(0.32, 0.40) * s
        a = _ellipse_alpha(rng, cx, cy, rx, ry, size, falloff=0.30)
    elif kind == "lobed":
        a = np.zeros((size, size), dtype=np.float32)
        n_lobes = int(rng.integers(2, 4))
        for _ in range(n_lobes):
            cx = s / 2 + rng.uniform(-0.13, 0.13) * s
            cy = s / 2 + rng.uniform(-0.13, 0.13) * s
            rx = rng.uniform(0.22, 0.31) * s
            ry = rng.uniform(0.22, 0.31) * s
            a = np.maximum(a, _ellipse_alpha(rng, cx, cy, rx, ry, size, falloff=0.25))
    elif kind == "small":
        cx = s / 2 + rng.uniform(-0.025, 0.025) * s
        cy = s / 2 + rng.uniform(-0.025, 0.025) * s
        rx = rng.uniform(0.20, 0.28) * s
        ry = rng.uniform(0.19, 0.26) * s
        a = _ellipse_alpha(rng, cx, cy, rx, ry, size, falloff=0.35)
    else:
        raise ValueError(f"unknown canopy kind: {kind}")
    # Sub-pixel grain so identical sprites don't read as flat dots.
    grain = rng.uniform(0.92, 1.0, size=a.shape).astype(np.float32)
    return a * grain


def _build_sprite_pool(seed: int) -> list[np.ndarray]:
    """Pre-render SPRITE_POOL_SIZE alpha masks. Mix: 50% round, 30% lobed, 20% small."""
    rng = np.random.default_rng(seed)
    n = SPRITE_POOL_SIZE
    kinds = (["round"] * (n // 2) + ["lobed"] * (3 * n // 10) + ["small"] * (n // 5))
    while len(kinds) < n:
        kinds.append("round")
    rng.shuffle(kinds)
    return [_make_canopy(rng, k) for k in kinds]


# Pre-baked size buckets so the inner loop never calls PIL.resize per placement.
SIZE_BUCKETS = (2, 3, 4, 5, 6, 7, 8, 9)


def _build_sized_pool(seed: int) -> dict[int, list[np.ndarray]]:
    """Return {target_px: [alpha_mask]} so placement only needs random pick."""
    base = _build_sprite_pool(seed)
    sized: dict[int, list[np.ndarray]] = {}
    for sz in SIZE_BUCKETS:
        sized[sz] = [_resize_alpha(a, sz) for a in base]
    return sized


def _build_density(codes: np.ndarray, seed: int) -> np.ndarray:
    """Forest density mask in [0,1] from raw WorldCover class codes.

    At ultra resolution (200 MP) the full-res density mask plus its noise
    twin would burn 1.6 GB of float32. Density is a smooth, blurry mask
    anyway, so we always cap it at DENSITY_MAX_DIM (4096) and let the
    scatter loop index it with a scale factor.
    """
    h, w = codes.shape
    target_max = max(h, w)
    if target_max > DENSITY_MAX_DIM:
        scale = DENSITY_MAX_DIM / target_max
        small_h = max(1, int(round(h * scale)))
        small_w = max(1, int(round(w * scale)))
        small_codes = np.array(
            Image.fromarray(codes, mode="L").resize((small_w, small_h), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    else:
        small_codes = codes
        small_h, small_w = h, w

    raw = (small_codes == FOREST_CLASS).astype(np.float32)
    soft = gaussian_filter(raw, sigma=DENSITY_BLUR_SIGMA)
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(
        rng.normal(0.0, 1.0, (small_h, small_w)).astype(np.float32),
        sigma=NOISE_SIGMA,
    )
    nmin, nmax = float(noise.min()), float(noise.max())
    noise = (noise - nmin) / (nmax - nmin) if nmax > nmin else np.zeros_like(noise)
    density = soft * (1.0 - NOISE_AMOUNT + NOISE_AMOUNT * noise)
    return np.clip(density, 0.0, 1.0)


def _resize_alpha(alpha: np.ndarray, target: int) -> np.ndarray:
    img = Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L")
    img = img.resize((target, target), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def render_forest(
    color_png: Path,
    codes_png: Path,
    meta_path: Path,
    out_png: Path,
    *,
    seed: int = 0xF0E5,
) -> Path:
    """Composite a procedural forest layer onto the climate-painted preview."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    b = meta["bbox"]
    bbox = BBox(b["south"], b["north"], b["west"], b["east"])

    # uint8 canvas keeps peak memory bounded at 200 MP (600 MB instead of
    # 2.4 GB float32). Per-sprite blending lifts a tiny slice to float32 and
    # writes it back as uint8 — overhead is negligible.
    with Image.open(color_png) as img_color:
        canvas = np.array(img_color.convert("RGB"), dtype=np.uint8)
    H, W, _ = canvas.shape

    with Image.open(codes_png) as codes_img:
        if codes_img.size != (W, H):
            codes_img = codes_img.resize((W, H), Image.Resampling.NEAREST)
        codes = np.asarray(codes_img, dtype=np.uint8)

    density = _build_density(codes, seed=seed)
    if density.max() < DENSITY_FLOOR:
        log.info("  forest skipped: no class-%d cells above floor", FOREST_CLASS)
        Image.fromarray(canvas, "RGB").save(out_png, optimize=True)
        return out_png

    # Density may have been computed at lower resolution (DENSITY_MAX_DIM cap)
    # to bound memory at ultra. Convert canvas-pixel indices to density-pixel
    # indices via a scale factor when sampling.
    density_h, density_w = density.shape
    dh_scale = np.float32(density_h / H)
    dw_scale = np.float32(density_w / W)

    tint = climate_tint_per_row(H, bbox)

    # Phase 4 — continuous mass underlay so dense forest reads as solid green
    # (no climate-paint leaking through gaps). Sprites paint LIGHTER on top,
    # giving the "sun-lit crown top" highlight that the references show.
    half_cell = 1
    inv255 = np.float32(1.0 / 255.0)
    mass_floor = np.float32(MASS_FLOOR)
    mass_full = np.float32(MASS_FULL)
    mass_amax = np.float32(MASS_ALPHA_MAX)
    for y0 in range(0, H, MASS_STRIPE_ROWS):
        y1 = min(y0 + MASS_STRIPE_ROWS, H)
        # Sample low-res density at this canvas stripe via index scaling.
        cy_idx = np.minimum(
            (np.arange(y0, y1, dtype=np.int32) * dh_scale).astype(np.int32),
            density_h - 1,
        )
        cx_idx = np.minimum(
            (np.arange(W, dtype=np.int32) * dw_scale).astype(np.int32),
            density_w - 1,
        )
        d_chunk = density[cy_idx[:, None], cx_idx[None, :]]
        # Smoothstep in float32 for the alpha ramp.
        t = np.clip((d_chunk - mass_floor) / (mass_full - mass_floor), 0.0, 1.0)
        alpha = (t * t * (3.0 - 2.0 * t) * mass_amax).astype(np.float32, copy=False)
        if alpha.max() <= 0.0:
            continue
        a3 = alpha[..., None]
        tint_chunk = np.broadcast_to(tint[y0:y1, None, :], (y1 - y0, W, 3))
        slice_u8 = canvas[y0:y1]
        blended = slice_u8.astype(np.float32) * (np.float32(1.0) - a3)
        blended += tint_chunk * a3
        canvas[y0:y1] = np.clip(blended, 0, 255).astype(np.uint8)
        del d_chunk, t, alpha, a3, tint_chunk, blended, slice_u8

    # No continuous mass underlay — forest IS the mass of overlapping micro
    # sprites. Pre-baked sized pool keeps the inner loop allocation-free.
    sized = _build_sized_pool(seed ^ 0xA17C)
    bucket_arr = np.array(SIZE_BUCKETS, dtype=np.int32)

    rng = np.random.default_rng(seed ^ 0x9C42)
    cell = PLACEMENT_CELL
    half = cell // 2

    # Vectorize candidate sampling: for every grid cell, decide accept/reject
    # via density-weighted Bernoulli. Then loop only over accepted cells.
    cy_grid = np.arange(half, H - half, cell, dtype=np.int32)
    cx_grid = np.arange(half, W - half, cell, dtype=np.int32)
    yy, xx = np.meshgrid(cy_grid, cx_grid, indexing="ij")

    # Sample density at the matching low-res index for each canvas cell.
    dyy = np.minimum((yy * dh_scale).astype(np.int32), density_h - 1)
    dxx = np.minimum((xx * dw_scale).astype(np.int32), density_w - 1)
    d_at_cells = density[dyy, dxx]
    del dyy, dxx

    accept_prob = np.where(d_at_cells >= DENSITY_FLOOR,
                           d_at_cells ** 0.55, 0.0)
    accept = rng.random(d_at_cells.shape) < accept_prob
    ay, ax = np.where(accept)
    n_candidates = ay.size

    if n_candidates == 0:
        Image.fromarray(canvas, "RGB").save(out_png, optimize=True)
        log.info("  forest 0 sprites (density too low)")
        return out_png

    # Pre-roll all per-placement randoms in batch — much faster than per-call.
    px_jitter = rng.integers(-half, half + 1, size=n_candidates)
    py_jitter = rng.integers(-half, half + 1, size=n_candidates)
    pos_x = xx[ay, ax] + px_jitter
    pos_y = yy[ay, ax] + py_jitter
    densities = d_at_cells[ay, ax]
    scales = rng.uniform(SPRITE_SCALE_MIN, SPRITE_SCALE_MAX, size=n_candidates)
    scales *= 0.85 + densities * 0.30
    targets_px = np.clip(np.round(SPRITE_SOURCE_SIZE * scales), 2, max(SIZE_BUCKETS)).astype(np.int32)
    # Snap to nearest pre-baked bucket
    snapped = bucket_arr[np.argmin(np.abs(targets_px[:, None] - bucket_arr[None, :]), axis=1)]
    sprite_idx = rng.integers(0, SPRITE_POOL_SIZE, size=n_candidates)
    alpha_scales = rng.uniform(SPRITE_ALPHA_MIN, SPRITE_ALPHA_MAX, size=n_candidates).astype(np.float32)
    jitter_rgb = rng.uniform(-8.0, 8.0, size=(n_candidates, 3)).astype(np.float32)
    # Sprites paint LIGHTER than the mass underlay (Phase 4): each crown reads
    # as a sun-lit highlight on top of the dark forest mass.
    brighten = (rng.uniform(0.95, 1.10, size=n_candidates) * SPRITE_BRIGHTEN).astype(np.float32)

    n_placed = 0
    for i in range(n_candidates):
        sz = int(snapped[i])
        a = sized[sz][int(sprite_idx[i])] * alpha_scales[i]
        py, px = int(pos_y[i]), int(pos_x[i])
        h_s = a.shape[0]
        y0 = py - h_s // 2
        x0 = px - h_s // 2
        y1, x1 = y0 + h_s, x0 + h_s
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue
        sprite_rgb = np.clip(tint[py] * brighten[i] + jitter_rgb[i], 0.0, 255.0)
        canvas_slice = canvas[y0:y1, x0:x1]
        a3 = a[..., None]
        # Lift the tiny uint8 slice to float32 just for the blend, write back.
        blended = canvas_slice.astype(np.float32) * (np.float32(1.0) - a3)
        blended += sprite_rgb[None, None, :] * a3
        canvas_slice[:] = np.clip(blended, 0, 255).astype(np.uint8)
        n_placed += 1

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGB").save(out_png, optimize=True)
    log.info("  forest %d sprites  -> %s", n_placed, out_png.name)
    return out_png
