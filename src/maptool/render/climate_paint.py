"""Render a 16-bit heightmap as a Bohemia / Songs-of-Syx-style preview.

Pipeline:
  1. Normalize elevation to [0, 1].
  2. Pick a climate-aware color via latitude-weighted blend of three elevation
     palettes (DRY < 38°N, MOIST 38-44°N, TEMPERATE > 44°N).
  3. Lay a sandy beach band over the coastline (no line overlay).
  4. Multiply by hillshade (315°/45°) for geomorphology.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation, distance_transform_edt

from maptool.regions import BBox
from maptool.render.paint_noise import DEFAULT_AMPLITUDE, build_low_tier, sample_chunk
from maptool.render.palette import DRY_MED, MOIST_MED, TEMPERATE, interp_palette

log = logging.getLogger(__name__)

SEA_LEVEL_NORM = 0.005
BEACH_COLOR = np.array([228, 208, 168], dtype=np.float32)
HILLSHADE_MAX_DIM = 8192  # cap hillshade computation; bigger -> sharper relief
                          # at zoom; memory peak ~700 MB during gradient/slope.

# Layered compositor colors (Phase 2/3): rocky cliffs, snow caps, sharp sand line.
ROCKY_COLOR = np.array([128, 112, 96], dtype=np.float32)     # warm grey-brown stone
SNOW_COLOR = np.array([248, 248, 250], dtype=np.float32)     # off-white painterly
SAND_HOT_COLOR = np.array([245, 228, 188], dtype=np.float32) # bright dry sand at coast

# Mask thresholds — slope-driven biome transitions.
ROCKY_SLOPE_LO = 0.18
ROCKY_SLOPE_HI = 0.45
SNOW_NORM_LO = 0.78
SNOW_NORM_HI = 0.86
SNOW_SLOPE_LIMIT_LO = 0.10
SNOW_SLOPE_LIMIT_HI = 0.30
SNOW_ALPHA = 0.92
COAST_EDGE_PX = 2  # land-side ring width for the sharp sand highlight

# Beach width is now proportional to the smaller image dimension, so a 4k map
# and a 512px crop both look like the same scene at different zoom levels
# (Memory: feedback_terrain_style — "thick brush" was #1 complaint).
BEACH_WIDTH_FRAC = 0.0045  # ~18 px on a 4k-wide map
BEACH_WIDTH_MIN = 6
BEACH_WIDTH_MAX = 24


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


CLIMATE_STRIPE_ROWS = 128  # row-stripe height for the climate-paint chunker


def _climate_color_chunk(norm_chunk: np.ndarray, lat_chunk: np.ndarray) -> np.ndarray:
    """One row stripe. Accumulate weighted palettes into a single buffer."""
    t_dm = _smoothstep(38.0, 42.0, lat_chunk)[..., None].astype(np.float32, copy=False)
    t_mt = _smoothstep(42.0, 45.5, lat_chunk)[..., None].astype(np.float32, copy=False)
    one = np.float32(1.0)
    w_dry = (one - t_dm) * (one - t_mt)
    w_moist = t_dm * (one - t_mt)
    w_temp = t_mt

    out = interp_palette(norm_chunk, DRY_MED)
    out *= w_dry
    moist = interp_palette(norm_chunk, MOIST_MED)
    moist *= w_moist
    out += moist
    del moist
    temp = interp_palette(norm_chunk, TEMPERATE)
    temp *= w_temp
    out += temp
    return out


def climate_color(norm_elev: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    """Blend the three palettes by latitude. Returns (H,W,3) float32 in [0,255].

    Processes the frame in row stripes so an ultra-resolution render
    (200 MP) never holds three full-frame palettes plus weights at the same
    time. Peak memory ~ output buffer (~2.4 GB at 200 MP) + ~120 MB / stripe.
    """
    h, w = norm_elev.shape
    out = np.empty((h, w, 3), dtype=np.float32)
    for y0 in range(0, h, CLIMATE_STRIPE_ROWS):
        y1 = min(y0 + CLIMATE_STRIPE_ROWS, h)
        out[y0:y1] = _climate_color_chunk(norm_elev[y0:y1], lat_grid[y0:y1])
    return out


HILLSHADE_RANGE = (0.25, 1.05)


def compute_hillshade(
    norm_elev: np.ndarray,
    height_scale: float = 0.35,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    *,
    return_uint8: bool = False,
) -> np.ndarray:
    """Standard Horn hillshade, returns (H,W) float32 in [0.25, 1.05].

    Computed at a downscaled resolution (max HILLSHADE_MAX_DIM) and bilinearly
    upsampled to the input shape. The 30 m DEM source caps useful detail
    around 8 k px anyway, and hillshade is naturally smooth — so working at
    lower resolution avoids 4-5 GB of float32 gradient/slope/aspect temps at
    ultra resolution.
    """
    h, w = norm_elev.shape
    scale = min(1.0, HILLSHADE_MAX_DIM / max(h, w))
    if scale < 1.0:
        small_h = max(2, int(round(h * scale)))
        small_w = max(2, int(round(w * scale)))
        small = _downscale_norm(norm_elev, small_h, small_w)
    else:
        small_h, small_w = h, w
        small = norm_elev

    az = np.float32(np.deg2rad(360.0 - azimuth_deg + 90.0))
    alt = np.float32(np.deg2rad(altitude_deg))
    sin_alt = np.float32(np.sin(alt))
    cos_alt = np.float32(np.cos(alt))

    scaled = small * np.float32(height_scale * small_h)
    dy, dx = np.gradient(scaled)
    del scaled
    grad_mag = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    slope = np.float32(np.pi / 2.0) - np.arctan(grad_mag).astype(np.float32, copy=False)
    del grad_mag
    aspect = np.arctan2(-dx, dy).astype(np.float32, copy=False)
    del dx, dy
    shade_small = sin_alt * np.sin(slope) + cos_alt * np.cos(slope) * np.cos(az - aspect)
    del slope, aspect
    np.clip(shade_small, 0.25, 1.05, out=shade_small)

    lo, hi = HILLSHADE_RANGE
    rng = np.float32(hi - lo)
    if scale < 1.0:
        shade_u8_small = ((shade_small - np.float32(lo)) * np.float32(255.0 / rng) + 0.5).astype(np.uint8)
        upsampled_u8 = np.array(
            Image.fromarray(shade_u8_small, mode="L").resize((w, h), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        if return_uint8:
            return upsampled_u8
        return upsampled_u8.astype(np.float32) * np.float32(rng / 255.0) + np.float32(lo)
    if return_uint8:
        return ((shade_small - np.float32(lo)) * np.float32(255.0 / rng) + 0.5).astype(np.uint8)
    return shade_small.astype(np.float32, copy=False)


BEACH_EDT_MAX_DIM = 4096  # cap distance-transform input -> bounded memory


def _downscale_norm(norm: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour downscale of a float32 array via numpy fancy index.

    PIL's Image.fromarray on a 200 MP float32 buffer allocates an 800 MB temp
    inside the resize step which OOMs. The DEM source is already pre-smoothed
    (bicubic upsample + optional pre-blur) so nearest-neighbour at low res is
    visually equivalent at the target scale.
    """
    h, w = norm.shape
    yi = np.linspace(0, h - 1, target_h).astype(np.int64)
    xi = np.linspace(0, w - 1, target_w).astype(np.int64)
    return norm[yi[:, None], xi[None, :]].astype(np.float32, copy=True)


def compute_slope(norm_elev: np.ndarray, *, return_uint8: bool = False) -> np.ndarray:
    """Slope magnitude tanh-normalized to [0, 1].

    Computed at the same downsampled resolution as hillshade for memory; the
    rocky-grey and snow-cap masks both consume this so it's pre-derived once.
    """
    h, w = norm_elev.shape
    scale = min(1.0, HILLSHADE_MAX_DIM / max(h, w))
    if scale < 1.0:
        small_h = max(2, int(round(h * scale)))
        small_w = max(2, int(round(w * scale)))
        small = _downscale_norm(norm_elev, small_h, small_w)
    else:
        small_h, small_w = h, w
        small = norm_elev

    scaled = small * np.float32(0.35 * small_h)
    dy, dx = np.gradient(scaled)
    grad_mag = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    del dx, dy
    slope_norm = np.tanh(grad_mag * np.float32(0.18)).astype(np.float32, copy=False)

    if scale < 1.0:
        slope_u8_small = (slope_norm * 255.0 + 0.5).astype(np.uint8)
        upsampled = np.array(
            Image.fromarray(slope_u8_small, mode="L").resize((w, h), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        if return_uint8:
            return upsampled
        return upsampled.astype(np.float32) * np.float32(1.0 / 255.0)
    if return_uint8:
        return (slope_norm * 255.0 + 0.5).astype(np.uint8)
    return slope_norm


def compute_coast_edge(norm_elev: np.ndarray, ring_px: int = COAST_EDGE_PX) -> np.ndarray:
    """uint8 255 on a thin ring of land cells touching the sea, 0 elsewhere.

    Computed at downscaled resolution (max 4096) like beach/hillshade so binary
    dilation never holds three 200 MB bool buffers at ultra. The ring widens
    proportionally on upsample, which still reads as a sharp sand line.
    """
    h, w = norm_elev.shape
    scale = min(1.0, BEACH_EDT_MAX_DIM / max(h, w))
    if scale < 1.0:
        small_h = max(2, int(round(h * scale)))
        small_w = max(2, int(round(w * scale)))
        small = _downscale_norm(norm_elev, small_h, small_w)
    else:
        small = norm_elev

    sea_small = small < SEA_LEVEL_NORM
    if not sea_small.any() or sea_small.all():
        return np.zeros((h, w), dtype=np.uint8)

    dilated = binary_dilation(sea_small, iterations=ring_px)
    np.logical_and(dilated, ~sea_small, out=dilated)
    coast_small_u8 = dilated.astype(np.uint8) * np.uint8(255)
    del sea_small, dilated

    if scale < 1.0:
        coast_u8 = np.array(
            Image.fromarray(coast_small_u8, mode="L").resize((w, h), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        return coast_u8
    return coast_small_u8


def compute_beach_blend(norm_elev: np.ndarray, *, return_uint8: bool = False) -> np.ndarray:
    """1.0 right at the coast, linear ramp to 0 *BEACH_WIDTH_PX* inland.

    scipy's distance_transform_edt allocates 2 × H × W float64 internally; for a
    200 MP frame that's 3 GB and OOMs on 16 GB systems. The beach gradient is
    naturally smooth, so we compute the EDT at a downscaled resolution (max
    4096 px) and bilinearly upsample the blend mask back. No visible detail loss.

    *return_uint8* keeps the upsampled blend at 1 byte/pixel (200 MB at 200 MP)
    instead of float32 (800 MB) — let render_preview do the float cast in
    stripes so peak memory stays bounded.
    """
    h, w = norm_elev.shape
    empty_dtype = np.uint8 if return_uint8 else np.float32
    if not (norm_elev >= SEA_LEVEL_NORM).any() or not (norm_elev < SEA_LEVEL_NORM).any():
        return np.zeros((h, w), dtype=empty_dtype)

    scale = min(1.0, BEACH_EDT_MAX_DIM / max(h, w))
    if scale < 1.0:
        small_h = max(1, int(round(h * scale)))
        small_w = max(1, int(round(w * scale)))
        small_norm = _downscale_norm(norm_elev, small_h, small_w)
    else:
        small_h, small_w = h, w
        small_norm = norm_elev

    sea_small = small_norm < SEA_LEVEL_NORM
    width_px = int(np.clip(
        round(min(small_h, small_w) * BEACH_WIDTH_FRAC),
        BEACH_WIDTH_MIN,
        BEACH_WIDTH_MAX,
    ))
    dist_land = distance_transform_edt(~sea_small).astype(np.float32, copy=False)
    blend_small = np.clip(np.float32(1.0) - dist_land * np.float32(1.0 / width_px), 0.0, 1.0)
    blend_small *= (~sea_small).astype(np.float32)

    if scale < 1.0:
        # Quantise to uint8 before the PIL roundtrip; float32 mode forces PIL
        # to allocate a 4-byte-per-pixel temp buffer (~800 MB at 200 MP).
        blend_small_u8 = (blend_small * 255.0 + 0.5).astype(np.uint8)
        blend_u8 = np.array(
            Image.fromarray(blend_small_u8, mode="L").resize((w, h), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        # Mask the upsampled blend with the full-res sea boundary so coast
        # pixels match the actual coastline (no soft sand creeping into water).
        full_sea = norm_elev < SEA_LEVEL_NORM
        blend_u8[full_sea] = 0
        if return_uint8:
            return blend_u8
        return blend_u8.astype(np.float32) * np.float32(1.0 / 255.0)
    if return_uint8:
        return (blend_small * 255.0 + 0.5).astype(np.uint8)
    return blend_small


def lat_grid_for(h: int, w: int, bbox: BBox) -> np.ndarray:
    """North row -> bbox.north, south row -> bbox.south. Equirectangular."""
    lats = np.linspace(bbox.north, bbox.south, h, dtype=np.float32)
    return np.broadcast_to(lats[:, None], (h, w))


def _bbox_from_meta(meta: dict) -> BBox:
    b = meta["bbox"]
    return BBox(b["south"], b["north"], b["west"], b["east"])


def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_preview(
    heightmap_path: Path,
    meta_path: Path,
    out_path: Path,
) -> Path:
    """Read a 16-bit heightmap + meta sidecar, write the climate-painted RGB PNG."""
    meta = load_meta(meta_path)
    bbox = _bbox_from_meta(meta)

    # Keep the heightmap as uint16 (400 MB at 200 MP) instead of float32
    # (800 MB). Decode a float32 norm chunk inside the stripe loop on demand.
    with Image.open(heightmap_path) as img:
        arr_u16 = np.asarray(img, dtype=np.uint16)
    mn_u16 = int(arr_u16.min())
    mx_u16 = int(arr_u16.max())
    if mx_u16 == mn_u16:
        norm_full = np.zeros_like(arr_u16, dtype=np.float32)
    else:
        # Build a temporary full-resolution float32 norm just for the
        # pre-derive masks (each downscales internally), then free it.
        norm_full = (arr_u16.astype(np.float32) - np.float32(mn_u16)) * np.float32(
            1.0 / (mx_u16 - mn_u16)
        )

    h, w = arr_u16.shape

    # Pre-compute modulators as uint8 (200 MB each at 200 MP).
    beach_u8 = compute_beach_blend(norm_full, return_uint8=True)
    hs_u8 = compute_hillshade(norm_full, return_uint8=True)
    slope_u8 = compute_slope(norm_full, return_uint8=True)
    coast_u8 = compute_coast_edge(norm_full)
    del norm_full  # 800 MB freed; we re-derive per-stripe from arr_u16

    norm_scale = np.float32(1.0 / (mx_u16 - mn_u16) if mx_u16 != mn_u16 else 1.0)
    norm_offset = np.float32(mn_u16)

    lat_grid = lat_grid_for(h, w, bbox)
    final = np.empty((h, w, 3), dtype=np.uint8)

    # Hand-paint texture: pre-bake low-freq pattern once; high-freq is per-stripe.
    paint_low = build_low_tier()
    paint_rng = np.random.default_rng(0xBADC0DE)
    paint_amp = np.float32(DEFAULT_AMPLITUDE)

    hs_lo, hs_hi = HILLSHADE_RANGE
    hs_scale = np.float32((hs_hi - hs_lo) / 255.0)
    hs_offset = np.float32(hs_lo)
    inv255 = np.float32(1.0 / 255.0)

    for y0 in range(0, h, CLIMATE_STRIPE_ROWS):
        y1 = min(y0 + CLIMATE_STRIPE_ROWS, h)
        # Decode the float32 norm chunk fresh from uint16 (memory-cheap).
        norm_chunk = (arr_u16[y0:y1].astype(np.float32) - norm_offset) * norm_scale
        chunk = _climate_color_chunk(norm_chunk, lat_grid[y0:y1])

        # 1. Hand-paint texture modulation (Phase 1).
        noise_chunk = sample_chunk(paint_low, h, w, y0, y1, paint_rng)
        chunk *= np.float32(1.0) + noise_chunk[..., None] * paint_amp
        del noise_chunk

        # 2. Slope-driven rocky-grey blend (Phase 2). Steep terrain shifts
        # toward warm-grey stone independently of climate zone.
        slope_chunk = slope_u8[y0:y1].astype(np.float32) * inv255
        rocky = _smoothstep(ROCKY_SLOPE_LO, ROCKY_SLOPE_HI, slope_chunk)
        rocky3 = rocky[..., None]
        chunk *= np.float32(1.0) - rocky3
        chunk += ROCKY_COLOR[None, None, :] * rocky3
        del rocky, rocky3

        # 3. Hillshade modulation (relief). Applied before snow so peaks stay bright.
        hs_chunk = hs_u8[y0:y1].astype(np.float32) * hs_scale + hs_offset
        chunk *= hs_chunk[..., None]
        del hs_chunk

        # 4. Snow caps (Phase 3a). High elevation AND not too steep -> painted
        # white over the shaded base, with a touch of underlying through.
        elev_term = _smoothstep(SNOW_NORM_LO, SNOW_NORM_HI, norm_chunk)
        flat_term = np.float32(1.0) - _smoothstep(SNOW_SLOPE_LIMIT_LO, SNOW_SLOPE_LIMIT_HI, slope_chunk)
        snow = (elev_term * flat_term * np.float32(SNOW_ALPHA))
        snow3 = snow[..., None]
        chunk *= np.float32(1.0) - snow3
        chunk += SNOW_COLOR[None, None, :] * snow3
        del elev_term, flat_term, snow, snow3, slope_chunk

        # 5. Beach soft gradient (existing).
        beach_chunk = beach_u8[y0:y1].astype(np.float32) * inv255
        beach3 = beach_chunk[..., None]
        chunk *= np.float32(1.0) - beach3
        chunk += BEACH_COLOR[None, None, :] * beach3
        del beach3, beach_chunk

        # 6. Sharp sand highlight at the sea-land step (Phase 3b).
        coast_chunk = coast_u8[y0:y1].astype(np.float32) * inv255
        coast3 = coast_chunk[..., None]
        chunk *= np.float32(1.0) - coast3
        chunk += SAND_HOT_COLOR[None, None, :] * coast3
        del coast3, coast_chunk

        np.clip(chunk, 0, 255, out=chunk)
        final[y0:y1] = chunk.astype(np.uint8)
        del chunk, norm_chunk

    del beach_u8, hs_u8, slope_u8, coast_u8, lat_grid, paint_low, arr_u16

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Subtle unsharp mask brings back edge crispness that bicubic upsamples
    # of beach/hillshade/coast and the soft paint-noise modulation otherwise
    # smear when the user zooms in on the final PNG.
    final_img = Image.fromarray(final, "RGB")
    final_img = final_img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=110, threshold=2))
    final_img.save(out_path, "PNG", optimize=True)
    log.info("  preview %dx%d  lat %.1f..%.1f  -> %s",
             w, h, bbox.south, bbox.north, out_path.name)
    return out_path
