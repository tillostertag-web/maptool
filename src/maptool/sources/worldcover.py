"""ESA WorldCover 2021 v200 10m landcover.

Tiles are 3deg x 3deg cloud-optimized GeoTIFFs on AWS S3. We use rasterio's
HTTP-Range support to read only the windows we need (no full tile downloads).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds as window_from_bounds

from maptool.regions import BBox, output_dims

log = logging.getLogger(__name__)

S3_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
TILE_DEG = 3

# WorldCover v200 class IDs -> RGB. Muted for cartography, not the spec neon.
WORLDCOVER_COLORS: dict[int, tuple[int, int, int]] = {
    0:   ( 45,  75, 110),   # nodata -> water-tinted (avoids coastal seam)
    10:  ( 60,  95,  55),   # tree cover (dark green)
    20:  (155, 145,  85),   # shrubland (olive-tan)
    30:  (175, 195, 110),   # grassland
    40:  (205, 195, 140),   # cropland
    50:  (160, 110,  95),   # built-up (muted red)
    60:  (195, 185, 155),   # bare / sparse
    70:  (245, 245, 245),   # snow / ice
    80:  ( 45,  75, 110),   # permanent water (matches preview deep-water)
    90:  (110, 145, 135),   # herbaceous wetland
    95:  ( 70, 110,  85),   # mangroves
    100: (200, 210, 170),   # moss / lichen
}


def _tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map.tif"


def _tiles_for_bbox(bbox: BBox) -> list[tuple[int, int]]:
    """All tile SW corners that intersect bbox (multiples of TILE_DEG)."""
    lat_start = int(bbox.south // TILE_DEG) * TILE_DEG
    lon_start = int(bbox.west // TILE_DEG) * TILE_DEG
    eps = 1e-9
    lat_end = int((bbox.north - eps) // TILE_DEG) * TILE_DEG
    lon_end = int((bbox.east - eps) // TILE_DEG) * TILE_DEG
    tiles = []
    for lat in range(lat_start, lat_end + 1, TILE_DEG):
        for lon in range(lon_start, lon_end + 1, TILE_DEG):
            tiles.append((lat, lon))
    return tiles


def _fetch_window(
    tile_lat: int,
    tile_lon: int,
    ix_west: float,
    ix_south: float,
    ix_east: float,
    ix_north: float,
    out_shape: tuple[int, int],
) -> np.ndarray | None:
    """Read a tile sub-window remotely and resample to out_shape."""
    url = f"{S3_BASE}/{_tile_name(tile_lat, tile_lon)}"
    t0 = time.time()
    try:
        with rasterio.open(url) as src:
            window = window_from_bounds(
                ix_west, ix_south, ix_east, ix_north, transform=src.transform
            )
            data = src.read(
                1,
                window=window,
                out_shape=out_shape,
                resampling=Resampling.mode,  # categorical: nearest/mode, never bilinear
                boundless=True,
                fill_value=0,
            )
        log.info("  %s read %dx%d in %.1fs",
                 _tile_name(tile_lat, tile_lon), out_shape[1], out_shape[0], time.time() - t0)
        return data
    except Exception as e:  # noqa: BLE001
        log.warning("  FAIL %s: %s", _tile_name(tile_lat, tile_lon), e)
        return None


def build_mosaic(bbox: BBox, max_dim: int) -> np.ndarray:
    """Stitch a class-code mosaic for *bbox*, max(width,height)<=max_dim."""
    out_w, out_h = output_dims(bbox, max_dim)
    mosaic = np.zeros((out_h, out_w), dtype=np.uint8)

    lat_per_px = bbox.lat_span() / out_h
    lon_per_px = bbox.lon_span() / out_w

    tiles = _tiles_for_bbox(bbox)
    log.info("  %d WorldCover tile(s) for bbox", len(tiles))

    for lat, lon in tiles:
        tile_s, tile_n = lat, lat + TILE_DEG
        tile_w, tile_e = lon, lon + TILE_DEG

        ix_s = max(bbox.south, tile_s)
        ix_n = min(bbox.north, tile_n)
        ix_w = max(bbox.west, tile_w)
        ix_e = min(bbox.east, tile_e)
        if ix_s >= ix_n or ix_w >= ix_e:
            continue

        out_y0 = int(round((bbox.north - ix_n) / lat_per_px))
        out_y1 = int(round((bbox.north - ix_s) / lat_per_px))
        out_x0 = int(round((ix_w - bbox.west) / lon_per_px))
        out_x1 = int(round((ix_e - bbox.west) / lon_per_px))
        out_y0 = max(0, min(out_h, out_y0))
        out_y1 = max(0, min(out_h, out_y1))
        out_x0 = max(0, min(out_w, out_x0))
        out_x1 = max(0, min(out_w, out_x1))

        local_h = out_y1 - out_y0
        local_w = out_x1 - out_x0
        if local_h <= 0 or local_w <= 0:
            continue

        data = _fetch_window(lat, lon, ix_w, ix_s, ix_e, ix_n, (local_h, local_w))
        if data is None:
            continue
        mosaic[out_y0:out_y0 + local_h, out_x0:out_x0 + local_w] = data

    return mosaic


def colorize(classes: np.ndarray) -> np.ndarray:
    """Class-code (H,W) uint8 -> RGB (H,W,3) uint8."""
    h, w = classes.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = WORLDCOVER_COLORS[0]
    for code, color in WORLDCOVER_COLORS.items():
        mask = classes == code
        if mask.any():
            out[mask] = color
    return out


def write_landcover(mosaic: np.ndarray, codes_path: Path, color_path: Path) -> None:
    """Write the categorical PNG and the colorized RGB PNG."""
    codes_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mosaic, "L").save(codes_path, "PNG", optimize=True)
    Image.fromarray(colorize(mosaic), "RGB").save(color_path, "PNG", optimize=True)


def class_distribution(mosaic: np.ndarray) -> list[tuple[int, float]]:
    """Returns (class_id, percentage) sorted descending, omitting <0.1%."""
    unique, counts = np.unique(mosaic, return_counts=True)
    total = counts.sum()
    pairs = [(int(c), 100.0 * n / total) for c, n in zip(unique, counts)]
    pairs.sort(key=lambda p: -p[1])
    return [p for p in pairs if p[1] >= 0.1]
