"""DEM source from OpenTopography.

Streams a GeoTIFF for a bbox using one of several DEM products
(SRTMGL1 30m, SRTMGL3 90m, COP30 30m, AW3D30 30m, NASADEM 30m), resamples to a
max-dimension PNG-friendly size, and writes a 16-bit grayscale PNG plus a JSON
sidecar with the elevation range.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import gaussian_filter

from maptool.regions import BBox

log = logging.getLogger(__name__)

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
DEFAULT_DEM_TYPE = "SRTMGL1"   # 30m, replaces former 90m default; better detail
SEA_NODATA_THRESHOLD = -1000.0  # SRTM nodata is -32768; sea cells are typically >=-1

SUPPORTED_DEMS = ("SRTMGL1", "SRTMGL3", "COP30", "COP90", "AW3D30", "NASADEM")


def _build_url(bbox: BBox, api_key: str, dem_type: str) -> str:
    params = {
        "demtype": dem_type,
        "south": bbox.south,
        "north": bbox.north,
        "west": bbox.west,
        "east": bbox.east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    return f"{OPENTOPO_URL}?{urlencode(params)}"


def download_dem(bbox: BBox, api_key: str, out_path: Path, dem_type: str = DEFAULT_DEM_TYPE) -> Path:
    """Stream a GeoTIFF for *bbox* into *out_path*."""
    if dem_type not in SUPPORTED_DEMS:
        raise ValueError(f"unsupported DEM {dem_type!r}; pick one of {SUPPORTED_DEMS}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = _build_url(bbox, api_key, dem_type)
    log.info("DEM download: %s S=%.2f N=%.2f W=%.2f E=%.2f",
             dem_type, bbox.south, bbox.north, bbox.west, bbox.east)

    with requests.get(url, stream=True, timeout=900) as r:
        if r.status_code != 200:
            raise RuntimeError(
                f"OpenTopography {r.status_code}: {r.text[:500]}"
            )
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                total += len(chunk)
        log.info("  %.1f MB -> %s", total / 1_048_576, out_path.name)
    return out_path


def geotiff_to_png16(
    tif_path: Path,
    png_path: Path,
    region_name: str,
    bbox: BBox,
    max_dim: int,
    dem_type: str = DEFAULT_DEM_TYPE,
    pre_blur_input_px: float = 0.0,
) -> dict:
    """Read a DEM GeoTIFF, resample, optionally blur, write 16-bit PNG + JSON sidecar.

    *pre_blur_input_px* gauss-blurs the resampled DEM before saving so that
    upscaled low-res DEMs (e.g. SRTMGL3 stretched to 20k) don't produce hard
    terraced steps in the climate-painted hillshade. Set to 0 to disable.
    """
    with rasterio.open(tif_path) as src:
        # If bbox is a strict subset of the GeoTIFF coverage, read only that
        # window (tile rendering reuses one big region GeoTIFF for many tiles).
        try:
            window = window_from_bounds(
                bbox.west, bbox.south, bbox.east, bbox.north, transform=src.transform
            )
            wh = max(1, int(round(window.height)))
            ww = max(1, int(round(window.width)))
        except Exception:  # noqa: BLE001
            window = None
            wh, ww = src.height, src.width

        # Pick output dims from the requested bbox aspect (not the source).
        aspect = bbox.aspect()
        if aspect >= 1.0:
            new_w = max_dim
            new_h = max(1, int(round(max_dim / aspect)))
        else:
            new_h = max_dim
            new_w = max(1, int(round(max_dim * aspect)))

        elevation = src.read(
            1,
            window=window,
            out_shape=(new_h, new_w),
            resampling=Resampling.cubic,
            boundless=True,
            fill_value=0,
        ).astype(np.float32)
        log.info("  GeoTIFF window %dx%d -> %dx%d", ww, wh, new_w, new_h)

    # Treat SRTM nodata cells as sea level; clamps "-32768" sentinel.
    elevation = np.where(elevation < SEA_NODATA_THRESHOLD, 0.0, elevation)
    # Bicubic resampling between nodata (-32768) and land introduces shallow
    # negative artefacts (e.g. -300 m at coastlines). Clamp them to sea level
    # so the saved heightmap's min stays at 0 and sea-mask thresholds work.
    np.maximum(elevation, 0.0, out=elevation)

    if pre_blur_input_px > 0:
        elevation = gaussian_filter(elevation, sigma=pre_blur_input_px)

    mn = float(elevation.min())
    mx = float(elevation.max())
    log.info("  elevation %.0f m .. %.0f m", mn, mx)

    if mx > mn:
        norm = (elevation - mn) / (mx - mn)
    else:
        norm = np.zeros_like(elevation)
    img_u16 = (norm * 65535.0).astype(np.uint16)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u16).save(png_path, "PNG", optimize=True)

    meta = {
        "region": region_name,
        "bbox": {
            "south": bbox.south, "north": bbox.north,
            "west": bbox.west, "east": bbox.east,
        },
        "elevation_m": {"min": mn, "max": mx},
        "pixel": {"width": int(img_u16.shape[1]), "height": int(img_u16.shape[0])},
        "source": {"dem": dem_type, "provider": "OpenTopography", "crs": "EPSG:4326"},
    }
    meta_path = png_path.with_name(f"{region_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
