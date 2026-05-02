"""Optional vector overlay: rivers as thin lines on top of a climate-painted preview.

No coastline overlay — the coast comes from terrain contrast + the sandy beach
band in ``climate_paint.py`` (Memory: feedback_aquila_reference — no line overlays).

If no GeoJSON exists for a region, the annotated PNG is just a copy of the
color preview.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from maptool.regions import BBox

log = logging.getLogger(__name__)

RIVER_COLOR = (55, 75, 100, 235)
RIVER_WIDTH = 2


def _latlon_to_pixel(lon: float, lat: float, bbox: BBox, w: int, h: int) -> tuple[float, float]:
    x = (lon - bbox.west) / bbox.lon_span() * w
    y = (bbox.north - lat) / bbox.lat_span() * h
    return x, y


def _iter_linestrings(features: list[dict]) -> Iterable[list]:
    for feat in features:
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "LineString" and coords:
            yield coords
        elif gtype == "MultiLineString" and coords:
            yield from coords


def _load_features(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("features", [])


def annotate(
    color_png: Path,
    out_png: Path,
    bbox: BBox,
    rivers_geojson: Path | None = None,
) -> Path:
    """Draw rivers (if provided) onto *color_png*, write to *out_png*."""
    img = Image.open(color_png).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    n_pts = 0
    if rivers_geojson is not None:
        rivers = _load_features(rivers_geojson)
        for ls in _iter_linestrings(rivers):
            pixels = [_latlon_to_pixel(p[0], p[1], bbox, img.width, img.height) for p in ls]
            if len(pixels) >= 2:
                draw.line(pixels, fill=RIVER_COLOR, width=RIVER_WIDTH)
                n_pts += len(pixels)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_png, "PNG", optimize=True)
    log.info("  annotated rivers=%d pts -> %s", n_pts, out_png.name)
    return out_png
