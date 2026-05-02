"""Vector overlay renderer — draws OSM linestrings and points on top of the
painted map. Anti-aliased via 2x super-sampling.

Used for:
* rivers (blue thin lines)
* roads (tan curving lines)
* settlements (small circle marker + name label)

Each call returns a fresh PIL.Image; the caller composites it onto the base.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from maptool.regions import BBox
from maptool.sources.osm import Linestring, PointFeature

log = logging.getLogger(__name__)


# Render styles -------------------------------------------------------------
RIVER_COLOR = (78, 116, 152, 235)
RIVER_WIDTH = 3                  # pixels at 1x — supersampled 2x

ROAD_COLOR = (188, 158, 122, 215)
ROAD_WIDTH = 2

SETTLEMENT_FILL = (240, 230, 210, 255)
SETTLEMENT_OUTLINE = (40, 30, 20, 255)
SETTLEMENT_RADIUS = 5
LABEL_COLOR = (32, 26, 22, 255)
LABEL_HALO = (245, 240, 228, 235)
LABEL_OFFSET_X = 8
LABEL_OFFSET_Y = -4


def latlon_to_pixel(lon: float, lat: float, bbox: BBox, w: int, h: int) -> tuple[float, float]:
    """Equirectangular projection to pixel space."""
    x = (lon - bbox.west) / bbox.lon_span() * w
    y = (bbox.north - lat) / bbox.lat_span() * h
    return x, y


def _project(coords, bbox: BBox, w: int, h: int, scale: float) -> list[tuple[float, float]]:
    out = []
    for lon, lat in coords:
        x = (lon - bbox.west) / bbox.lon_span() * w * scale
        y = (bbox.north - lat) / bbox.lat_span() * h * scale
        out.append((x, y))
    return out


def _new_overlay(w: int, h: int, scale: int = 2) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    """Allocate a transparent supersampled overlay image."""
    over = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(over, "RGBA")
    return over, draw, scale


def _downsample(over: Image.Image, w: int, h: int) -> Image.Image:
    return over.resize((w, h), Image.Resampling.LANCZOS)


def draw_lines(
    canvas: Image.Image,
    lines: list[Linestring],
    bbox: BBox,
    *,
    color: tuple[int, int, int, int],
    width: int,
) -> Image.Image:
    """Composite anti-aliased polylines on top of *canvas* (in place, returns canvas)."""
    if not lines:
        return canvas
    w, h = canvas.size
    over, draw, s = _new_overlay(w, h)
    for ls in lines:
        pts = _project(ls.coords, bbox, w, h, s)
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=max(1, width * s), joint="curve")
    overlay_lr = _downsample(over, w, h)
    canvas.alpha_composite(overlay_lr)
    return canvas


def draw_settlements(
    canvas: Image.Image,
    points: list[PointFeature],
    bbox: BBox,
    *,
    font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    """Draw small markers + name labels for each point feature."""
    if not points:
        return canvas
    w, h = canvas.size
    over, draw, s = _new_overlay(w, h)
    radius = SETTLEMENT_RADIUS * s
    for p in points:
        cx, cy = latlon_to_pixel(p.lon, p.lat, bbox, w, h)
        cx *= s
        cy *= s
        # Outer ring + inner fill -> simple settlement icon.
        draw.ellipse(
            (cx - radius - s, cy - radius - s, cx + radius + s, cy + radius + s),
            fill=SETTLEMENT_OUTLINE,
        )
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=SETTLEMENT_FILL,
        )

    overlay_lr = _downsample(over, w, h)
    canvas.alpha_composite(overlay_lr)

    # Labels: drawn at native resolution (text rendering is its own AA).
    if font is None:
        font = _default_font(size=14)
    label_draw = ImageDraw.Draw(canvas, "RGBA")
    for p in points:
        cx, cy = latlon_to_pixel(p.lon, p.lat, bbox, w, h)
        name = p.tags.get("name", "")
        if not name:
            continue
        tx = cx + LABEL_OFFSET_X
        ty = cy + LABEL_OFFSET_Y
        # Halo (1-px ring around text) for legibility on busy biome textures.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                label_draw.text((tx + dx, ty + dy), name, fill=LABEL_HALO, font=font)
        label_draw.text((tx, ty), name, fill=LABEL_COLOR, font=font)
    return canvas


def _default_font(size: int = 14) -> ImageFont.ImageFont:
    """Fall back through common system fonts so the studio still renders if
    the project doesn't ship its own TTF yet."""
    candidates = [
        "C:/Windows/Fonts/Georgia.ttf",
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/Library/Fonts/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_overlays(
    canvas_path: Path,
    out_path: Path,
    bbox: BBox,
    cache_dir: Path,
    *,
    rivers: bool = True,
    roads: bool = True,
    settlements: bool = True,
) -> Path:
    """Read *canvas_path*, fetch OSM, draw layers, write to *out_path*."""
    from maptool.sources.osm import fetch_rivers, fetch_roads, fetch_settlements

    img = Image.open(canvas_path).convert("RGBA")

    if roads:
        rds = fetch_roads(bbox, cache_dir)
        log.info("  roads: %d ways", len(rds))
        draw_lines(img, rds, bbox, color=ROAD_COLOR, width=ROAD_WIDTH)
    if rivers:
        rvs = fetch_rivers(bbox, cache_dir)
        log.info("  rivers: %d ways", len(rvs))
        draw_lines(img, rvs, bbox, color=RIVER_COLOR, width=RIVER_WIDTH)
    if settlements:
        sts = fetch_settlements(bbox, cache_dir)
        log.info("  settlements: %d nodes", len(sts))
        draw_settlements(img, sts, bbox)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, "PNG", optimize=True)
    return out_path
