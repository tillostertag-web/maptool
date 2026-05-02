"""Post-process operations for the biome studio.

These transform an already-generated PNG without calling fal.ai again:

* ``adjust_colors`` shifts hue and scales saturation / brightness via PIL.
* ``tile_preview`` arranges the texture in a NxN grid so the user can see
  whether the material tiles cleanly when sampled across a larger area.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance


def adjust_colors(
    src: Path | Image.Image,
    *,
    hue_shift_deg: float = 0.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
) -> Image.Image:
    """Return a new image with HSV shifted (degrees / multipliers).

    *hue_shift_deg* in [-180, 180]. *saturation* and *brightness* are
    multipliers (1.0 = no change, 0 = greyscale / black).
    """
    img = Image.open(src) if isinstance(src, Path) else src
    rgb = img.convert("RGB")

    if hue_shift_deg != 0.0:
        hsv = rgb.convert("HSV")
        h, s, v = hsv.split()
        shift = int(round((hue_shift_deg / 360.0) * 256.0)) % 256
        h = h.point(lambda p, shift=shift: (p + shift) % 256)
        rgb = Image.merge("HSV", (h, s, v)).convert("RGB")

    if saturation != 1.0:
        rgb = ImageEnhance.Color(rgb).enhance(float(saturation))
    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(rgb).enhance(float(brightness))

    return rgb


def tile_preview(
    src: Path | Image.Image,
    *,
    grid: int = 3,
    out_size: int = 768,
) -> Image.Image:
    """Render the texture as a *grid* x *grid* tiled preview at *out_size* px.

    The cell size adapts so the whole grid fits *out_size*. The user can spot
    seams (visible grid lines on every tile boundary) at a glance.
    """
    img = Image.open(src) if isinstance(src, Path) else src
    rgb = img.convert("RGB")
    cell = max(1, out_size // grid)
    tile = rgb.resize((cell, cell), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (cell * grid, cell * grid))
    for ty in range(grid):
        for tx in range(grid):
            canvas.paste(tile, (tx * cell, ty * cell))
    return canvas


def save_adjusted_to(src: Path, dst: Path, **kwargs) -> Path:
    """Apply ``adjust_colors`` and write to *dst*."""
    out = adjust_colors(src, **kwargs)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, optimize=True)
    return dst
