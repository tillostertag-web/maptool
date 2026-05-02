"""End-to-end orchestration: heightmap + landcover + paint + annotate + thumbs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from maptool.regions import BBox
from maptool.render.annotate import annotate
from maptool.render.biome_compositor import DEFAULT_TILE_PX, render_with_biomes
from maptool.render.climate_paint import render_preview
from maptool.render.forest import render_forest
from maptool.render.tree_sprites import render_tree_layer
from maptool.sources import srtm, worldcover
from maptool.sources.srtm import DEFAULT_DEM_TYPE

PAINT_MODES = ("biome", "climate")
DEFAULT_PAINT_MODE = "biome"

log = logging.getLogger(__name__)

THUMB_MAX = 256
DEFAULT_MAX_DIM = 4096


@dataclass(frozen=True)
class Layout:
    """Where on disk a region's artifacts live."""
    out_dir: Path
    cache_dir: Path
    name: str
    dem_type: str = DEFAULT_DEM_TYPE
    cache_name: str | None = None  # if set, srtm_tif uses this instead of *name*

    @property
    def height_png(self) -> Path:
        return self.out_dir / f"{self.name}_height.png"

    @property
    def height_meta(self) -> Path:
        return self.out_dir / f"{self.name}_meta.json"

    @property
    def lc_codes(self) -> Path:
        return self.out_dir / f"{self.name}_landcover_codes.png"

    @property
    def lc_color(self) -> Path:
        return self.out_dir / f"{self.name}_landcover.png"

    @property
    def color(self) -> Path:
        return self.out_dir / f"{self.name}_color.png"

    @property
    def forest(self) -> Path:
        return self.out_dir / f"{self.name}_forest.png"

    @property
    def annotated(self) -> Path:
        return self.out_dir / f"{self.name}_annotated.png"

    @property
    def previews(self) -> Path:
        return self.out_dir / "previews"

    @property
    def rivers_geojson(self) -> Path:
        return self.out_dir / "vectors" / f"{self.name}_rivers.geojson"

    @property
    def srtm_tif(self) -> Path:
        # Filename matches the aquila project so its existing GeoTIFFs can be
        # dropped into our cache directly. Tile-scoped layouts override
        # *cache_name* so all tiles in a region share one cached GeoTIFF.
        stem = self.cache_name or self.name
        return self.cache_dir / f"{stem}_{self.dem_type.lower()}.tif"


def _write_thumb(src: Path, dst: Path, max_dim: int = THUMB_MAX) -> None:
    """In-place thumbnail keeps peak memory bounded at ultra resolution.

    Opening a 200 MP PNG, calling .copy(), then .resize() roughly triples
    memory (~2.4 GB held simultaneously). Image.thumbnail() resizes in place
    so we only ever hold one buffer.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.draft(None, (max_dim * 4, max_dim * 4))  # hint decoders to subsample
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        img.save(dst, "PNG", optimize=True)


def _pre_blur_for(layout: Layout, max_dim: int) -> float:
    """Pick a small blur in input-DEM pixels so upscaled DEMs don't terrace.

    GL3 (90m) needs a small blur on heavy upscale or stair-steps show through.
    GL1 (30m) is dense enough that bicubic upscale alone is smooth — set to 0
    so the heightmap stays as crisp as the source allows.
    """
    if layout.dem_type == "SRTMGL3" and max_dim > 6000:
        return 0.7
    return 0.0


def build_heightmap(
    name: str,
    bbox: BBox,
    layout: Layout,
    api_key: str,
    max_dim: int = DEFAULT_MAX_DIM,
    force: bool = False,
) -> dict:
    if layout.srtm_tif.exists() and not force:
        log.info("DEM cached: %s", layout.srtm_tif.name)
    else:
        srtm.download_dem(bbox, api_key, layout.srtm_tif, dem_type=layout.dem_type)
    return srtm.geotiff_to_png16(
        layout.srtm_tif, layout.height_png, name, bbox, max_dim,
        dem_type=layout.dem_type,
        pre_blur_input_px=_pre_blur_for(layout, max_dim),
    )


def build_landcover(name: str, bbox: BBox, layout: Layout, max_dim: int = DEFAULT_MAX_DIM) -> None:
    log.info("Landcover: %s", name)
    mosaic = worldcover.build_mosaic(bbox, max_dim)
    worldcover.write_landcover(mosaic, layout.lc_codes, layout.lc_color)
    for cls, pct in worldcover.class_distribution(mosaic):
        log.info("  class %3d  %5.1f%%", cls, pct)
    _write_thumb(layout.lc_color, layout.previews / f"{name}_landcover_thumb.png")


def render(
    name: str,
    bbox: BBox,
    layout: Layout,
    *,
    with_forest: bool = True,
    paint_mode: str = DEFAULT_PAINT_MODE,
    tile_px: int = DEFAULT_TILE_PX,
) -> None:
    log.info("Render: %s  paint=%s  tile_px=%d", name, paint_mode, tile_px)
    if not layout.height_png.exists() or not layout.height_meta.exists():
        raise FileNotFoundError(
            f"Missing {layout.height_png.name} or {layout.height_meta.name}; "
            f"run `maptool build` first."
        )
    if paint_mode == "biome":
        if not layout.lc_codes.exists():
            raise FileNotFoundError(
                f"Biome paint needs {layout.lc_codes.name}; run `build` first or use --paint climate."
            )
        render_with_biomes(
            layout.height_png, layout.lc_codes, layout.height_meta, layout.color,
            cache_root=layout.cache_dir, tile_px=tile_px,
        )
    elif paint_mode == "climate":
        render_preview(layout.height_png, layout.height_meta, layout.color)
    else:
        raise ValueError(f"unknown paint_mode {paint_mode!r}")

    # Forest layer: in 'biome' mode we use the new image-sprite scatter
    # (loaded from <cache>/sprites/trees/), in 'climate' mode the older
    # procedural forest.py with mass underlay + per-sprite tinting.
    base_for_overlay = layout.color
    if with_forest and layout.lc_codes.exists():
        if paint_mode == "biome":
            render_tree_layer(
                layout.color, layout.lc_codes, layout.forest, layout.cache_dir,
                height_png=layout.height_png, meta_path=layout.height_meta,
            )
        else:
            render_forest(layout.color, layout.lc_codes, layout.height_meta, layout.forest)
        base_for_overlay = layout.forest
        _write_thumb(layout.forest, layout.previews / f"{name}_forest_thumb.png")
    elif with_forest:
        log.info("  forest skipped: %s missing", layout.lc_codes.name)

    rivers_path = layout.rivers_geojson if layout.rivers_geojson.exists() else None
    if rivers_path is not None:
        annotate(base_for_overlay, layout.annotated, bbox, rivers_geojson=rivers_path)
        _write_thumb(layout.annotated, layout.previews / f"{name}_annotated_thumb.png")
    else:
        # No rivers to overlay -> annotated would just duplicate forest/color.
        # Skip the extra ~125 MB write at ultra resolution.
        log.info("  annotate skipped (no rivers geojson)")
    _write_thumb(layout.color, layout.previews / f"{name}_thumb.png")


def build_all(
    name: str,
    bbox: BBox,
    layout: Layout,
    api_key: str,
    max_dim: int = DEFAULT_MAX_DIM,
    force: bool = False,
    skip_landcover: bool = False,
    with_forest: bool = True,
    paint_mode: str = DEFAULT_PAINT_MODE,
    tile_px: int = DEFAULT_TILE_PX,
) -> None:
    """Full pipeline for one region."""
    log.info("=== %s ===", name)
    log.info("BBox: S=%.2f N=%.2f W=%.2f E=%.2f  DEM=%s  max_dim=%d",
             bbox.south, bbox.north, bbox.west, bbox.east, layout.dem_type, max_dim)
    build_heightmap(name, bbox, layout, api_key, max_dim, force)
    if not skip_landcover:
        build_landcover(name, bbox, layout, max_dim)
    render(name, bbox, layout, with_forest=with_forest, paint_mode=paint_mode, tile_px=tile_px)
    log.info("Done: %s", name)
