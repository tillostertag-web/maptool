"""maptool CLI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from maptool import __version__
from maptool.assets.biomes import CATALOG_STANDARD, by_slug
from maptool.assets.generator import generate_biome, generate_catalog, list_cached
from maptool.gallery import write_gallery
from maptool.pipeline import (
    DEFAULT_MAX_DIM,
    DEFAULT_PAINT_MODE,
    PAINT_MODES,
    Layout,
    build_all,
    render,
)
from maptool.render.biome_compositor import DEFAULT_TILE_PX
from maptool.regions import REGIONS, BBox, parse_bbox, subdivide_into_tiles
from maptool.sources.srtm import DEFAULT_DEM_TYPE, SUPPORTED_DEMS

log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO_ROOT / "out"
DEFAULT_CACHE = REPO_ROOT / ".cache"

QUALITY_PRESETS: dict[str, dict] = {
    "standard": {"max_dim": 4096,  "dem": "SRTMGL3"},  # fastest, ~30s/region
    "high":     {"max_dim": 10240, "dem": "SRTMGL1"},  # default recommendation
    "ultra":    {"max_dim": 16384, "dem": "SRTMGL1"},  # showcase, ~4-6 min, ~130 MP
}
SAFE_MAX_DIM = 24576  # protect against accidental 10x renders that OOM


def _apply_quality(args: argparse.Namespace) -> None:
    """Quality preset overrides explicit --max-dim/--dem only if user kept defaults."""
    if not getattr(args, "quality", None):
        return
    preset = QUALITY_PRESETS[args.quality]
    if args.max_dim == DEFAULT_MAX_DIM:
        args.max_dim = preset["max_dim"]
    if getattr(args, "dem", None) == DEFAULT_DEM_TYPE:
        args.dem = preset["dem"]


def _resolve_target(args: argparse.Namespace) -> tuple[str, BBox]:
    if args.bbox is not None:
        bbox = parse_bbox(args.bbox)
        name = args.name or "custom"
        return name, bbox
    if args.region is not None:
        return args.region, REGIONS[args.region]
    raise SystemExit("Specify --region or --bbox.")


def _api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENTOPO_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENTOPO_API_KEY missing. Copy .env.example to .env and paste the key.\n"
            "Get a free key at https://portal.opentopography.org/myopentopo"
        )
    return key


def _layout(name: str, args: argparse.Namespace) -> Layout:
    out_dir = Path(args.out_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    dem = getattr(args, "dem", DEFAULT_DEM_TYPE)
    return Layout(out_dir=out_dir, cache_dir=cache_dir, name=name, dem_type=dem)


def _validate_max_dim(max_dim: int) -> int:
    if max_dim <= 0:
        raise SystemExit(f"--max-dim must be positive, got {max_dim}")
    if max_dim > SAFE_MAX_DIM:
        log.warning("--max-dim %d exceeds safe cap %d; clamping. Use bigger machines for >24k.",
                    max_dim, SAFE_MAX_DIM)
        return SAFE_MAX_DIM
    return max_dim


def cmd_build(args: argparse.Namespace) -> int:
    _apply_quality(args)
    args.max_dim = _validate_max_dim(args.max_dim)
    api_key = _api_key()
    targets: list[tuple[str, BBox]]
    if args.all:
        targets = list(REGIONS.items())
    else:
        targets = [_resolve_target(args)]

    for name, bbox in targets:
        layout = _layout(name, args)
        build_all(
            name, bbox, layout, api_key,
            max_dim=args.max_dim,
            force=args.force,
            skip_landcover=args.skip_landcover,
            with_forest=not args.no_forest,
            paint_mode=args.paint,
            tile_px=args.tile_px,
            with_osm=args.osm,
        )
    write_gallery(Path(args.out_dir).resolve())
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    _apply_quality(args)
    args.max_dim = _validate_max_dim(args.max_dim)
    targets: list[tuple[str, BBox]]
    if args.all:
        targets = list(REGIONS.items())
    else:
        targets = [_resolve_target(args)]

    for name, bbox in targets:
        layout = _layout(name, args)
        render(
            name, bbox, layout,
            with_forest=not args.no_forest,
            paint_mode=args.paint,
            tile_px=args.tile_px,
            with_osm=args.osm,
        )
    write_gallery(Path(args.out_dir).resolve())
    return 0


def cmd_gallery(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    path = write_gallery(out_dir)
    if args.open:
        webbrowser.open(path.as_uri())
    return 0


def cmd_list_regions(_args: argparse.Namespace) -> int:
    name_w = max(len(n) for n in REGIONS)
    for name, bb in REGIONS.items():
        print(f"  {name:<{name_w}}  S={bb.south:>6.2f} N={bb.north:>6.2f} "
              f"W={bb.west:>7.2f} E={bb.east:>7.2f}")
    return 0


def _render_one_to_png(
    name: str,
    bbox: BBox,
    cache_name: str,
    cache_dir: Path,
    out_png: Path,
    api_key: str,
    dem_type: str,
    max_dim: int,
) -> None:
    """Run the full pipeline for *bbox* into a temp dir, then copy the final
    forest PNG (or color if no forest) to *out_png*."""
    with tempfile.TemporaryDirectory(prefix="maptool_tile_") as tmp:
        tmp_path = Path(tmp)
        layout = Layout(
            out_dir=tmp_path,
            cache_dir=cache_dir,
            name=name,
            dem_type=dem_type,
            cache_name=cache_name,
        )
        build_all(name, bbox, layout, api_key, max_dim=max_dim, with_forest=True)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        if layout.forest.exists():
            shutil.copy(layout.forest, out_png)
        elif layout.color.exists():
            shutil.copy(layout.color, out_png)
        else:
            raise RuntimeError(f"render produced no output for {name}")


def cmd_tile(args: argparse.Namespace) -> int:
    api_key = _api_key()
    targets: list[tuple[str, BBox]]
    if args.all:
        targets = list(REGIONS.items())
    else:
        targets = [_resolve_target(args)]

    out_root = Path(args.out_dir).resolve()
    dem = getattr(args, "dem", DEFAULT_DEM_TYPE)
    overview_max = _validate_max_dim(args.overview_max_dim)
    tile_max = _validate_max_dim(args.tile_size)

    for region_name, bbox in targets:
        region_dir = out_root / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        log.info("=== TILE PYRAMID: %s ===", region_name)
        log.info("BBox: S=%.2f N=%.2f W=%.2f E=%.2f  tile_deg=%.2f  tile_size=%d",
                 bbox.south, bbox.north, bbox.west, bbox.east, args.tile_deg, tile_max)

        # 1. Overview render: whole bbox at overview_max-dim.
        overview_png = region_dir / "overview.png"
        log.info("Overview render: %s", overview_png.name)
        _render_one_to_png(
            name=region_name, bbox=bbox, cache_name=region_name,
            cache_dir=Path(args.cache_dir).resolve(),
            out_png=overview_png, api_key=api_key, dem_type=dem,
            max_dim=overview_max,
        )

        # 2. Native-density tiles, each *tile_deg* x *tile_deg* at tile_size px.
        tiles = subdivide_into_tiles(bbox, args.tile_deg)
        log.info("Tiles: %d (tile_deg=%.2f)", len(tiles), args.tile_deg)

        manifest = {
            "region": region_name,
            "bbox": {"south": bbox.south, "north": bbox.north,
                     "west": bbox.west, "east": bbox.east},
            "dem": dem,
            "tile_deg": args.tile_deg,
            "tile_size": tile_max,
            "overview_max_dim": overview_max,
            "overview": "overview.png",
            "tiles": [],
        }
        for tx, ty, tile_bbox in tiles:
            tile_name = f"tile_{tx}_{ty}.png"
            tile_path = region_dir / tile_name
            log.info("Tile %d/%d  tx=%d ty=%d  S=%.2f N=%.2f W=%.2f E=%.2f",
                     len(manifest["tiles"]) + 1, len(tiles), tx, ty,
                     tile_bbox.south, tile_bbox.north, tile_bbox.west, tile_bbox.east)
            _render_one_to_png(
                name=f"{region_name}_t{tx}_{ty}",
                bbox=tile_bbox,
                cache_name=region_name,
                cache_dir=Path(args.cache_dir).resolve(),
                out_png=tile_path,
                api_key=api_key,
                dem_type=dem,
                max_dim=tile_max,
            )
            manifest["tiles"].append({
                "tx": tx, "ty": ty,
                "file": tile_name,
                "bbox": {"south": tile_bbox.south, "north": tile_bbox.north,
                         "west": tile_bbox.west, "east": tile_bbox.east},
            })

        manifest_path = region_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info("Manifest -> %s", manifest_path)
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")
    from maptool.studio.app import launch
    cache_root = Path(args.cache_dir).resolve()
    log.info("Launching biome studio at http://%s:%d ...", args.host, args.port)
    launch(cache_root, host=args.host, port=args.port, share=args.share)
    return 0


def cmd_assets_generate(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")
    cache_root = Path(args.cache_dir).resolve()
    if args.biome:
        catalog = by_slug(CATALOG_STANDARD)
        if args.biome not in catalog:
            raise SystemExit(f"unknown biome: {args.biome!r}; "
                             f"choose one of {list(catalog)}")
        generate_biome(catalog[args.biome], cache_root, seed=args.seed, force=args.force)
    else:
        generate_catalog(CATALOG_STANDARD, cache_root, seed=args.seed, force=args.force)
    return 0


def cmd_assets_list(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_dir).resolve()
    print(f"biome cache: {cache_root / 'biomes'}")
    for biome, cached in list_cached(cache_root):
        marker = "✓" if cached else "·"
        print(f"  {marker} {biome.slug:<18}  {biome.description}")
    return 0


def cmd_cache_import(args: argparse.Namespace) -> int:
    """Copy <region>_<dem>.tif files from a source directory into our cache."""
    src = Path(args.from_dir).expanduser().resolve()
    dst = Path(args.cache_dir).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"--from must be a directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    found = sorted(src.glob("*.tif"))
    if not found:
        log.warning("No .tif files in %s", src)
        return 0

    n_copied = 0
    for tif in found:
        # Match the maptool naming convention: <region>_<demtype>.tif
        # If the source file already follows this, keep it; otherwise skip with warning.
        stem = tif.stem
        if "_" not in stem:
            log.warning("skip %s (filename must be <region>_<demtype>.tif)", tif.name)
            continue
        target = dst / tif.name
        if target.exists() and not args.overwrite:
            log.info("skip %s (exists; use --overwrite to replace)", tif.name)
            continue
        shutil.copy2(tif, target)
        log.info("copied %s -> %s (%.1f MB)", tif.name, target,
                 target.stat().st_size / 1_048_576)
        n_copied += 1

    log.info("Cache import: %d file(s) copied into %s", n_copied, dst)
    return 0


def _add_target_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--region", choices=list(REGIONS.keys()))
    g.add_argument("--bbox", help="south,north,west,east in decimal degrees")
    g.add_argument("--all", action="store_true", help="run for every built-in region")
    p.add_argument("--name", help="override output name (only with --bbox)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT), help=f"default: {DEFAULT_OUT}")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help=f"default: {DEFAULT_CACHE}")
    p.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM,
                   help=f"max output pixel dimension (default {DEFAULT_MAX_DIM})")
    p.add_argument("--dem", choices=SUPPORTED_DEMS, default=DEFAULT_DEM_TYPE,
                   help=f"DEM source (default {DEFAULT_DEM_TYPE})")
    p.add_argument("--quality", choices=list(QUALITY_PRESETS),
                   help="preset bundling --max-dim and --dem (overrides only when those are at defaults)")
    p.add_argument("--paint", choices=list(PAINT_MODES), default=DEFAULT_PAINT_MODE,
                   help=f"painting backend: 'biome' tiles edited textures, "
                        f"'climate' uses the procedural climate paint (default {DEFAULT_PAINT_MODE})")
    p.add_argument("--tile-px", type=int, default=DEFAULT_TILE_PX,
                   help=f"biome texture tile size in canvas pixels (default {DEFAULT_TILE_PX})")
    p.add_argument("--osm", action="store_true",
                   help="add OSM overlays (rivers / roads / settlements; needs network)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="maptool", description="Climate-painted terrain map builder")
    parser.add_argument("--version", action="version", version=f"maptool {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: info, -vv: debug")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="download DEM + landcover, render preview, write gallery")
    _add_target_args(p_build)
    p_build.add_argument("--force", action="store_true", help="re-download even if cached")
    p_build.add_argument("--skip-landcover", action="store_true",
                         help="skip the WorldCover step")
    p_build.add_argument("--no-forest", action="store_true",
                         help="skip the procedural forest sprite layer")
    p_build.set_defaults(func=cmd_build)

    p_render = sub.add_parser("render", help="re-paint preview from existing heightmap (no download)")
    _add_target_args(p_render)
    p_render.add_argument("--no-forest", action="store_true",
                          help="skip the procedural forest sprite layer")
    p_render.set_defaults(func=cmd_render)

    p_gal = sub.add_parser("gallery", help="rebuild gallery.html from out/ contents")
    p_gal.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p_gal.add_argument("--open", action="store_true", help="open gallery in browser")
    p_gal.set_defaults(func=cmd_gallery)

    p_list = sub.add_parser("list-regions", help="show built-in regions")
    p_list.set_defaults(func=cmd_list_regions)

    p_tile = sub.add_parser("tile",
                            help="render an overview + grid of native-density tiles for a region")
    g_tile = p_tile.add_mutually_exclusive_group()
    g_tile.add_argument("--region", choices=list(REGIONS.keys()))
    g_tile.add_argument("--bbox", help="south,north,west,east in decimal degrees")
    g_tile.add_argument("--all", action="store_true", help="run for every built-in region")
    p_tile.add_argument("--name", help="output sub-directory name (only with --bbox)")
    p_tile.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p_tile.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p_tile.add_argument("--dem", choices=SUPPORTED_DEMS, default="SRTMGL1")
    p_tile.add_argument("--tile-deg", type=float, default=1.0,
                        help="tile coverage in degrees (default 1.0 -> ~111 km)")
    p_tile.add_argument("--tile-size", type=int, default=4096,
                        help="pixel width per tile (default 4096)")
    p_tile.add_argument("--overview-max-dim", type=int, default=4096,
                        help="pixel width for the whole-region overview tile (default 4096)")
    p_tile.set_defaults(func=cmd_tile)

    p_studio = sub.add_parser("studio",
                              help="launch the live biome studio (Gradio web UI)")
    p_studio.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p_studio.add_argument("--host", default="127.0.0.1")
    p_studio.add_argument("--port", type=int, default=7860)
    p_studio.add_argument("--share", action="store_true",
                          help="expose the UI on a public Gradio share URL")
    p_studio.set_defaults(func=cmd_studio)

    p_gen = sub.add_parser("assets-generate",
                           help="generate biome textures via fal.ai (cached locally)")
    p_gen.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p_gen.add_argument("--biome", help="generate only one biome (slug)")
    p_gen.add_argument("--seed", type=int, default=20260501,
                       help="seed for deterministic generation (default 20260501)")
    p_gen.add_argument("--force", action="store_true",
                       help="re-generate even if a cached texture exists with the same prompt")
    p_gen.set_defaults(func=cmd_assets_generate)

    p_assets_list = sub.add_parser("assets-list", help="list cached biome textures")
    p_assets_list.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p_assets_list.set_defaults(func=cmd_assets_list)

    p_cache = sub.add_parser("cache-import",
                             help="import <region>_<demtype>.tif files into the local cache")
    p_cache.add_argument("--from", dest="from_dir", required=True,
                         help="source directory (e.g. aquila/assets/terrain/raw)")
    p_cache.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p_cache.add_argument("--overwrite", action="store_true",
                         help="replace existing cache files with the same name")
    p_cache.set_defaults(func=cmd_cache_import)

    args = parser.parse_args(argv)
    level = logging.WARNING if args.verbose == 0 else logging.INFO if args.verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(message)s")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
