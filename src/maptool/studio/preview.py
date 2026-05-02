"""Live map preview baked into the studio.

Generates a SYNTHETIC sample (1024×512 px) that covers every biome class
the compositor knows — central peak with snow/rock, forest ring, grass
band, coast strip, surrounding sea — so the studio is self-contained and
not coupled to any real region. The compositor then runs against the
current biome textures whenever a slider is released.

The synthetic sample lives in ``<cache>/studio_sample/`` and is rebuilt
on first launch only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from maptool.render.biome_compositor import render_with_biomes
from maptool.render.tree_sprites import render_tree_layer

log = logging.getLogger(__name__)

SAMPLE_W = 1024
SAMPLE_H = 512
SAMPLE_TILE_PX = 32        # smaller than full-map tiles -> visible texturing


def _sample_dir(cache_root: Path) -> Path:
    return cache_root / "studio_sample"


def _generate_synthetic(seed: int = 20260501) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a (heightmap_u16, codes_u8, meta_dict) covering every biome."""
    yy, xx = np.mgrid[:SAMPLE_H, :SAMPLE_W].astype(np.float32)
    cx, cy = SAMPLE_W / 2, SAMPLE_H / 2
    # Radial distance normalized so the corners are at ~1.0.
    rmax = float(np.hypot(SAMPLE_W / 2, SAMPLE_H / 2))
    r = np.hypot(xx - cx, yy - cy) / rmax

    # Gaussian peak — naturally tapers to ~0 with a smooth gradient
    # everywhere, no clipping ring like the previous linear cap created.
    elev_mtn = np.exp(-((r * 2.4) ** 2)).astype(np.float32)

    # Low-frequency noise varies the plains so we get scattered water bodies
    # and coast pockets without a hard radial boundary.
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (SAMPLE_H, SAMPLE_W)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=22.0)
    nmin, nmax = float(noise.min()), float(noise.max())
    if nmax > nmin:
        noise = (noise - nmin) / (nmax - nmin)
    elev = elev_mtn * 0.85 + noise * 0.18
    elev = np.clip(elev, 0.0, 1.0)

    height_u16 = (elev * 65535.0).astype(np.uint16)

    # Landcover classes derived from elevation. Boundaries chosen to land in
    # the same elevation bands the compositor's classify_biome cares about.
    codes = np.full((SAMPLE_H, SAMPLE_W), fill_value=30, dtype=np.uint8)
    codes[elev < 0.06] = 80                       # water
    codes[(elev >= 0.06) & (elev < 0.10)] = 60    # coast / bare strip
    codes[(elev >= 0.10) & (elev < 0.18)] = 40    # cropland
    codes[(elev >= 0.18) & (elev < 0.40)] = 30    # grass / scrub
    codes[(elev >= 0.40) & (elev < 0.66)] = 10    # forest (mountain ring)
    codes[elev >= 0.66] = 60                      # bare upper / rocky

    # Scatter forest patches across the plains via TWO noise fields:
    # a small-sigma one for many compact patches, plus a large-sigma one for
    # occasional bigger forest masses. Combined with OR so both scales appear.
    def _norm(arr: np.ndarray) -> np.ndarray:
        a, b = float(arr.min()), float(arr.max())
        if b <= a:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - a) / (b - a)).astype(np.float32, copy=False)

    forest_small = _norm(gaussian_filter(
        rng.normal(0.0, 1.0, (SAMPLE_H, SAMPLE_W)).astype(np.float32),
        sigma=10.0,
    ))
    forest_large = _norm(gaussian_filter(
        rng.normal(0.0, 1.0, (SAMPLE_H, SAMPLE_W)).astype(np.float32),
        sigma=32.0,
    ))
    forest_patches = (
        ((forest_small > 0.66) | (forest_large > 0.62))
        & ((codes == 30) | (codes == 40))
    )
    codes[forest_patches] = 10

    # Shrubland patches scattered through the plains so the bushes layer has
    # something to scatter on.
    shrub_noise = _norm(gaussian_filter(
        rng.normal(0.0, 1.0, (SAMPLE_H, SAMPLE_W)).astype(np.float32),
        sigma=14.0,
    ))
    shrub_patches = (shrub_noise > 0.58) & ((codes == 30) | (codes == 40))
    codes[shrub_patches] = 20

    # Latitude range chosen mid-Mediterranean so the lat-aware classifier
    # behaves typically (forest preferred over arid in northern half).
    meta = {
        "region": "studio_sample",
        "bbox": {"south": 36.0, "north": 44.0, "west": 0.0, "east": 16.0},
        "elevation_m": {"min": 0.0, "max": 3000.0},
        "pixel": {"width": SAMPLE_W, "height": SAMPLE_H},
        "source": {"dem": "synthetic", "provider": "studio", "crs": "EPSG:4326"},
    }
    return height_u16, codes, meta


def _ensure_sample(cache_root: Path) -> tuple[Path, Path, Path]:
    sample_dir = _sample_dir(cache_root)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_height = sample_dir / "height.png"
    sample_codes = sample_dir / "codes.png"
    sample_meta = sample_dir / "meta.json"

    if sample_height.exists() and sample_codes.exists() and sample_meta.exists():
        return sample_height, sample_codes, sample_meta

    height_u16, codes, meta = _generate_synthetic()
    Image.fromarray(height_u16).save(sample_height, "PNG", optimize=True)
    Image.fromarray(codes, mode="L").save(sample_codes, "PNG", optimize=True)
    sample_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("studio synthetic sample baked -> %s", sample_dir)
    return sample_height, sample_codes, sample_meta


def render_sample(cache_root: Path, *, tile_px: int = SAMPLE_TILE_PX) -> Path:
    """Compose the synthetic sample with current biome textures + tree sprites."""
    sample_height, sample_codes, sample_meta = _ensure_sample(cache_root)
    out_path = _sample_dir(cache_root) / "preview.png"
    render_with_biomes(
        sample_height, sample_codes, sample_meta, out_path,
        cache_root=cache_root, tile_px=tile_px,
    )
    # Overlay tree sprites where landcover == forest. Pass heightmap + meta
    # so the species is picked by latitude and trees stay off cliffs.
    render_tree_layer(
        out_path, sample_codes, out_path, cache_root,
        height_png=sample_height, meta_path=sample_meta,
    )
    return out_path
