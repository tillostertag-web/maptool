"""Generate biome textures via fal.ai and cache them on disk.

Cache layout:
    <cache_dir>/biomes/<slug>.png         the texture itself
    <cache_dir>/biomes/<slug>.json        prompt + seed + timestamp metadata

If the texture file exists and the metadata matches the requested prompt,
*generate_biome* is a no-op. Pass ``force=True`` to re-generate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from maptool.assets.biomes import Biome, CATALOG_STANDARD
from maptool.assets.fal_client import download_image, generate_image

log = logging.getLogger(__name__)


DEFAULT_SEED = 20260501


@dataclass(frozen=True)
class BiomeAsset:
    biome: Biome
    image_path: Path
    meta_path: Path


def biomes_dir(cache_root: Path) -> Path:
    return cache_root / "biomes"


def asset_paths(cache_root: Path, biome: Biome) -> BiomeAsset:
    base = biomes_dir(cache_root)
    return BiomeAsset(
        biome=biome,
        image_path=base / f"{biome.slug}.png",
        meta_path=base / f"{biome.slug}.json",
    )


def _read_meta(asset: BiomeAsset) -> dict | None:
    if not asset.meta_path.exists():
        return None
    try:
        with open(asset.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def is_cached(asset: BiomeAsset, expected_prompt: str) -> bool:
    if not asset.image_path.exists():
        return False
    meta = _read_meta(asset)
    if meta is None:
        return False
    return meta.get("prompt") == expected_prompt


def generate_biome(
    biome: Biome,
    cache_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> BiomeAsset:
    """Generate (or reuse) a tileable texture for *biome* under *cache_root*."""
    asset = asset_paths(cache_root, biome)
    prompt = biome.full_prompt()
    if not force and is_cached(asset, prompt):
        log.info("biome %s cached -> %s", biome.slug, asset.image_path)
        return asset

    log.info("biome %s: generating ...", biome.slug)
    result = generate_image(prompt, seed=seed)
    download_image(result, asset.image_path)

    asset.meta_path.write_text(
        json.dumps(
            {
                "slug": biome.slug,
                "prompt": prompt,
                "seed": seed,
                "model": result.get("model") if isinstance(result, dict) else None,
                "request": {"image_size": "square_hd"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("biome %s ready  -> %s", biome.slug, asset.image_path)
    return asset


def generate_catalog(
    catalog: tuple[Biome, ...],
    cache_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> list[BiomeAsset]:
    return [generate_biome(b, cache_root, seed=seed, force=force) for b in catalog]


def list_cached(cache_root: Path, catalog: tuple[Biome, ...] = CATALOG_STANDARD) -> list[tuple[Biome, bool]]:
    """Returns [(biome, cached?)] for every biome in the catalog."""
    out = []
    for b in catalog:
        a = asset_paths(cache_root, b)
        out.append((b, is_cached(a, b.full_prompt())))
    return out
