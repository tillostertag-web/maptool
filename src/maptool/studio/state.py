"""Mutable working state for the procedural biome studio.

Each biome's editable parameters live in ``<cache>/biome_params.json`` and
the rendered preview PNG in ``<cache>/biomes/<slug>.png``. The studio
loads/saves both so a session restart restores your edits, and so the
regular ``maptool build`` pipeline reads the latest texture automatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from maptool.assets.biomes import CATALOG_STANDARD, Biome
from maptool.assets.procedural import DEFAULT_PARAMS, BiomeParams

log = logging.getLogger(__name__)


@dataclass
class StudioState:
    cache_root: Path
    params: dict[str, BiomeParams]   # slug -> BiomeParams
    seeds: dict[str, int]            # slug -> render seed

    @property
    def params_path(self) -> Path:
        return self.cache_root / "biome_params.json"

    def save(self) -> None:
        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {slug: p.to_dict() for slug, p in self.params.items()},
            "seeds": self.seeds,
        }
        self.params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state(cache_root: Path) -> StudioState:
    state_path = cache_root / "biome_params.json"
    params: dict[str, BiomeParams] = {}
    for biome in CATALOG_STANDARD:
        params[biome.slug] = DEFAULT_PARAMS[biome.slug]
    seeds = {b.slug: 20260501 for b in CATALOG_STANDARD}

    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            for slug, p_dict in data.get("params", {}).items():
                if slug in params:
                    params[slug] = BiomeParams.from_dict(p_dict)
            for slug, sd in data.get("seeds", {}).items():
                if slug in seeds:
                    seeds[slug] = int(sd)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("could not parse %s: %s", state_path, e)

    return StudioState(cache_root=cache_root, params=params, seeds=seeds)
