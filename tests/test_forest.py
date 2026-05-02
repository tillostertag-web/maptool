from __future__ import annotations

import numpy as np

from maptool.regions import BBox
from maptool.render.forest import (
    DENSITY_FLOOR,
    SPRITE_POOL_SIZE,
    SPRITE_SOURCE_SIZE,
    TINT_DRY,
    TINT_TEMPERATE,
    _build_density,
    _build_sprite_pool,
    _make_canopy,
    climate_tint_per_row,
)


def test_climate_tint_per_row_blends_south_to_north():
    bbox = BBox(36.0, 47.0, 0.0, 5.0)
    tint = climate_tint_per_row(11, bbox)
    assert tint.shape == (11, 3)
    # Row 0 = north pole of bbox = colder/temperate
    # Row -1 = south = dry/warmer
    assert np.allclose(tint[0], TINT_TEMPERATE, atol=1.0)
    assert np.allclose(tint[-1], TINT_DRY, atol=1.0)


def test_make_canopy_returns_alpha_in_range():
    rng = np.random.default_rng(0)
    for kind in ("round", "lobed", "small"):
        a = _make_canopy(rng, kind)
        assert a.shape == (SPRITE_SOURCE_SIZE, SPRITE_SOURCE_SIZE)
        assert a.min() >= 0.0
        assert a.max() <= 1.0
        # Sprite must actually contain something visible.
        assert a.max() > 0.5


def test_sprite_pool_has_expected_size():
    pool = _build_sprite_pool(seed=42)
    assert len(pool) == SPRITE_POOL_SIZE
    # All variants should differ pixel-for-pixel.
    flat = np.stack([p.ravel() for p in pool])
    assert len(np.unique(flat, axis=0)) >= SPRITE_POOL_SIZE - 2  # tolerate rare collisions


def test_density_mask_zero_when_no_forest():
    codes = np.zeros((64, 64), dtype=np.uint8)  # no class-10 cells
    d = _build_density(codes, seed=0)
    assert d.max() < DENSITY_FLOOR


def test_density_mask_nonzero_for_forest_blob():
    codes = np.zeros((64, 64), dtype=np.uint8)
    codes[20:40, 20:40] = 10  # WorldCover tree-cover class
    d = _build_density(codes, seed=0)
    assert d.shape == (64, 64)
    assert d.min() >= 0.0
    assert d.max() <= 1.0
    assert d[30, 30] > DENSITY_FLOOR
