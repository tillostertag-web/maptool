from __future__ import annotations

import numpy as np

from maptool.regions import BBox
from maptool.render.climate_paint import (
    SEA_LEVEL_NORM,
    climate_color,
    compute_beach_blend,
    compute_hillshade,
    lat_grid_for,
)
from maptool.render.palette import DRY_MED, MOIST_MED, TEMPERATE


def test_palette_stops_are_monotonic_and_in_range():
    for name, p in [("DRY_MED", DRY_MED), ("MOIST_MED", MOIST_MED), ("TEMPERATE", TEMPERATE)]:
        stops = p[:, 0]
        assert np.all(np.diff(stops) > 0), f"{name} stops not strictly increasing"
        assert stops[0] == 0.0 and stops[-1] == 1.0, f"{name} must span [0,1]"
        assert p[:, 1:].min() >= 0 and p[:, 1:].max() <= 255


def test_lat_grid_north_to_south():
    bb = BBox(36.0, 44.0, 10.0, 14.0)
    grid = lat_grid_for(8, 4, bb)
    assert grid.shape == (8, 4)
    assert grid[0, 0] == 44.0
    assert grid[-1, 0] == 36.0


def test_climate_color_blends_by_latitude():
    norm = np.full((3, 1), 0.1, dtype=np.float32)
    south_lat = np.full((3, 1), 36.0, dtype=np.float32)
    north_lat = np.full((3, 1), 47.0, dtype=np.float32)
    south = climate_color(norm, south_lat)
    north = climate_color(norm, north_lat)
    # At lowland norm 0.1, dry-med is warm sand (R > G); temperate is green (G > R).
    assert south[0, 0, 0] > south[0, 0, 1]               # dry: red > green
    assert north[0, 0, 1] > north[0, 0, 0]               # temperate: green > red
    # And the south is unambiguously redder than the north.
    assert south[0, 0, 0] > north[0, 0, 0]


def test_beach_blend_is_zero_when_no_water():
    norm = np.full((10, 10), 0.5, dtype=np.float32)
    blend = compute_beach_blend(norm)
    assert blend.max() == 0.0


def test_beach_blend_zero_when_no_land():
    norm = np.zeros((10, 10), dtype=np.float32)  # all sea
    blend = compute_beach_blend(norm)
    assert blend.max() == 0.0


def test_beach_blend_peaks_at_coast():
    # Big enough that BEACH_WIDTH_FRAC kicks in above the floor.
    norm = np.zeros((400, 400), dtype=np.float32)
    norm[:, 200:] = 0.3  # right half is land
    blend = compute_beach_blend(norm)
    # The land cell directly at the coast is the brightest beach.
    coast_cell = blend[200, 200]
    far_inland = blend[200, 399]
    assert coast_cell > far_inland
    assert coast_cell > 0.5
    assert far_inland == 0.0
    # The blend monotonically decreases inland.
    inland_strip = blend[200, 200:220]
    assert np.all(np.diff(inland_strip) <= 1e-6)


def test_hillshade_in_expected_range():
    rng = np.random.default_rng(seed=0)
    norm = rng.random((32, 32), dtype=np.float32)
    hs = compute_hillshade(norm)
    assert hs.shape == (32, 32)
    assert hs.min() >= 0.25 and hs.max() <= 1.05


def test_sea_level_constant_sane():
    assert 0 < SEA_LEVEL_NORM < 0.05
