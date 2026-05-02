from __future__ import annotations

import pytest

from maptool.regions import REGIONS, BBox, output_dims, parse_bbox, subdivide_into_tiles


def test_builtin_regions_are_valid():
    for name, bb in REGIONS.items():
        assert bb.is_valid(), f"{name} bbox invalid"
        assert -90 <= bb.south < bb.north <= 90, f"{name} latitude out of range"
        assert -180 <= bb.west < bb.east <= 180, f"{name} longitude out of range"


def test_parse_bbox_round_trip():
    bb = parse_bbox("36.5,38.5,12.0,16.0")
    assert bb == BBox(36.5, 38.5, 12.0, 16.0)


def test_parse_bbox_rejects_inverted_axes():
    with pytest.raises(ValueError):
        parse_bbox("40,38,12,16")  # south > north
    with pytest.raises(ValueError):
        parse_bbox("36,38,16,12")  # west > east


def test_parse_bbox_rejects_wrong_arity():
    with pytest.raises(ValueError):
        parse_bbox("36,38,12")
    with pytest.raises(ValueError):
        parse_bbox("36,38,12,16,18")


def test_parse_bbox_rejects_non_numeric():
    with pytest.raises(ValueError):
        parse_bbox("a,b,c,d")


def test_output_dims_landscape():
    bb = BBox(36.5, 38.5, 12.0, 16.0)  # 4° wide, 2° tall, aspect=2
    w, h = output_dims(bb, 4096)
    assert w == 4096
    assert h == 2048


def test_output_dims_portrait():
    bb = BBox(36.0, 47.0, 6.0, 9.0)  # 3° wide, 11° tall
    w, h = output_dims(bb, 4096)
    assert h == 4096
    assert w < h


def test_subdivide_into_tiles_aligned():
    bb = BBox(36.0, 38.0, 12.0, 16.0)  # 4° lon × 2° lat
    tiles = subdivide_into_tiles(bb, tile_deg=1.0)
    # 4 cols x 2 rows = 8 tiles
    assert len(tiles) == 8
    coords = {(tx, ty) for tx, ty, _ in tiles}
    assert coords == {(tx, ty) for tx in range(4) for ty in range(2)}
    # Tile (0, 0) is north-west
    nw = next(tb for tx, ty, tb in tiles if tx == 0 and ty == 0)
    assert nw.north == 38.0
    assert nw.west == 12.0
    # Tile (3, 1) is south-east
    se = next(tb for tx, ty, tb in tiles if tx == 3 and ty == 1)
    assert se.south == 36.0
    assert se.east == 16.0


def test_subdivide_into_tiles_clipped_at_edges():
    # 1.5° wide bbox with 1° tiles -> last tile gets clipped to 0.5° wide
    bb = BBox(36.0, 37.0, 12.0, 13.5)
    tiles = subdivide_into_tiles(bb, tile_deg=1.0)
    assert len(tiles) == 2
    # tx=1 should be clipped at the east edge
    east_tile = next(tb for tx, ty, tb in tiles if tx == 1)
    assert east_tile.east == 13.5
    assert east_tile.west == 13.0


def test_subdivide_into_tiles_rejects_invalid_size():
    bb = BBox(36.0, 38.0, 12.0, 16.0)
    with pytest.raises(ValueError):
        subdivide_into_tiles(bb, tile_deg=0)
    with pytest.raises(ValueError):
        subdivide_into_tiles(bb, tile_deg=-1.0)
