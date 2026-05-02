"""Built-in region definitions and BBox helpers.

A BBox is a tuple ``(south, north, west, east)`` in decimal degrees, EPSG:4326.
"""

from __future__ import annotations

import math
from typing import NamedTuple


class BBox(NamedTuple):
    south: float
    north: float
    west: float
    east: float

    def is_valid(self) -> bool:
        return self.south < self.north and self.west < self.east

    def lat_span(self) -> float:
        return self.north - self.south

    def lon_span(self) -> float:
        return self.east - self.west

    def aspect(self) -> float:
        """Lon/Lat ratio. Equirectangular pixel aspect."""
        return self.lon_span() / self.lat_span()


REGIONS: dict[str, BBox] = {
    # Italy / Tyrrhenian — sanity-check region (small, fast)
    "sicily":         BBox(36.5, 38.5, 12.0, 16.0),
    "italy-central":  BBox(40.0, 44.0, 10.0, 16.0),
    "italy-full":     BBox(36.0, 47.0,  6.0, 19.0),
    # Western Mediterranean — Punic War 218 BCE theatre
    "iberia":         BBox(36.0, 44.0, -10.0,  4.0),
    "iberia-south":   BBox(36.0, 40.0,  -6.0,  0.0),
    "africa-procons": BBox(31.0, 38.0,   7.0, 14.0),
    "sardinia":       BBox(38.8, 41.4,   8.0, 10.0),
    "corsica":        BBox(41.3, 43.1,   8.5,  9.6),
    "westmed":        BBox(31.0, 47.0, -10.0, 19.0),  # everything
}


def parse_bbox(s: str) -> BBox:
    """Parse a CLI 'south,north,west,east' string."""
    parts = s.split(",")
    if len(parts) != 4:
        raise ValueError("--bbox needs 4 values: south,north,west,east")
    try:
        south, north, west, east = (float(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"--bbox values must be numeric: {e}") from None
    bb = BBox(south, north, west, east)
    if not bb.is_valid():
        raise ValueError(
            f"Invalid bbox: south<north ({south}<{north})? west<east ({west}<{east})?"
        )
    return bb


def output_dims(bbox: BBox, max_dim: int) -> tuple[int, int]:
    """Pixel dimensions for an equirectangular projection of bbox, capped at max_dim."""
    aspect = bbox.aspect()
    if aspect >= 1.0:
        return max_dim, int(round(max_dim / aspect))
    return int(round(max_dim * aspect)), max_dim


def subdivide_into_tiles(bbox: BBox, tile_deg: float) -> list[tuple[int, int, BBox]]:
    """Split *bbox* into a grid of (tx, ty, sub_bbox) tiles.

    *tx* increases eastward from the bbox.west edge, *ty* increases southward
    from bbox.north so tile (0, 0) is the north-west corner.

    Each tile's sub-bbox is *tile_deg* x *tile_deg* in WGS84, clipped to the
    parent bbox at the south/east edges if the bbox isn't an integer multiple.
    """
    if tile_deg <= 0:
        raise ValueError(f"tile_deg must be positive, got {tile_deg}")

    n_x = max(1, math.ceil(bbox.lon_span() / tile_deg))
    n_y = max(1, math.ceil(bbox.lat_span() / tile_deg))

    tiles = []
    for ty in range(n_y):
        north = bbox.north - ty * tile_deg
        south = max(bbox.south, north - tile_deg)
        for tx in range(n_x):
            west = bbox.west + tx * tile_deg
            east = min(bbox.east, west + tile_deg)
            tile_bb = BBox(south, north, west, east)
            tiles.append((tx, ty, tile_bb))
    return tiles
