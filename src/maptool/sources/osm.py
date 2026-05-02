"""OpenStreetMap data fetcher via the Overpass API.

Fetches rivers, roads and settlement nodes for an arbitrary bbox and caches
the raw GeoJSON-style response on disk. Subsequent calls hit the cache; only
new bbox + query combinations talk to the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from maptool.regions import BBox

log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_TIMEOUT = 60   # seconds for the Overpass query


@dataclass(frozen=True)
class Linestring:
    """A polyline with WGS84 coordinates ((lon, lat) pairs)."""
    coords: tuple[tuple[float, float], ...]
    tags: dict[str, str]


@dataclass(frozen=True)
class PointFeature:
    """A point in WGS84 plus its OSM tags."""
    lon: float
    lat: float
    tags: dict[str, str]


def _cache_key(bbox: BBox, query: str) -> str:
    raw = f"{bbox.south:.4f},{bbox.north:.4f},{bbox.west:.4f},{bbox.east:.4f}|{query}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, bbox: BBox, query: str) -> Path:
    return cache_dir / "osm" / f"{_cache_key(bbox, query)}.json"


def _fetch_overpass(query: str, *, timeout: int) -> dict:
    """Send a raw Overpass QL query, return the parsed JSON response."""
    log.info("Overpass query (%d chars)...", len(query))
    t0 = time.time()
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Overpass {r.status_code}: {r.text[:300]}")
    payload = r.json()
    log.info("  Overpass done in %.1fs (%d elements)",
             time.time() - t0, len(payload.get("elements", [])))
    return payload


def fetch(bbox: BBox, query_body: str, cache_dir: Path, *, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Cached Overpass fetch. *query_body* is the part inside the [bbox]; clause."""
    cp = _cache_path(cache_dir, bbox, query_body)
    if cp.exists():
        log.info("OSM cached: %s", cp.name)
        with open(cp, "r", encoding="utf-8") as f:
            return json.load(f)

    bbox_str = f"{bbox.south},{bbox.west},{bbox.north},{bbox.east}"
    full = f"[out:json][timeout:{timeout}][bbox:{bbox_str}];\n{query_body}\nout geom;"
    payload = _fetch_overpass(full, timeout=timeout)

    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def _ways_to_linestrings(payload: dict) -> list[Linestring]:
    out: list[Linestring] = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        coords = tuple((float(p["lon"]), float(p["lat"])) for p in geom)
        out.append(Linestring(coords=coords, tags=el.get("tags") or {}))
    return out


def _nodes_to_points(payload: dict) -> list[PointFeature]:
    out: list[PointFeature] = []
    for el in payload.get("elements", []):
        if el.get("type") != "node":
            continue
        out.append(PointFeature(
            lon=float(el["lon"]),
            lat=float(el["lat"]),
            tags=el.get("tags") or {},
        ))
    return out


def fetch_rivers(bbox: BBox, cache_dir: Path) -> list[Linestring]:
    """Fetch waterways (rivers + larger streams) within *bbox*."""
    query = (
        '(way["waterway"="river"];'
        ' way["waterway"="canal"];'
        ' way["waterway"="stream"];);'
    )
    return _ways_to_linestrings(fetch(bbox, query, cache_dir))


def fetch_roads(bbox: BBox, cache_dir: Path) -> list[Linestring]:
    """Fetch major roads (motorway/trunk/primary/secondary)."""
    query = 'way["highway"~"^(motorway|trunk|primary|secondary)$"];'
    return _ways_to_linestrings(fetch(bbox, query, cache_dir))


def fetch_settlements(bbox: BBox, cache_dir: Path) -> list[PointFeature]:
    """Fetch named settlements (city / town / village)."""
    query = 'node["place"~"^(city|town|village)$"]["name"];'
    return _nodes_to_points(fetch(bbox, query, cache_dir))
