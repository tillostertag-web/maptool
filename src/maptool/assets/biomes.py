"""Biome catalog: prompts that drive fal.ai-generated tileable textures.

Concrete drone-photography framing works far better than abstract
"tileable texture" framing — Flux's training has many "drone shot of
material from 30m height" examples but few "seamless texture" ones.

Lessons baked in:
  - never negate (no "no clearing"); describe what fills the frame
  - state coverage explicitly ("filling every pixel edge to edge")
  - specify drone altitude in meters; low altitude frames as material
  - specify density ("rocks every meter", "shrubs every 2 meters")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Biome:
    slug: str
    description: str
    prompt: str

    def full_prompt(self) -> str:
        return self.prompt


CATALOG_STANDARD: tuple[Biome, ...] = (
    Biome(
        slug="sea",
        description="open Mediterranean water surface",
        prompt=(
            "Aerial drone photograph from 30 meters directly above the open ocean, "
            "deep slate-blue sea water surface filling every pixel edge to edge, "
            "gentle ripple pattern and faint wave reflections evenly across the whole frame, "
            "calm Mediterranean water, top-down orthographic view, water-only frame"
        ),
    ),
    Biome(
        slug="coast",
        description="sandy beach material",
        prompt=(
            "Aerial drone photograph from 5 meters directly above a Mediterranean beach, "
            "fine ocher sand grain texture filling every pixel edge to edge, "
            "scattered tiny pebbles and shells visible across the whole frame, "
            "wet drag marks left by waves, dry coastal sand, top-down view"
        ),
    ),
    Biome(
        slug="arid",
        description="dry mediterranean lowland scrub",
        prompt=(
            "Aerial drone photograph from 30 meters directly above semi-arid scrubland, "
            "dense low grey-green dry shrubs every 2 meters covering the whole frame "
            "edge to edge, ocher sandy soil between shrubs, North African desert vegetation, "
            "top-down orthographic view"
        ),
    ),
    Biome(
        slug="temperate_grass",
        description="fertile mid-latitude grassland mosaic",
        prompt=(
            "Aerial drone photograph from 30 meters directly above Mediterranean farmland, "
            "patchwork of small green and ocher field strips covering every pixel edge to edge, "
            "low dry-stone walls between fields, summer wheat and grass, top-down view"
        ),
    ),
    Biome(
        slug="temperate_forest",
        description="dense mediterranean broadleaf and pine forest canopy",
        prompt=(
            "Aerial drone photograph from 30 meters directly above a dense Mediterranean "
            "oak and pine forest, rounded tree crowns packed tightly across every pixel "
            "of the frame edge to edge, dark olive green canopy, top-down orthographic view, "
            "repeating canopy pattern, painted satellite map quality"
        ),
    ),
    Biome(
        slug="alpine_rock",
        description="bare exposed mountain rock and scree",
        prompt=(
            "Aerial drone photograph from 30 meters directly above a mountain scree slope, "
            "broken angular grey-brown stones of varied sizes packed tightly every meter "
            "covering the whole frame edge to edge, exposed alpine rock, top-down view"
        ),
    ),
    Biome(
        slug="cliffs",
        description="high weathered rocky band below the snowline",
        prompt=(
            "Aerial drone photograph from 50 meters directly above a high-altitude "
            "weathered rock and dirt slope, warm grey-brown stone with patches of "
            "dry brown earth and tiny scrub, no vegetation cover, top-down view"
        ),
    ),
    Biome(
        slug="snow",
        description="pristine alpine snow surface",
        prompt=(
            "Aerial drone photograph from 10 meters directly above fresh fallen snow, "
            "pristine white snow surface filling every pixel edge to edge, "
            "subtle wind drift patterns and small ripple texture across the whole frame, "
            "faint shadow modulation in folds, top-down view, snow-only frame"
        ),
    ),
)


def by_slug(catalog: tuple[Biome, ...] = CATALOG_STANDARD) -> dict[str, Biome]:
    return {b.slug: b for b in catalog}
