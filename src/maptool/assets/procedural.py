"""Procedural biome texture generator.

Takes a small bundle of parameters (two RGB colors, two noise scales,
brightness, contrast) and renders a tileable square texture in pure
numpy. No API calls, deterministic given a seed.

Each pixel is a smoothstep blend between *base_color* and *accent_color*
driven by FBM-ish noise (one low-frequency octave for big patches plus
one high-frequency octave for canvas grain).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


@dataclass
class BiomeParams:
    slug: str
    base_color: tuple[int, int, int]
    accent_color: tuple[int, int, int]
    noise_scale: float = 22.0      # sigma for the low-frequency octave
    grain_scale: float = 1.8       # sigma for the high-frequency octave
    grain_weight: float = 0.30     # mix weight for the grain octave (0..1)
    brightness: float = 1.0        # multiplier on the final RGB
    contrast: float = 1.0          # contrast around mid-grey

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "base_color": list(self.base_color),
            "accent_color": list(self.accent_color),
            "noise_scale": float(self.noise_scale),
            "grain_scale": float(self.grain_scale),
            "grain_weight": float(self.grain_weight),
            "brightness": float(self.brightness),
            "contrast": float(self.contrast),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BiomeParams":
        return cls(
            slug=str(d["slug"]),
            base_color=tuple(int(c) for c in d["base_color"]),  # type: ignore[arg-type]
            accent_color=tuple(int(c) for c in d["accent_color"]),  # type: ignore[arg-type]
            noise_scale=float(d.get("noise_scale", 22.0)),
            grain_scale=float(d.get("grain_scale", 1.8)),
            grain_weight=float(d.get("grain_weight", 0.30)),
            brightness=float(d.get("brightness", 1.0)),
            contrast=float(d.get("contrast", 1.0)),
        )


DEFAULT_PARAMS: dict[str, BiomeParams] = {
    "sea":              BiomeParams("sea",              (38, 70, 105),    (60, 92, 130),    noise_scale=24.0, grain_scale=2.0, grain_weight=0.20),
    "coast":            BiomeParams("coast",            (220, 200, 160),  (245, 230, 195),  noise_scale=14.0, grain_scale=1.5, grain_weight=0.35),
    "arid":             BiomeParams("arid",             (180, 158, 110),  (148, 122, 88),   noise_scale=18.0, grain_scale=1.5, grain_weight=0.42),
    "temperate_grass":  BiomeParams("temperate_grass",  (110, 128, 78),   (148, 158, 92),   noise_scale=22.0, grain_scale=1.8, grain_weight=0.36),
    "temperate_forest": BiomeParams("temperate_forest", (40, 55, 26),     (72, 88, 42),     noise_scale=12.0, grain_scale=1.4, grain_weight=0.42),
    "alpine_rock":      BiomeParams("alpine_rock",      (110, 96, 84),    (78, 66, 58),     noise_scale=14.0, grain_scale=1.3, grain_weight=0.55),
    "cliffs":           BiomeParams("cliffs",           (105, 90, 76),    (138, 122, 102),  noise_scale=16.0, grain_scale=1.4, grain_weight=0.48),
    "snow":             BiomeParams("snow",             (245, 248, 252),  (220, 230, 240),  noise_scale=22.0, grain_scale=2.0, grain_weight=0.18),
}


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Stretch *arr* to [0, 1] (zero-safe)."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32, copy=False)


def render_biome(params: BiomeParams, *, size: int = 1024, seed: int = 20260501) -> Image.Image:
    """Render a procedural square texture for *params*. Returns a PIL.Image."""
    rng = np.random.default_rng(seed)

    low = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    low = gaussian_filter(low, sigma=max(1.0, params.noise_scale))
    low = _normalize(low)

    high = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    high = gaussian_filter(high, sigma=max(0.5, params.grain_scale))
    high = _normalize(high)

    weight = float(np.clip(params.grain_weight, 0.0, 1.0))
    mix = low * np.float32(1.0 - weight) + high * np.float32(weight)
    mix = mix[..., None]

    base = np.array(params.base_color, dtype=np.float32)
    accent = np.array(params.accent_color, dtype=np.float32)
    rgb = base[None, None, :] * (np.float32(1.0) - mix) + accent[None, None, :] * mix

    if params.brightness != 1.0:
        rgb *= np.float32(params.brightness)
    if params.contrast != 1.0:
        rgb = (rgb - np.float32(128.0)) * np.float32(params.contrast) + np.float32(128.0)

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
