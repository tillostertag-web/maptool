"""Multi-octave paint-texture noise for the climate-painted preview.

Two-tier strategy keeps memory bounded at ultra resolution:

* **Low-frequency tier** is a small (2048 x 2048) FBM-style buffer pre-baked
  once at module init. Sample bicubic-upscaled chunks during the stripe loop
  to give large painterly patches.
* **High-frequency tier** is freshly generated per stripe at canvas resolution
  with a small gaussian (sigma ~ 1.5-2 px) — fine canvas grain that survives
  bilinear viewer zoom as visible granularity.

Combined the layer reads as "hand-painted parchment texture" instead of
"smooth computer-rendered hillshade".
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

log = logging.getLogger(__name__)

LOW_TIER_SIZE = 2048
LOW_OCTAVES: tuple[tuple[float, float], ...] = (
    (28.0, 0.65),  # broad patches  (sigma_px, weight)
    (10.0, 0.30),  # mid brushwork
    (4.0,  0.05),  # fine canvas grain at low-tier scale
)
HIGH_FREQ_SIGMA = 1.6
HIGH_FREQ_WEIGHT = 0.30   # blend with low tier (low gets 1 - HIGH_FREQ_WEIGHT)
DEFAULT_AMPLITUDE = 0.10  # ±10 % multiplicative on RGB


def _normalize(arr: np.ndarray) -> np.ndarray:
    std = float(arr.std())
    if std < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = arr / std
    return out.astype(np.float32, copy=False)


def build_low_tier(seed: int = 0xC0FF) -> np.ndarray:
    """Pre-baked low-frequency FBM pattern, zero-mean, std≈1, in [-3, 3] roughly."""
    rng = np.random.default_rng(seed)
    accum = np.zeros((LOW_TIER_SIZE, LOW_TIER_SIZE), dtype=np.float32)
    for sigma, weight in LOW_OCTAVES:
        n = rng.normal(0.0, 1.0, accum.shape).astype(np.float32)
        n = gaussian_filter(n, sigma=sigma).astype(np.float32, copy=False)
        accum += _normalize(n) * np.float32(weight)
    return _normalize(accum)


def sample_low_tier(
    low_tier: np.ndarray,
    canvas_h: int,
    canvas_w: int,
    y0: int,
    y1: int,
) -> np.ndarray:
    """Bicubic sample a (y1-y0, canvas_w) chunk from the low-tier pattern."""
    lh, lw = low_tier.shape
    # Crop the matching latitude slice from the low-tier so subsequent stripes
    # are continuous. Map full canvas (canvas_h tall) -> (lh tall) linearly.
    src_y0 = int(round(y0 * lh / canvas_h))
    src_y1 = int(round(y1 * lh / canvas_h))
    src_y0 = max(0, min(lh - 1, src_y0))
    src_y1 = max(src_y0 + 1, min(lh, src_y1))

    src_chunk = low_tier[src_y0:src_y1]
    src_min = float(src_chunk.min())
    src_max = float(src_chunk.max())
    span = src_max - src_min if src_max > src_min else 1.0
    src_u8 = ((src_chunk - src_min) * (255.0 / span)).astype(np.uint8)

    target_h = y1 - y0
    img = Image.fromarray(src_u8, mode="L").resize(
        (canvas_w, target_h), Image.Resampling.BICUBIC
    )
    upsampled = np.asarray(img, dtype=np.float32) * np.float32(span / 255.0) + np.float32(src_min)
    return upsampled


def gen_high_freq(stripe_h: int, canvas_w: int, rng: np.random.Generator) -> np.ndarray:
    """Per-stripe high-freq grain, zero-mean, std≈1."""
    n = rng.normal(0.0, 1.0, (stripe_h, canvas_w)).astype(np.float32)
    n = gaussian_filter(n, sigma=HIGH_FREQ_SIGMA).astype(np.float32, copy=False)
    return _normalize(n)


def sample_chunk(
    low_tier: np.ndarray,
    canvas_h: int,
    canvas_w: int,
    y0: int,
    y1: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Combined paint-noise chunk in roughly [-1, 1] (zero-mean, std≈1)."""
    low = sample_low_tier(low_tier, canvas_h, canvas_w, y0, y1)
    high = gen_high_freq(y1 - y0, canvas_w, rng)
    return low * np.float32(1.0 - HIGH_FREQ_WEIGHT) + high * np.float32(HIGH_FREQ_WEIGHT)
