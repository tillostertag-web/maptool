"""Climate-zone elevation palettes.

Each palette is a list of stops ``(norm_elev, R, G, B)``, interpolated channel-wise.

v0.1 fix vs the original aquila palettes (see Memory: terrain_progress —
"mountains need more contrast"): an extra dark-rock stop at norm 0.55 in every
palette so high terrain doesn't bleach into a grey mush at norm 0.7+.
"""

from __future__ import annotations

import numpy as np

# Stops: (norm_elev, R, G, B)
# - 0.000  deep water
# - 0.005  shallow water
# - 0.008  water-sand transition (pre-beach)
# - 0.020  coast
# - 0.100  flat / lowland
# - 0.250  hills
# - 0.450  rocky
# - 0.550  high rocky shadow  <-- NEW: mountain contrast fix
# - 0.700  exposed rock face
# - 0.880  near snow
# - 1.000  snow

DRY_MED = np.array([
    (0.000,  40,  60,  95),
    (0.005,  70, 105, 140),
    (0.008, 165, 185, 190),
    (0.020, 215, 195, 150),
    (0.100, 195, 170, 115),
    (0.250, 165, 135,  95),
    (0.450, 130, 105,  85),
    (0.550,  90,  72,  60),   # darker, more saturated rock
    (0.700, 150, 140, 125),
    (0.880, 220, 215, 210),
    (1.000, 250, 250, 252),
], dtype=np.float32)

MOIST_MED = np.array([
    (0.000,  40,  60,  95),
    (0.005,  70, 105, 140),
    (0.008, 175, 190, 175),
    (0.020, 175, 180, 125),
    (0.100, 135, 155, 100),
    (0.250, 105, 120,  85),
    (0.450, 115,  95,  75),
    (0.550,  72,  58,  48),   # darker rock
    (0.700, 150, 140, 125),
    (0.880, 220, 215, 210),
    (1.000, 250, 250, 252),
], dtype=np.float32)

TEMPERATE = np.array([
    (0.000,  40,  60,  95),
    (0.005,  70, 105, 140),
    (0.008, 185, 190, 160),
    (0.020, 130, 160, 105),
    (0.100,  90, 130,  80),
    (0.250,  85, 105,  70),
    (0.450, 110,  95,  75),
    (0.550,  68,  56,  46),   # darker rock
    (0.700, 150, 140, 125),
    (0.880, 220, 215, 210),
    (1.000, 250, 250, 252),
], dtype=np.float32)


def interp_palette(norm: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """norm: (H,W) float32 in [0,1] -> (H,W,3) float32 RGB in [0,255].

    np.interp returns float64; we cast each channel down before stacking so
    a 200 MP frame (ultra mode) doesn't allocate a 4.8 GB float64 buffer.
    """
    stops = palette[:, 0]
    out = np.empty((norm.shape[0], norm.shape[1], 3), dtype=np.float32)
    out[..., 0] = np.interp(norm, stops, palette[:, 1]).astype(np.float32, copy=False)
    out[..., 1] = np.interp(norm, stops, palette[:, 2]).astype(np.float32, copy=False)
    out[..., 2] = np.interp(norm, stops, palette[:, 3]).astype(np.float32, copy=False)
    return out
