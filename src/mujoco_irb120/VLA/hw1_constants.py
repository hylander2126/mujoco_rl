from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# HW1 shared constants
# ---------------------------------------------------------------------------
#
# Keep HW1 task constants here so the environment, scripted expert, and later
# training/eval tools agree about home pose, colors, and waypoint defaults.

HW1_HOME_Q = np.zeros(6, dtype=np.float32)
HW1_HOME_Q[4] = -1.5708  # Point the tray upward.

HW1_CUBE_COLORS = ("red", "blue")

HW1_CUBE_RGBA = {
    "red": np.array([1.0, 0.05, 0.05, 1.0]),
    "blue": np.array([0.05, 0.2, 1.0, 1.0]),
}

HW1_BIN_SITE_BY_COLOR = {
    "red": "site:red_bin",
    "blue": "site:blue_bin",
}

HW1_PRE_DROP_SECONDS = 2.0
HW1_TIP_SECONDS = 1.0
HW1_DROP_HOLD_SECONDS = 1.0
HW1_RETURN_HOME_SECONDS = 1.0

HW1_PRE_DROP_XYZ_BY_COLOR = {
    "red": np.array([0.78, -0.28, 0.38], dtype=float),
    "blue": np.array([0.78, 0.28, 0.38], dtype=float),
}

# Positive/negative signs assume the bins are split along world y.
HW1_TRAY_TIP_RAD_BY_COLOR = {
    "red": -0.75,
    "blue": 0.75,
}

