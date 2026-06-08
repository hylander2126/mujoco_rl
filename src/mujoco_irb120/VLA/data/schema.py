from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VLAStep:
    """One behavior-cloning sample for a VLA-style robot policy."""

    image: np.ndarray
    state: np.ndarray
    instruction: str
    action: np.ndarray
    episode_idx: int
    step_idx: int


NPZ_KEYS = (
    "images",
    "states",
    "actions",
    "instructions",
    "episode_idx",
    "step_idx",
)

