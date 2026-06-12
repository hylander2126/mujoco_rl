from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VLADataset(Dataset):
    """Loads simulated VLA rollouts saved by `collect_sim_data.py`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        data = np.load(self.path, allow_pickle=False)
        self.images = data["images"]
        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.instructions = data["instructions"].astype(str)
        self.record_stride = int(data["record_stride"]) if "record_stride" in data.files else None
        self.sim_timestep = float(data["sim_timestep"]) if "sim_timestep" in data.files else None
        self.max_sim_time = float(data["max_sim_time"]) if "max_sim_time" in data.files else None

        if not (len(self.images) == len(self.states) == len(self.actions) == len(self.instructions)):
            raise ValueError("Dataset arrays have inconsistent lengths.")

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image = torch.from_numpy(self.images[idx]).float() / 255.0
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape {tuple(image.shape)}")
        image = image.permute(2, 0, 1)

        return {
            "image": image,
            "state": torch.from_numpy(self.states[idx]),
            "instruction": self.instructions[idx],
            "action": torch.from_numpy(self.actions[idx]),
        }
