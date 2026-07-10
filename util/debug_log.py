from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from util.paths import REPO_ROOT

DEBUG_ROOT = REPO_ROOT / "outputs" / "robot_learning" / "debug"


def new_debug_dir(script_name: str) -> Path:
    """Create a fresh timestamped directory for one script run's debug artifacts."""
    run_dir = DEBUG_ROOT / script_name / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_frame(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


class StepLogger:
    """Accumulates per-step scalars/arrays over an episode, then dumps them to a
    .npz trace and a quick multi-panel PNG so a run can be inspected afterward
    without needing to watch a video."""

    def __init__(self):
        self._fields: dict[str, list] = {}

    def log(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self._fields.setdefault(key, []).append(np.asarray(value, dtype=float))

    def save(self, npz_path: Path, plot_path: Path | None = None, t_key: str = "t") -> None:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.stack(v) for k, v in self._fields.items()}
        np.savez_compressed(npz_path, **arrays)
        if plot_path is not None:
            _plot_step_arrays(arrays, plot_path, t_key=t_key)


def _plot_step_arrays(arrays: dict[str, np.ndarray], path: Path, t_key: str = "t") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = arrays.get(t_key)
    plot_keys = [k for k in arrays if k != t_key]
    if t is None or not plot_keys:
        return

    fig, axes = plt.subplots(len(plot_keys), 1, figsize=(9, 2.4 * len(plot_keys)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, plot_keys):
        values = arrays[key]
        if values.ndim == 1:
            ax.plot(t, values)
        else:
            for dim in range(values.shape[1]):
                ax.plot(t, values[:, dim], label=f"{key}[{dim}]", linewidth=1)
            ax.legend(fontsize=6, ncol=min(values.shape[1], 6))
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sim time (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
