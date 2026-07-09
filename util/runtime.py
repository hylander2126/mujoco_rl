from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch


def select_torch_device() -> torch.device:
    """Prefer CUDA, while safely falling back when its runtime cannot initialize."""
    try:
        with warnings.catch_warnings(record=True) as cuda_warnings:
            warnings.simplefilter("always")
            cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.device("cuda")
            print(f"[runtime] Using CUDA: {torch.cuda.get_device_name(device)}")
            return device
        if cuda_warnings:
            print(f"[runtime] CUDA unavailable ({cuda_warnings[-1].message}); using CPU.")
        else:
            print("[runtime] CUDA unavailable; using CPU.")
    except (RuntimeError, OSError) as exc:
        print(f"[runtime] CUDA initialization failed ({exc}); using CPU.")
    return torch.device("cpu")


class EpisodeVideoRecorder:
    """Write simulation frames to an MP4 without opening an on-screen viewer."""

    def __init__(self, path: Path, fps: int = 30, output_size: tuple[int, int] | None = None):
        try:
            import imageio.v2 as imageio
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Video rendering requires imageio, imageio-ffmpeg, and Pillow. "
                "Install the project requirements and try again."
            ) from exc

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.output_size = output_size
        self._image_type = Image
        self._frame_period = 1.0 / fps
        self._next_frame_time = 0.0
        self._writer = imageio.get_writer(
            self.path,
            fps=fps,
            codec="libx264",
            quality=10,
            macro_block_size=None,
        )

    def is_frame_due(self, sim_time: float) -> bool:
        return sim_time + 1e-9 >= self._next_frame_time

    def capture(self, frame: np.ndarray, sim_time: float, *, force: bool = False) -> None:
        if not force and sim_time + 1e-9 < self._next_frame_time:
            return
        frame = np.asarray(frame, dtype=np.uint8)
        if self.output_size is not None and (frame.shape[1], frame.shape[0]) != self.output_size:
            frame = np.asarray(
                self._image_type.fromarray(frame).resize(
                    self.output_size,
                    resample=self._image_type.Resampling.LANCZOS,
                )
            )
        self._writer.append_data(frame)
        while self._next_frame_time <= sim_time + 1e-9:
            self._next_frame_time += self._frame_period

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            print(f"Saved video to {self.path}")

    def __enter__(self) -> "EpisodeVideoRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
