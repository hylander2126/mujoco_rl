from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Single-file rollout datasets require h5py. Install it in the "
            "training environment with: pip install h5py"
        ) from exc
    return h5py


class HDF5Writer:
    """Append episodes to one HDF5 file without retaining the run in RAM."""

    def __init__(self, output_path: Path):
        h5py = _require_h5py()
        self.path = Path(output_path)
        if self.path.suffix != ".h5":
            raise ValueError(f"Rollout output must use the .h5 extension, got {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self.num_episodes = 0
        self.total_samples = 0

    def write_episode(self, ep_num: int, images: np.ndarray, **meta: np.ndarray) -> None:
        arrays = {"images": images, **meta}
        n = len(images)
        if n == 0:
            return
        for key, values in arrays.items():
            values = np.asarray(values)
            string_dtype = None
            if values.dtype.kind == "U":
                # Widths can vary between episodes ("red" then "blue"), so a
                # width inferred from the first episode would truncate later
                # labels. HDF5 variable-length UTF-8 remains compact and safe.
                string_dtype = _require_h5py().string_dtype(encoding="utf-8")
                values = values.astype(object)
            if key not in self._file:
                # Training indexes shuffled frames individually. Keep images
                # one frame per chunk so a 49 KB sample read cannot pull a
                # multi-megabyte block of neighboring images into memory.
                chunk_rows = 1 if key == "images" else min(n, 256)
                self._file.create_dataset(
                    key,
                    data=values,
                    dtype=string_dtype,
                    maxshape=(None, *values.shape[1:]),
                    chunks=(chunk_rows, *values.shape[1:]),
                )
            else:
                dataset = self._file[key]
                start = len(dataset)
                dataset.resize(start + n, axis=0)
                dataset[start:] = values
        self.num_episodes += 1
        self.total_samples += n
        self._file.flush()

    def finalize(self, **dataset_meta) -> None:
        self._file.attrs["num_episodes"] = self.num_episodes
        self._file.attrs["total_samples"] = self.total_samples
        for key, value in dataset_meta.items():
            self._file.attrs[key] = np.asarray(value).item()
        self._file.close()


class _HDF5Images:
    """Process-local lazy image view, safe with DataLoader worker processes."""

    def __init__(self, path: Path, shape: tuple, dtype):
        self.path = Path(path)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._pid = None
        self._file = None

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i: int) -> np.ndarray:
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = _require_h5py().File(self.path, "r")
            self._pid = pid
        return np.asarray(self._file["images"][int(i)])


class RolloutDataset:
    """Read-only view over one HDF5 rollout; images remain lazy."""

    def __init__(self, arrays: dict, files: list[str]):
        self._arrays = arrays
        self.files = files

    def __getitem__(self, key: str):
        return self._arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self._arrays


def _load_hdf5(path: Path) -> RolloutDataset:
    h5py = _require_h5py()
    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        image_dataset = h5["images"]
        arrays = {
            key: h5[key].asstr()[:] if h5[key].dtype.kind in {"O", "S"} else np.asarray(h5[key])
            for key in keys
            if key != "images"
        }
        arrays["images"] = _HDF5Images(path, tuple(image_dataset.shape), image_dataset.dtype)
        arrays.update(
            {
                key: value
                for key, value in h5.attrs.items()
                if key not in {"num_episodes", "total_samples"}
            }
        )
    return RolloutDataset(arrays, sorted(set(keys) | set(arrays)))


def load_rollout_dataset(path: Path) -> RolloutDataset:
    """Load one HDF5 rollout dataset with lazy image access."""
    path = Path(path)
    if path.suffix != ".h5":
        raise ValueError(f"Rollout dataset must be an .h5 file, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"No rollout dataset found at {path}")
    return _load_hdf5(path)
