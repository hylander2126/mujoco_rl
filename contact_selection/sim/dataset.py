"""Strict JSONL outcomes with scene manifests and complete MuJoCo snapshots."""
import hashlib
import json
from pathlib import Path

import numpy as np


def json_value(value):
    """Convert NumPy values and represent unavailable/nonfinite diagnostics as null."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')


def append_record(path: Path, record: dict) -> None:
    with path.open('a') as stream:
        stream.write(json.dumps(json_value(record), allow_nan=False) + '\n')


def read_records(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def content_id(value) -> str:
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True).encode()).hexdigest()[:16]
