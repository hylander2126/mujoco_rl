from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs"
ROBOT_LEARNING_OUTPUTS = OUTPUT_ROOT / "robot_learning"
PARAMETER_ESTIMATION_OUTPUTS = OUTPUT_ROOT / "parameter_estimation"
PUSH_SELECTION_OUTPUTS = OUTPUT_ROOT / "push_selection"


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve `path` against the repo root if it isn't already absolute."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
