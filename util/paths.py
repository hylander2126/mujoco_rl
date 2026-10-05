from __future__ import annotations

from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs"
PARAMETER_ESTIMATION_OUTPUTS = OUTPUT_ROOT / "parameter_estimation"
PUSH_SELECTION_OUTPUTS = OUTPUT_ROOT / "push_selection"
CONTACT_SELECTION_OUTPUTS = OUTPUT_ROOT / "contact_selection"


def dated(name: str) -> str:
    """`YYYY-MM-DD_name`, so output folders sort chronologically."""
    return f"{date.today():%Y-%m-%d}_{name}"


def latest_suite() -> Path | None:
    """Newest full rerun under `outputs/contact_selection/suites/`, if any."""
    suites = sorted((CONTACT_SELECTION_OUTPUTS / "suites").glob("*/suite_summary.json"))
    return suites[-1].parent if suites else None


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve `path` against the repo root if it isn't already absolute."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
