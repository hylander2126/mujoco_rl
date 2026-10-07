"""The hardware press heuristic, scored on the same sim candidates as the learned selectors.

Hardware (`hardware_selector.select_contact_points`, press mode) keeps cloud points
whose combined moment-arm score

    dz - press_pivot_weight * dx        (dz: height above pivot, dx: inboard distance)

is within `score_tolerance` of the best, then centres along the pivot edge and
takes the most edgeward point. Here the same rule ranks the simulator's candidate
set, using the exact sim pivot (the near-X support edge, which is what the press
mode's `estimate_pivot(preferred=-X)` targets). That makes it directly comparable
to the logistic selectors: same candidates, same robust labels, different ranking.
It isolates the scoring rule from perception error; `sim_cloud.py` adds that back.

Two deliberate differences from hardware, both because sim candidates are a
sparse grid rather than a dense cloud:
- Edge centring uses |press_offset_xy[1]| (offset from the top centre along
  world Y, the pivot-edge axis) instead of the observed edge midpoint.
- The cloud-only fallbacks (relaxed pull epsilon, topmost band for normal-less
  caps) never trigger: candidates already pass `min_upward_normal` (0.95), which
  is stricter than hardware's normal_z >= 0.7.
The heuristic never abstains while any candidate exists.
"""
from __future__ import annotations

import numpy as np

# Defaults of hardware_selector.select_contact_points().
HARDWARE_WEIGHT = 1.0
HARDWARE_TOLERANCE = 0.003


def heuristic_index(rows: list[dict], weight: float = HARDWARE_WEIGHT,
                    tolerance: float = HARDWARE_TOLERANCE) -> int | None:
    """Position in `rows` (one object's candidates) the hardware rule would press."""
    if not rows:
        return None
    dz = np.array([row['features']['pivot_dz_m'] for row in rows])
    dx = np.array([row['features']['pivot_dx_m'] for row in rows])
    combined = dz - weight * dx
    band = np.flatnonzero(combined >= combined.max() - tolerance)
    along_edge = np.abs([rows[i]['candidate']['press_offset_xy'][1] for i in band])
    band = band[along_edge <= along_edge.min() + 1e-10]
    return int(band[np.argmin(dx[band])])


def heuristic_report(contacts: list[dict], weight: float = HARDWARE_WEIGHT,
                     tolerance: float = HARDWARE_TOLERANCE) -> dict:
    """Same scene schema as `selector.evaluate`, so `compare.summarize` applies.

    The rule has no probability output, so Brier score is None.
    """
    by_object: dict[str, list[dict]] = {}
    for row in contacts:
        by_object.setdefault(row['object'], []).append(row)
    scenes = []
    for name, rows in sorted(by_object.items()):
        labels = np.array([row['robust_feasible'] for row in rows], dtype=bool)
        pick = heuristic_index(rows, weight, tolerance)
        scenes.append({'object': name, 'split': rows[0]['split'], 'candidates': len(rows),
                       'robust_feasible': int(labels.sum()), 'oracle_success': bool(labels.any()),
                       'selected_index': rows[pick]['candidate']['index'] if pick is not None else None,
                       'selected_success': bool(labels[pick]) if pick is not None else None,
                       'selected_score': None, 'abstain_reason': None if pick is not None else 'no_candidates',
                       'brier_score': None})
    return {'weight': weight, 'tolerance': tolerance, 'scenes': scenes}
