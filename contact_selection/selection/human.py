"""Human ground-truth contacts: where a person would press, which edge to tip over, how to pull.

Read from config/human_contacts.json. All coordinates are in cm in one fixed frame per
object, whatever edge is chosen: origin at the default robot-side tipping edge
(the simulator's press-pull pivot), x away from the robot, y to the robot's left.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HUMAN = Path(__file__).resolve().parents[1] / 'config' / 'human_contacts.json'


@dataclass(frozen=True)
class HumanContact:
    contact_cm: tuple[float, float]
    tipping_edge_cm: tuple[tuple[float, float], tuple[float, float]] | None  # None -> default edge
    pull_direction: tuple[float, float]  # unit vector in the same frame
    pull_through_com: bool
    note: str


def _unit(v) -> tuple[float, float]:
    v = np.asarray(v, dtype=float)
    if v.shape != (2,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-9:
        raise ValueError(f'pull_direction must be a nonzero [dx, dy], got {v}')
    v = v / np.linalg.norm(v)
    return float(v[0]), float(v[1])


def resolve(entry: dict, com_cm: tuple[float, float]) -> HumanContact | None:
    """One object's entry, with defaults applied. None if no contact is given yet.

    pull_direction:
      null           -> toward the robot, [-1, 0] (the default edge's pull)
      [dx, dy]       -> that horizontal direction
      "through_com"  -> along the line from the 2D CoM through the contact, pointing
                        toward the tipping side, so the pull's line of action passes
                        through the CoM (no yaw moment from the pull)
    """
    if entry.get('x_cm') is None or entry.get('y_cm') is None:
        return None
    contact = (float(entry['x_cm']), float(entry['y_cm']))
    edge = entry.get('tipping_edge_cm')
    if edge is not None:
        edge = np.asarray(edge, dtype=float)
        if edge.shape != (2, 2) or not np.isfinite(edge).all() or np.linalg.norm(edge[1] - edge[0]) < 0.5:
            raise ValueError('tipping_edge_cm must be [[x1, y1], [x2, y2]], at least 0.5 cm apart')
        edge = tuple(map(tuple, edge.tolist()))
    pull = entry.get('pull_direction')
    through = pull == 'through_com'
    if pull is None:
        direction = (-1.0, 0.0)
    elif through:
        away = np.subtract(contact, com_cm)
        if np.linalg.norm(away) < 0.5:
            raise ValueError('"through_com" needs the contact at least 0.5 cm from the CoM')
        if edge is not None:  # point toward the chosen edge's side
            toward_edge = np.mean(edge, axis=0) - np.asarray(com_cm)
        else:
            toward_edge = np.array([-1.0, 0.0])
        direction = _unit(away if away @ toward_edge >= 0 else -away)
    else:
        direction = _unit(pull)
    return HumanContact(contact, edge, direction, through, entry.get('note', ''))


def load(path: Path = HUMAN) -> dict[str, dict]:
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith('_')}
