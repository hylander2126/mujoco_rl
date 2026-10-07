"""Robustness of discrete contact candidates to reconstructed-geometry error.

Candidates are generated once on the nominal scene by the unchanged
``contact_selection.candidate_generator.generate_candidates`` (which also runs
the IK and approach-collision checks). Each Monte Carlo draw then perturbs what
a camera would report: hull vertices, the object's rigid pose, and the pivot.
The geometry-dependent parts of the pipeline run again on that belief at the
same fixed world XY points:

- ``upper_surface`` (on a top face, clear of the silhouette by ``edge_margin_m``,
  upward normal), the near-X support edge (``valid_pivot``) and pivot-span checks.
  These mirror ``generate_candidates`` exactly; they live inline there, so they are
  repeated here rather than edited in place.
- ``extract_features`` on the perturbed position, normal, pivot and bounds.
- Score: the trained selector (``score_contacts``) when a ``model.json`` is
  given. Otherwise the friction margin ``mu_table - tan(pivot_ray_angle)``, from
  the failure mechanism documented in ``features.pivot_ray_angle``.

Feasible means it passes the geometric checks *and* ``tan(ray angle) < mu_table``.
That second term is a proxy for the simulated label, not the label: re-simulating
every draw is out of scope. It is conservative: it passes 14/25 nominal box contacts
at mu = 0.2 and 5/12 on the L at 0.25, where the saved sim sweeps passed 20/25 and
8/12 (``features.py``
notes the threshold only approaches tan < mu for light objects or hard presses).
The geometry-only probability is reported alongside so the proxy can be discounted.
``contact_resim`` gets the real labels for a few picks: at the default noise the
proxy's 0.59 (box candidate 0) and 0.77 (L candidate 6) were 1.0 in simulation. IK, joint margins and approach collisions are robot-side
and held at their nominal values.

Positions are never averaged. Each candidate gets a feasibility probability and
a selection frequency, and the robust pick is an existing candidate.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from contact_selection.candidate_generator import Candidate, collision_hulls, generate_candidates, upper_surface
from contact_selection.features import extract_features, pivot_ray_angle
from contact_selection.generate import prepare_experiment_scene, set_initial_object_position
from contact_selection.scene import OBJECTS


@dataclass(frozen=True)
class GeometryNoise:
    """One-sigma reconstruction errors (camera). Defaults are assumptions, not calibrations."""
    vertex_mm: float = 1.0   # independent per-vertex surface noise, xyz
    pose_mm: float = 2.0     # rigid object translation, xy (z is pinned by the table)
    yaw_deg: float = 1.0     # rigid rotation about vertical through the centroid
    pivot_mm: float = 2.0    # pivot x and z, on top of the rigid motion

    def scaled(self, k: float) -> GeometryNoise:
        return GeometryNoise(*(k * v for v in asdict(self).values()))


@dataclass
class Scene:
    name: str
    candidates: list[Candidate]
    hull_points: list[np.ndarray]   # world vertices per collision hull
    pivot: np.ndarray
    body_pos: np.ndarray
    body_R: np.ndarray
    geometry_cfg: dict
    mu: float
    pivot_override: bool            # pivot from arc_center_xz rather than the object site
    sim: dict | None = field(default=None, repr=False)  # model, data, cfg, feasibility (for contact_resim)


def build_scene(config_path: Path, object_id: int | None = None) -> Scene:
    """Nominal candidates for one object of a contact-selection config, prepared like ``generate``."""
    config = json.loads(config_path.read_text())
    object_id = config['objects'][0] if object_id is None else object_id
    name = OBJECTS[object_id]
    preset = config['simulation_preset']
    model, data, base_cfg, reference, _ = prepare_experiment_scene(object_id, preset)
    pid = int(model.site_bodyid[model.site('site:obj_frame').id])
    initial = config.get('initial_object_position_by_object', {}).get(name)
    if initial is not None:
        set_initial_object_position(model, data, initial)
    mujoco.mj_forward(model, data)
    if config.get('center_com_y', False):
        position = data.xpos[pid].copy()
        position[1] -= data.xipos[pid, 1]
        set_initial_object_position(model, data, position)
    cfg = replace(base_cfg, **{**config['controller'], **config.get('controller_by_object', {}).get(name, {})})
    candidates, report = generate_candidates(
        model, data, config['candidates'], config['geometry'], cfg,
        reference_points_xy=[reference.position[:2]] if reference is not None else None)
    if not candidates:
        raise ValueError(f'{config_path.name}: no nominal candidates ({report.get("scene_rejection")})')
    hulls = collision_hulls(model, data, pid)
    mu = max(model.geom_friction[model.geom('table').id, 0], *model.geom_friction[model.geom_bodyid == pid, 0])
    return Scene(f'{config_path.stem}:{name}', candidates, [h.points[h.vertices] for h in hulls],
                 np.asarray(report['pivot']), data.xpos[pid].copy(), data.xmat[pid].reshape(3, 3).copy(),
                 config['geometry'], float(mu), cfg.arc_center_xz is not None,
                 {'model': model, 'data': data, 'cfg': cfg, 'feasibility': config['feasibility']})


def perturb_geometry(scene: Scene, noise: GeometryNoise, rng: np.random.Generator):
    """Return (hulls, pivot, body_pos, body_R) as a noisy reconstruction would report them."""
    allpts = np.concatenate(scene.hull_points)
    c = allpts.mean(0)
    R = Rotation.from_euler('z', rng.normal(0, noise.yaw_deg), degrees=True).as_matrix()
    t = 1e-3 * rng.normal(0, noise.pose_mm, 3) * np.array([1.0, 1.0, 0.0])

    def rigid(p):
        return (p - c) @ R.T + c + t
    hulls = [ConvexHull(rigid(p) + 1e-3 * rng.normal(0, noise.vertex_mm, p.shape)) for p in scene.hull_points]
    pivot = rigid(scene.pivot) + 1e-3 * rng.normal(0, noise.pivot_mm, 3) * np.array([1.0, 0.0, 1.0])
    return hulls, pivot, rigid(scene.body_pos), R @ scene.body_R


def evaluate_geometry(scene: Scene, hulls, pivot, body_pos, body_R, selector: dict | None) -> dict:
    """Feasibility, score and boundary slack for every candidate on one geometry belief."""
    from contact_selection.selector import score_contacts
    g = scene.geometry_cfg
    margin = g['edge_margin_m']
    vertices = np.concatenate([h.points[h.vertices] for h in hulls])
    lo, hi = vertices.min(0), vertices.max(0)
    if scene.pivot_override:
        pivot = pivot.copy()
        pivot[1] = (lo[1] + hi[1]) / 2  # generate_candidates re-centres an overridden pivot in Y
    bottom = vertices[vertices[:, 2] <= lo[2] + g['pivot_tolerance_m']]
    near = bottom[bottom[:, 0] <= bottom[:, 0].min() + g['pivot_tolerance_m']]
    valid_pivot = (abs(pivot[0] - near[:, 0].min()) <= g['pivot_tolerance_m']
                   and abs(pivot[2] - lo[2]) <= g['pivot_tolerance_m']
                   and near[:, 1].max() - near[:, 1].min() > 2 * margin)
    top_xy = ConvexHull(vertices[:, :2])
    n = len(scene.candidates)
    geom_ok, feasible = np.zeros(n, bool), np.zeros(n, bool)
    slack, scores, rows = np.full(n, -np.inf), np.full(n, -np.inf), []
    for i, cand in enumerate(scene.candidates):
        xy = np.asarray(cand.position[:2])
        hit = upper_surface(hulls, xy, margin, g['min_upward_normal'])
        p, normal = hit if hit is not None else (np.array([*xy, hi[2]]), np.array([0.0, 0.0, 1.0]))
        dx, dz = p[0] - pivot[0], p[2] - pivot[2]
        # Slack in metres to each boundary; the minimum is the distance to the nearest one.
        edge = -np.max(top_xy.equations[:, :2] @ xy + top_xy.equations[:, 2]) - margin
        span = min(dx, p[1] - (near[:, 1].min() + margin), near[:, 1].max() - margin - p[1])
        friction = scene.mu * dz - dx  # horizontal room before tan(angle) reaches mu
        slack[i] = min(edge, span, friction)
        # Same comparisons as generate_candidates: strictly past the pivot, inclusive Y span.
        geom_ok[i] = (valid_pivot and hit is not None and dx > 0
                      and near[:, 1].min() + margin <= p[1] <= near[:, 1].max() - margin)
        feasible[i] = geom_ok[i] and friction > 0
        local = body_R.T @ (p - body_pos)
        rows.append({'candidate': cand.to_dict(), 'features': extract_features(
            replace(cand, position=p.tolist(), normal=np.asarray(normal).tolist(), object_position=local.tolist()),
            {'bounds': [lo, hi], 'pivot': pivot})})
        scores[i] = scene.mu - np.tan(pivot_ray_angle(dx, dz))
    if selector is not None:
        scores = score_contacts(selector, rows)
    return {'feasible': feasible, 'geometry_ok': geom_ok, 'score': scores, 'slack': slack, 'valid_pivot': bool(valid_pivot)}


def select(result: dict, threshold: float | None) -> int | None:
    """Highest-scoring feasible candidate (above the selector threshold, if any), or abstain."""
    eligible = result['feasible'] & (result['score'] >= threshold if threshold is not None else True)
    idx = np.flatnonzero(eligible)
    return int(idx[np.argmax(result['score'][idx])]) if len(idx) else None


def robustness(scene: Scene, noise: GeometryNoise, samples: int, seed: int = 0,
               selector: dict | None = None, threshold: float = 0.5, p_min: float = 0.95) -> dict:
    """Per-candidate P(feasible), selection frequency and nominal score; nominal vs robust pick."""
    thr = threshold if selector is not None else None
    nominal = evaluate_geometry(scene, [ConvexHull(p) for p in scene.hull_points], scene.pivot,
                                scene.body_pos, scene.body_R, selector)
    rng = np.random.default_rng(seed)
    n = len(scene.candidates)
    feas, geom, picks, abstain, scene_rejections = np.zeros(n), np.zeros(n), np.zeros(n), 0, 0
    for _ in range(samples):
        res = evaluate_geometry(scene, *perturb_geometry(scene, noise, rng), selector)
        feas += res['feasible']
        geom += res['geometry_ok']
        scene_rejections += not res['valid_pivot']
        k = select(res, thr)
        if k is None:
            abstain += 1
        else:
            picks[k] += 1
    p_feas, freq = feas / samples, picks / samples
    nominal_pick = select(nominal, thr)
    robust_set = np.flatnonzero(p_feas >= p_min)
    robust_pick = (int(robust_set[np.argmax(nominal['score'][robust_set])]) if len(robust_set)
                   else int(np.argmax(p_feas)))
    table = [{'index': c.index, 'x_m': c.position[0], 'y_m': c.position[1], 'z_m': c.position[2],
              'nominal_feasible': bool(nominal['feasible'][i]), 'nominal_score': float(nominal['score'][i]),
              'boundary_slack_mm': float(1e3 * nominal['slack'][i]),
              'p_feasible': float(p_feas[i]), 'p_geometry_ok': float(geom[i] / samples),
              'selection_freq': float(freq[i])}
             for i, c in enumerate(scene.candidates)]
    return {'scene': scene.name, 'noise': asdict(noise), 'samples': samples, 'mu_table': scene.mu,
            'score_source': 'selector model.json' if selector is not None
            else 'friction margin mu - tan(ray angle) (proxy; no selector given)',
            'feasibility': 'geometric checks + tan(ray angle) < mu_table (proxy, not the sim label)',
            'candidates': table, 'abstain_freq': abstain / samples,
            'scene_rejection_freq': scene_rejections / samples,
            'nominal_pick': nominal_pick,
            'nominal_pick_p_feasible': float(p_feas[nominal_pick]) if nominal_pick is not None else None,
            'robust_pick': robust_pick, 'robust_pick_p_feasible': float(p_feas[robust_pick]),
            'robust_rule': f'highest nominal score among candidates with P(feasible) >= {p_min}'}
