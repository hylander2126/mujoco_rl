"""Run the hardware perception-to-contact pipeline on simulated depth clouds.

Sim candidates and features use exact collision hulls and the true pivot. The
hardware sees a fused, segmented camera cloud and estimates normals and the
pivot from it (`hardware_selector.py`, a verbatim copy). This module ray-casts
that cloud from saved scenes so the two can be compared:

- parity: hardware normal/pivot estimates vs. simulator ground truth, and how
  far the selector features move when computed from the estimated pivot;
- end to end: `select_contact_points()` on the cloud, its press point mapped to
  the nearest labelled sim candidate, and with `--execute` actually pressed:
  the exact hardware point is run through the press-pull controller in every
  labelled physics scenario, giving its own robust (AND) label.

Camera model. `mj_multiRay` from each camera centre, on a regular angular grid
(no OpenGL needed), keeping hits on the payload body (perfect segmentation) and
adding Gaussian depth noise along the ray. Poses are APPROXIMATE: the hardware
extrinsics are not in this repo. They follow CONTACT_SELECTION.md (cam1/cam2 on
the robot side, cam3 on the far side) and are placed relative to the object.
The robot and table occlude as they would on hardware. Rays hit visual and
collision geoms alike, so the cloud follows the visual mesh while candidates
follow the collision hulls; the difference is part of what parity measures.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

# Camera centres relative to the object's bounding-box centre, metres (world frame).
CAMERAS = {'cam1': (-0.45, 0.30, 0.40), 'cam2': (-0.45, -0.30, 0.40), 'cam3': (0.45, 0.0, 0.35)}
# Sim candidates keep 6 mm from the silhouette (`edge_margin_m`), a sampling choice.
# The hardware rule (press_inset=0) deliberately presses at the near edge, so the
# executed pick uses this margin instead; IK and collision filters are unchanged.
EXECUTE_EDGE_MARGIN_M = 0.0
PIXEL_PITCH_M = 0.002   # ray spacing on the object at its distance (~RealSense after voxel filtering)
DEPTH_NOISE_M = 0.0015  # 1-sigma noise along the ray


def load_scene(scene_path: Path):
    import mujoco
    scene = json.loads(scene_path.read_text())
    model = mujoco.MjModel.from_binary_path(str(scene_path.parent / 'model.mjb'))
    data = mujoco.MjData(model)
    with np.load(scene_path.parent / 'initial_state.npz') as saved:
        mujoco.mj_setState(model, data, saved['state'], scene['state_spec'])
    mujoco.mj_forward(model, data)
    return scene, model, data


def payload_body(model) -> int:
    from mujoco_irb120.robot.controllers.robot import controller
    import mujoco
    data = mujoco.MjData(model)
    return controller(model, data).payload_body_id


def simulate_cloud(model, data, body_id: int, bounds, cameras=CAMERAS,
                   pitch=PIXEL_PITCH_M, noise=DEPTH_NOISE_M, seed=0):
    """Segmented, noisy object cloud plus ground-truth normals and source camera."""
    import mujoco
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(bounds)
    centre, radius = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
    points, normals, source = [], [], []
    for name, offset in cameras.items():
        origin = centre + np.asarray(offset)
        forward = centre - origin
        distance = np.linalg.norm(forward)
        forward /= distance
        right = np.cross(forward, [0, 0, 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        half = np.tan(np.arcsin(min(radius / distance, 0.99))) * 1.15
        n = int(np.ceil(2 * half * distance / pitch))
        u, v = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
        rays = forward + u.reshape(-1, 1) * right + v.reshape(-1, 1) * up
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        geom = np.zeros(len(rays), dtype=np.int32)
        dist = np.zeros(len(rays))
        normal = np.zeros(3 * len(rays))
        mujoco.mj_multiRay(model, data, origin, rays.reshape(-1), None, 1, -1,
                           geom, dist, normal, len(rays), 10.0)
        hit = (geom >= 0) & (dist > 0)
        hit[hit] &= model.geom_bodyid[geom[hit]] == body_id
        depth = dist[hit] + rng.normal(0, noise, hit.sum())
        points.append(origin + rays[hit] * depth[:, None])
        normals.append(normal.reshape(-1, 3)[hit])
        source += [name] * int(hit.sum())
    return np.concatenate(points), np.concatenate(normals), np.array(source)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def scene_parity(scene_path: Path, labels: dict[str, dict[int, bool]], seed=0) -> dict:
    """Parity and end-to-end pick for one saved scene.

    labels: scope -> {candidate index: robust label} for this object.
    """
    from contact_selection.hardware.hardware_selector import estimate_normals, estimate_pivot, select_contact_points
    from contact_selection.selection.features import pivot_ray_angle
    scene, model, data = load_scene(scene_path)
    geometry = scene['geometry']
    lo, hi = np.asarray(geometry['bounds'])
    table_z = float(lo[2])
    true_pivot = np.asarray(geometry['pivot'])
    cloud, true_normals, source = simulate_cloud(model, data, payload_body(model), geometry['bounds'], seed=seed)
    out = {'points': len(cloud), 'points_per_camera': {c: int((source == c).sum()) for c in CAMERAS},
           'table_z': table_z}

    # Normals: hardware PCA estimate vs ray-cast ground truth, on upward-facing points.
    estimated = estimate_normals(cloud)
    valid = np.isfinite(estimated).all(axis=1) & (true_normals[:, 2] >= 0.7)
    errors = angle_deg(estimated[valid], true_normals[valid])
    out['top_normal_error_deg'] = {'median': float(np.median(errors)), 'p90': float(np.percentile(errors, 90)),
                                   'valid_fraction': float(valid.sum() / max((true_normals[:, 2] >= 0.7).sum(), 1))}

    # Pivot: the press mode's estimate (preferred -X) vs the simulator's pivot.
    try:
        pivot = estimate_pivot(cloud, [-1., 0., 0.], table_z)
    except ValueError as exc:
        out['pivot'] = {'error': str(exc)}
        pivot = None
    else:
        out['pivot'] = {'kind': pivot['kind'], 'estimated': pivot['pivot'].tolist(), 'true': true_pivot.tolist(),
                        'dx_error_m': float(pivot['pivot'][0] - true_pivot[0]),
                        'direction': pivot['direction'].tolist()}

    # Features recomputed with the estimated pivot, over this scene's candidates.
    candidates = scene['candidates']
    positions = np.array([c['position'] for c in candidates])
    if pivot is not None:
        true_d = positions - true_pivot
        est_d = positions - pivot['pivot']
        true_ray = np.array([pivot_ray_angle(d[0], d[2]) for d in true_d])
        est_ray = np.array([pivot_ray_angle(d[0], d[2]) for d in est_d])
        out['feature_error'] = {'pivot_dx_m_max': float(np.max(np.abs(est_d[:, 0] - true_d[:, 0]))),
                                'pivot_ray_angle_deg_max': float(np.degrees(np.max(np.abs(est_ray - true_ray))))}

    # End to end: hardware press selection on the cloud -> nearest sim candidate.
    result = select_contact_points(cloud, table_z=table_z)
    press = result['press']
    if not press['available']:
        out['hardware_press'] = {'available': False, 'reason': press['reason']}
        return out
    point = np.asarray(press['point'])
    gaps = np.linalg.norm(positions[:, :2] - point[:2], axis=1)
    nearest = int(np.argmin(gaps))
    index = candidates[nearest]['index']
    out['hardware_press'] = {'available': True, 'point': point.tolist(), 'normal': press['normal'].tolist(),
                             'nearest_candidate': index, 'nearest_gap_m': float(gaps[nearest]),
                             'label_of_nearest': {scope: by_index[index] for scope, by_index in labels.items()}}
    return out


def hardware_candidate(scene_dir: Path, seed: int = 0) -> dict:
    """Cloud -> hardware press point -> a sim Candidate at that exact point.

    The point goes through the same `generate_candidates` filters as every sim
    candidate (surface, edge margin, pivot span, IK, approach collision), once
    with the dataset's own margin (reported as `default_filters`) and once with
    EXECUTE_EDGE_MARGIN_M, which decides whether it can be executed.
    Returns {candidate | None, point, default_filters, rejected}.
    """
    from contact_selection.sim.candidate_generator import generate_candidates
    from contact_selection.sim.controller import config_from_saved
    from contact_selection.hardware.hardware_selector import select_contact_points
    scene, model, data = load_scene(scene_dir / 'scene.json')
    config = json.loads((scene_dir.parent / 'config.json').read_text())
    lo = np.asarray(scene['geometry']['bounds'])[0]
    cloud, _, _ = simulate_cloud(model, data, payload_body(model), scene['geometry']['bounds'], seed=seed)
    press = select_contact_points(cloud, table_z=float(lo[2]))['press']
    if not press['available']:
        return {'candidate': None, 'point': None, 'rejected': f"hardware: {press['reason']}"}
    xy = np.asarray(press['point'][:2])
    cfg = config_from_saved(scene['controller'])

    def filtered(geometry):
        candidates, report = generate_candidates(model, data, 1, geometry, cfg, reference_points_xy=[xy])
        if candidates and np.linalg.norm(np.asarray(candidates[0].position[:2]) - xy) <= 1e-7:
            return candidates[0], None
        return None, next(iter(report.get('rejected', {})), report.get('scene_rejection', 'unknown'))

    candidate, reason = filtered({**config['geometry'], 'edge_margin_m': EXECUTE_EDGE_MARGIN_M})
    return {'candidate': candidate, 'point': press['point'].tolist(),
            'default_filters': filtered(config['geometry'])[1] or 'accepted',
            'rejected': None if candidate else f'sim filters: {reason}'}


def execute_hardware_pick(scene_dir: str, seed: int = 0) -> dict:
    """The hardware pick, executed once in this saved scenario snapshot (no trajectory saved)."""
    from contact_selection.sim.controller import config_from_saved
    from contact_selection.sim.rollout_evaluator import evaluate_rollout
    scene_dir = Path(scene_dir)
    pick = hardware_candidate(scene_dir, seed)
    out = {'dataset': scene_dir.parent.name, 'scene': scene_dir.name, 'hardware_point': pick['point'],
           'default_filters': pick.get('default_filters')}
    if pick['candidate'] is None:
        return {**out, 'feasible': False, 'rejected': pick['rejected']}
    scene, model, data = load_scene(scene_dir / 'scene.json')
    config = json.loads((scene_dir.parent / 'config.json').read_text())
    outcome, _ = evaluate_rollout(model, data, pick['candidate'], config_from_saved(scene['controller']),
                                  config['feasibility'])
    return {**out, 'feasible': outcome['feasible'], 'failure_modes': outcome['failure_modes'],
            'max_intended_tip_deg': outcome['metrics']['max_intended_tip_deg']}


def execute_all(scopes: dict[str, list[Path]], seed: int, workers: int) -> dict:
    """Robust label of the hardware pick per object and scope (AND over scenarios)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    scene_dirs = sorted({str(scene.parent) for dirs in scopes.values() for d in dirs
                         for scene in Path(d).glob('*/scene.json')})
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        results = dict(zip(scene_dirs, pool.map(execute_hardware_pick, scene_dirs, [seed] * len(scene_dirs))))
    for scene_dir, r in results.items():
        print(f"  {r['dataset']:28} {r['scene']:20} default filters: {r.get('default_filters', '-'):26} "
              f"feasible={r['feasible']} {r.get('rejected') or r.get('failure_modes')}", flush=True)
    by_object: dict[str, dict] = {}
    for scope, dirs in scopes.items():
        names = {Path(d).name for d in dirs}
        for scene_dir, r in results.items():
            if r['dataset'] in names:
                obj = r['scene'].rsplit('_trial_', 1)[0]
                entry = by_object.setdefault(obj, {}).setdefault(scope, {'robust_feasible': True, 'scenarios': 0})
                entry['robust_feasible'] &= bool(r['feasible'])
                entry['scenarios'] += 1
    return {'rollouts': results, 'robust': by_object}


def main():
    from contact_selection.commands.compare import SCOPES
    from contact_selection.sim.dataset import write_json
    from contact_selection.selection.selector import load_robust_contacts
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('suite', type=Path, help='Suite from `contact_selection rerun` (needs its saved scenes)')
    parser.add_argument('--extra', type=Path, nargs='+', default=[],
                        help='Extra dataset folders; each adds a stricter label scope (as in compare)')
    parser.add_argument('--labels', type=Path,
                        help='Folder holding the friction datasets to label with, when it differs from SUITE '
                             '(e.g. SUITE/features_ray after `refeature`)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--execute', action='store_true',
                        help='Also press the exact hardware point in every labelled scenario (minutes)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path,
                        help='JSON report (default: SUITE/cloud_parity_YYYY-MM-DD.json)')
    args = parser.parse_args()
    args.output = args.output or args.suite / f'cloud_parity_{date.today():%Y-%m-%d}.json'
    scopes = {'friction': [(args.labels or args.suite) / name for name in SCOPES['friction']]}
    scene_scopes = {'friction': [args.suite / name for name in SCOPES['friction']]}
    for folder in args.extra:
        extra = sorted(p.parent for p in folder.glob('*/rollouts.jsonl'))
        scopes[f'+{folder.name}'] = list(scopes.values())[-1] + extra
        scene_scopes[f'+{folder.name}'] = list(scene_scopes.values())[-1] + extra
    labels: dict[str, dict[str, dict[int, bool]]] = {}
    for scope, dirs in scopes.items():
        for row in load_robust_contacts(dirs)[0]:
            labels.setdefault(row['object'], {}).setdefault(scope, {})[row['candidate']['index']] = row['robust_feasible']
    report = {}
    for obj in sorted(labels):
        scene = next(iter(sorted(args.suite.glob(f'*_mu_0p50/{obj}_trial_01/scene.json'))), None) \
            or next(iter(sorted(args.suite.glob(f'*/{obj}_trial_01/scene.json'))))
        report[obj] = {'scene': str(scene), **scene_parity(scene, labels[obj], args.seed)}
        r = report[obj]
        hp = r['hardware_press']
        pv = r['pivot']
        print(f"{obj:11} pts={r['points']:6} normal err med/p90={r['top_normal_error_deg']['median']:.1f}/"
              f"{r['top_normal_error_deg']['p90']:.1f} deg  pivot dx err="
              f"{pv.get('dx_error_m', float('nan')) * 1000:+.1f} mm ({pv.get('kind', pv.get('error'))})  "
              + (f"press→cand {hp['nearest_candidate']} gap {hp['nearest_gap_m'] * 1000:.0f} mm "
                 f"labels {hp['label_of_nearest']}" if hp['available'] else f"press unavailable: {hp['reason']}"),
              flush=True)
    executed = None
    if args.execute:
        print('Pressing the hardware pick in every scenario:', flush=True)
        executed = execute_all(scene_scopes, args.seed, args.workers)
        for obj, by_scope in sorted(executed['robust'].items()):
            print(f"{obj:11} hardware pick robust: " + ', '.join(
                f"{scope}={v['robust_feasible']} ({v['scenarios']})" for scope, v in by_scope.items()))
    write_json(args.output, {'suite': str(args.suite), 'cameras': CAMERAS, 'pixel_pitch_m': PIXEL_PITCH_M,
                             'depth_noise_m': DEPTH_NOISE_M, 'seed': args.seed, 'objects': report,
                             'executed': executed})
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
