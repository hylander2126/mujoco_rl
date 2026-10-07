"""Execute grid and cloud selectors on external rigid-object fixtures.

Sampled oracle success qualifies a scenario for selection accuracy. Zero oracle
success means unresolved feasibility, not proof that an object cannot tip.
All controller failures remain visible in the output.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from contextlib import redirect_stdout
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import mujoco
import numpy as np

from contact_selection.sim.dataset import write_json
from contact_selection.sim.external_objects import prepare_mesh
from contact_selection.sim.candidate_generator import generate_candidates, collision_hulls, upper_surface
from contact_selection.sim.rollout_evaluator import evaluate_rollout
from contact_selection.selection.features import extract_features
from contact_selection.selection.heuristic import heuristic_index
from contact_selection.selection.robust_press import rank_positions, select_press
from contact_selection.hardware.sim_cloud import simulate_cloud

GEOMETRY = dict(edge_margin_m=0.004, min_upward_normal=0.9,
                approach_tolerance_m=0.002, pivot_tolerance_m=0.006,
                proposal_multiplier=8, require_extended_edge=False)
THRESHOLDS = dict(min_arc_contact_fraction=0.9, max_pivot_drift_m=0.01,
                  max_off_axis_deg=3.0, joint_limit_tolerance_rad=0.01)


def passive_stability(model, initial_data, seconds=2.0):
    """Check the stationary-object precondition on a copy before any robot action."""
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, initial_data)
    pid = model.body('payload').id
    origin = data.xpos[pid].copy()
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    angle = float(np.degrees(2 * np.arccos(np.clip(abs(data.xquat[pid, 0]), 0, 1))))
    drift = float(np.linalg.norm(data.xpos[pid, :2] - origin[:2]))
    return dict(stable=angle < 5 and drift < 0.005, rotation_deg=angle, translation_xy_m=drift,
                duration_sec=seconds)


def validated_cloud_pick(model, data, cfg, cloud, geometry):
    """Cloud inference with geometry/IK validation, never an outcome lookup."""
    pid = model.body('payload').id
    hulls = collision_hulls(model, data, pid)
    accepted = []
    def validate(point, normal):
        hit = upper_surface(hulls, point[:2], 0, GEOMETRY['min_upward_normal'])
        if hit is None or point[0] <= geometry['pivot'][0]:
            return False
        candidates, _ = generate_candidates(model, data, 1, {**GEOMETRY, 'edge_margin_m': 0.0}, cfg,
                                            reference_points_xy=[point[:2]])
        if candidates and np.allclose(candidates[0].position[:2], point[:2], rtol=0, atol=1e-7):
            accepted.append(candidates[0])
            return True
        return False
    selected = select_press(cloud, table_z=0.05, com_xy=data.xipos[pid, :2], validator=validate)
    return selected, accepted[0] if accepted else None


def run_scene(job):
    path, output, friction, mass, count, seed, yaw, scale = job
    name = Path(path).stem
    directory = Path(output) / f'{name}_mu{friction:g}_m{mass:g}_yaw{yaw:g}_scale{scale:g}'
    directory.mkdir(parents=True, exist_ok=False)
    model, data, cfg, metadata = prepare_mesh(path, friction=friction, mass=mass, yaw=yaw, scale=scale)
    pid = model.body('payload').id
    stability = passive_stability(model, data)
    candidates, geometry = generate_candidates(model, data, count, GEOMETRY, cfg)
    geometry['com_world_m'] = data.xipos[pid].tolist()
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    np.savez_compressed(directory / 'initial_state.npz', state=state)
    mujoco.mj_saveModel(model, str(directory / 'model.mjb'))
    manifest = dict(object_name=name, controller=asdict(cfg), state_spec=int(spec),
                    geometry=geometry, fixture=metadata, candidates=[c.to_dict() for c in candidates],
                    geometry_filters=GEOMETRY, feasibility=THRESHOLDS, passive_stability=stability,
                    source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    write_json(directory / 'scene.json', manifest)
    if not stability['stable']:
        result = dict(object=name, fixture=metadata, candidates=0, grid_successes=0,
                      sampled_eligible=False, eligibility='invalid_unstable_initial_fixture',
                      passive_stability=stability, failures={}, picks={})
        write_json(directory / 'result.json', result)
        print(directory.name, 'unstable fixture', stability, flush=True)
        return result
    rows = []
    with (directory / 'rollouts.jsonl').open('w') as stream:
        for candidate in candidates:
            with (directory / 'controller.log').open('a') as log, redirect_stdout(log):
                outcome, arrays = evaluate_rollout(model, data, candidate, cfg, THRESHOLDS)
            np.savez_compressed(directory / f'candidate_{candidate.index:03}.npz', **arrays)
            row = dict(candidate=candidate.to_dict(), features=extract_features(candidate, geometry), **outcome)
            stream.write(json.dumps(row) + '\n'); stream.flush()
            rows.append(row)
    picks = {}
    if rows:
        for label, index in [('legacy_grid', heuristic_index(rows)), ('ratio_grid', rank_positions(
                [c.position for c in candidates], geometry['pivot'], data.xipos[pid, :2]))]:
            picks[label] = dict(candidate_index=index, feasible=rows[index]['feasible'] if index is not None else False)
    cloud, _, _ = simulate_cloud(model, data, pid, geometry['bounds'], seed=seed)
    np.save(directory / 'cloud.npy', cloud)
    from contact_selection.hardware.hardware_selector import select_contact_points
    for label in ('legacy_cloud', 'robust_cloud'):
        try:
            selected = (select_contact_points(cloud, table_z=0.05)['press'] if label == 'legacy_cloud'
                        else validated_cloud_pick(model, data, cfg, cloud, geometry)[0])
        except ValueError as exc:
            selected = dict(available=False, reason=str(exc))
        write_json(directory / f'{label}.json', selected)
        if not selected['available']:
            picks[label] = dict(feasible=False, rejected=selected['reason'])
            continue
        xy = np.asarray(selected['point'][:2])
        exact, report = generate_candidates(model, data, 1, {**GEOMETRY, 'edge_margin_m': 0.0}, cfg,
                                            reference_points_xy=[xy])
        if not exact or not np.allclose(exact[0].position[:2], xy, rtol=0, atol=1e-7):
            picks[label] = dict(feasible=False, rejected='robot_or_surface_filter', report=report)
            continue
        with (directory / 'controller.log').open('a') as log, redirect_stdout(log):
            outcome, arrays = evaluate_rollout(model, data, exact[0], cfg, THRESHOLDS)
        np.savez_compressed(directory / f'{label}.npz', **arrays)
        picks[label] = dict(point=selected['point'], projected_candidate=exact[0].to_dict(), **outcome)
    # Include executed cloud points as witnesses; never call an object untippable
    # just because a coarse grid missed a successful contact.
    successes = sum(r['feasible'] for r in rows)
    eligible = bool(successes or any(r['feasible'] for r in picks.values()))
    result = dict(object=name, fixture=metadata, passive_stability=stability, candidates=len(rows), grid_successes=successes,
                  sampled_eligible=eligible, eligibility='witnessed_success' if eligible else 'unresolved_no_successful_sample',
                  failures=dict(Counter(f for r in rows for f in r['failure_modes'])), picks=picks)
    write_json(directory / 'result.json', result)
    print(directory.name, 'oracle', successes, '/', len(rows),
          {k: v['feasible'] for k, v in picks.items()}, flush=True)
    return result


def summarize(results):
    eligible = [r for r in results if r['sampled_eligible']]
    return dict(scenarios=len(results), objects=len({r['object'] for r in results}),
        eligible_scenarios=len(eligible), unresolved_scenarios=len(results)-len(eligible),
        selectors={key: dict(successes=sum(r['picks'].get(key, {}).get('feasible', False) for r in eligible),
                            denominator=len(eligible))
                   for key in ['legacy_grid', 'ratio_grid', 'legacy_cloud', 'robust_cloud']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=Path('outputs/contact_selection/assets/ycb'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--objects', nargs='*')
    p.add_argument('--frictions', type=float, nargs='+', default=[0.3, 0.5])
    p.add_argument('--masses', type=float, nargs='+', default=[0.4])
    p.add_argument('--candidates', type=int, default=12)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--yaw', type=float, default=0)
    p.add_argument('--scale', type=float, default=1)
    args = p.parse_args()
    paths = sorted(args.assets.glob('*.stl'))
    if args.objects:
        paths = [path for path in paths if path.stem in args.objects]
    if not paths or args.workers < 1 or args.candidates < 1:
        p.error('Need meshes, positive workers and candidates')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    sources = sorted(Path('contact_selection').rglob('*.py')) + sorted(Path('parameter_estimation/controllers').glob('*.py'))
    write_json(args.output / 'provenance.json', dict(mujoco=mujoco.__version__, numpy=np.__version__,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}))
    jobs = [(str(path), str(args.output), mu, mass, args.candidates, args.seed, args.yaw, args.scale)
            for path in paths for mu in args.frictions for mass in args.masses]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        results = list(pool.map(run_scene, jobs))
    report = dict(summary=summarize(results), scenes=results)
    write_json(args.output / 'report.json', report)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
