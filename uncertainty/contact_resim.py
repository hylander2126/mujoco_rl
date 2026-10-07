"""Simulated feasibility labels for a few contact picks under camera pose error.

``contact_robustness`` perturbs the *belief* and scores it with a friction proxy.
This module gets the real label instead: the believed geometry stays nominal, and the
*true* object in the simulator is moved by a reconstruction-sized error (rigid xy
translation and yaw about its centroid). The robot acts on the belief:

- the press point is the candidate's believed world XY. ``press_offset_xy`` is
  re-expressed against the true top centre, because the FSM reads that from the
  simulator;
- the arc centre is the believed pivot plus its own noise, passed as
  ``arc_center_xz`` (on hardware this is ARC_CENTER from the camera).

Then the unchanged ``evaluate_rollout`` labels the trial. Draw 0 of each candidate is
unperturbed and must reproduce the nominal label. Per-vertex shape noise is not
applied: the true shape is the shape, and its effect on the belief is only the press
height, which SQUASH finds by force anyway.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from uncertainty.contact_robustness import GeometryNoise, Scene


def _run(job: tuple) -> dict:
    """Worker: reload the saved scene, move the true object, run one rollout."""
    from contact_selection.candidate_generator import Candidate
    from contact_selection.controller import PressPullConfig, PressPullFSM
    from contact_selection.rollout_evaluator import evaluate_rollout
    from mujoco_irb120.robot.controllers.robot import controller
    scene_dir, cand, cfg_dict, thresholds, centroid, pivot_xz, draw = job
    model = mujoco.MjModel.from_binary_path(str(Path(scene_dir) / 'model.mjb'))
    data = mujoco.MjData(model)
    state = np.load(Path(scene_dir) / 'state.npy')
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, state, spec)
    pid = int(model.site_bodyid[model.site('site:obj_frame').id])
    adr = int(model.jnt_qposadr[model.body_jntadr[pid]])
    Rz = Rotation.from_euler('z', draw['yaw_deg'], degrees=True)
    pos, quat = data.qpos[adr:adr + 3].copy(), data.qpos[adr + 3:adr + 7].copy()  # quat is wxyz
    true_qpos = [*(np.asarray(centroid) + Rz.apply(pos - centroid) + [draw['dx_m'], draw['dy_m'], 0.0]),
                 *(Rz * Rotation.from_quat(quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]]

    def reset():
        mujoco.mj_setState(model, data, state, spec)
        data.qpos[adr:adr + 7] = true_qpos
        mujoco.mj_forward(model, data)
    reset()
    cfg = PressPullConfig(**{**cfg_dict, 'arc_center_xz': (pivot_xz[0] + draw['pivot_dx_m'],
                                                           pivot_xz[1] + draw['pivot_dz_m'])})
    top = PressPullFSM(controller(model, data), model, data, cfg).object_top_center()
    reset()
    candidate = Candidate(**{**cand, 'press_offset_xy': (np.asarray(cand['position'][:2]) - top[:2]).tolist()})
    outcome, _ = evaluate_rollout(model, data, candidate, cfg, thresholds)
    return {'candidate': cand['index'], **draw, 'feasible': bool(outcome['feasible']),
            'failure_modes': outcome['failure_modes'],
            'max_intended_tip_deg': float(outcome['metrics']['max_intended_tip_deg']),
            'max_pivot_drift_mm': 1e3 * float(outcome['metrics']['max_pivot_drift_m'])}


def resimulate(scene: Scene, indices: list[int], draws: int, noise: GeometryNoise = GeometryNoise(),
               seed: int = 0, workers: int = 8) -> dict:
    """`draws` rollouts per candidate (the first unperturbed). Returns per-draw rows and P(feasible)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    sim = scene.sim
    model, data, cfg = sim['model'], sim['data'], sim['cfg']
    rng = np.random.default_rng(seed)
    centroid = np.concatenate(scene.hull_points).mean(0)
    centroid[2] = 0.0
    pivot_xz = (float(scene.pivot[0]), float(scene.pivot[2]))
    by_index = {c.index: c for c in scene.candidates}
    with tempfile.TemporaryDirectory() as tmp:
        mujoco.mj_saveModel(model, str(Path(tmp) / 'model.mjb'))
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        np.save(Path(tmp) / 'state.npy', state)
        jobs = []
        for idx in indices:
            for k in range(draws):
                z = rng.normal(size=5) if k else np.zeros(5)
                draw = {'draw': k, 'dx_m': 1e-3 * noise.pose_mm * z[0], 'dy_m': 1e-3 * noise.pose_mm * z[1],
                        'yaw_deg': noise.yaw_deg * z[2],
                        'pivot_dx_m': 1e-3 * noise.pivot_mm * z[3], 'pivot_dz_m': 1e-3 * noise.pivot_mm * z[4]}
                jobs.append((tmp, by_index[idx].to_dict(), asdict(cfg), sim['feasibility'], centroid.tolist(),
                             pivot_xz, draw))
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            rows = list(pool.map(_run, jobs))
    summary = {}
    for idx in indices:
        r = [row for row in rows if row['candidate'] == idx]
        perturbed = [row['feasible'] for row in r if row['draw'] > 0]
        modes = [m for row in r if not row['feasible'] for m in row['failure_modes']]
        summary[idx] = {'nominal_feasible': r[0]['feasible'], 'p_feasible_sim': float(np.mean(perturbed)),
                        'draws': len(perturbed),
                        'failure_modes': {m: modes.count(m) for m in sorted(set(modes))}}
    return {'scene': scene.name, 'noise': asdict(noise), 'seed': seed, 'summary': summary, 'rows': rows}


if __name__ == '__main__':
    import argparse
    from uncertainty.contact_robustness import build_scene, robustness
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='+', default=['box_mu_0p20.json', 'l_mu_0p25.json'])
    ap.add_argument('--draws', type=int, default=21, help='per candidate, including the unperturbed draw 0')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for name in args.configs:
        scene = build_scene(root / 'contact_selection' / 'config' / name)
        rob = robustness(scene, GeometryNoise(), 500, seed=0)
        picks = sorted({rob['nominal_pick'], rob['robust_pick']} - {None})
        res = resimulate(scene, picks, args.draws, workers=args.workers)
        for idx in picks:
            res['summary'][idx]['p_feasible_proxy'] = rob['candidates'][idx]['p_feasible']
            res['summary'][idx]['role'] = ('nominal' if idx == rob['nominal_pick'] else '') + \
                ('robust' if idx == rob['robust_pick'] else '')
        out[name] = res
        print(name, json.dumps(res['summary'], indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float))
