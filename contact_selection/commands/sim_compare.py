"""Simulate the heuristic's and the learned selectors' picks side by side, and show them.

For each object, in its hardest saved scenario (default: table friction 0.15), four
press points are executed with the press-pull controller (constant wrist orientation):

  - Heuristic, hardware pipeline: simulated camera cloud -> select_contact_points()
  - Heuristic, on the sim candidate grid: the same dz - w*dx rule over grid candidates
  - Logistic, 15 and 16 features: the learned selectors over the same grid

Output: one figure with a rendered side view at peak tilt per pick, and tilt vs time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

OBJECTS = ['box', 'L', 'heart', 'flashlight', 'soda', 'monitor']
SELECTORS = {  # key: (label, colour) -- reference palette slots / ink
    'hardware': ('Heuristic · hardware pipeline', '#0b0b0b'),
    'heuristic': ('Heuristic · sim grid', '#2a78d6'),
    'logistic15': ('Learned · 15 features', '#eb6834'),
    'logistic16': ('Learned · 16 features', '#1baf7a'),
}
FIGURES = Path(__file__).resolve().parents[1] / 'figures'


def picks(scene_dir: Path, models: dict) -> dict:
    """Each selector's Candidate for this scene, or a reason it has none."""
    from contact_selection.hardware.sim_cloud import hardware_candidate
    from contact_selection.selection.features import extract_features
    from contact_selection.selection.heuristic import heuristic_index
    from contact_selection.selection.selector import predict_and_select
    from contact_selection.sim.candidate_generator import Candidate
    scene = json.loads((scene_dir / 'scene.json').read_text())
    candidates = [Candidate(**c) for c in scene['candidates']]
    rows = [{'candidate': c.to_dict(), 'features': extract_features(c, scene['geometry'])} for c in candidates]
    out = {}
    hw = hardware_candidate(scene_dir)
    out['hardware'] = (hw['candidate'], hw['rejected'])
    out['heuristic'] = (candidates[heuristic_index(rows)], None)
    for key, model in models.items():
        chosen = predict_and_select(model, candidates, scene['geometry'])
        c = chosen['selected_candidate']
        out[key] = (Candidate(**c) if c else None, None if c else f"abstains ({chosen['reason']})")
    return out


def run(job) -> dict:
    """One rollout with a frame rendered every 0.1 s; keeps the frame at peak tilt."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from contact_selection.hardware.sim_cloud import load_scene
    from contact_selection.sim.controller import config_from_saved
    from contact_selection.sim.rollout_evaluator import evaluate_rollout
    scene_dir, candidate = Path(job[0]), job[1]
    scene, model, data = load_scene(scene_dir / 'scene.json')
    config = json.loads((scene_dir.parent / 'config.json').read_text())
    lo, hi = np.asarray(scene['geometry']['bounds'])
    renderer = mujoco.Renderer(model, height=360, width=480)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [(lo[0] + hi[0]) / 2 - 0.05, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]
    camera.distance = 0.45 + 1.6 * np.linalg.norm(hi - lo) / 2
    camera.azimuth, camera.elevation = 90, -10
    r0 = data.xmat[model.site_bodyid[model.site('site:obj_frame').id]].reshape(3, 3).copy()
    body = model.site_bodyid[model.site('site:obj_frame').id]
    state = {'next': 0.0, 'best': -1.0, 'frame': None}

    def capture(m, d, _):
        if d.time + 1e-9 < state['next']:
            return
        state['next'] += 0.1
        tilt = np.degrees(Rotation.from_matrix(d.xmat[body].reshape(3, 3) @ r0.T).magnitude())
        if tilt >= state['best']:
            renderer.update_scene(d, camera=camera)
            state['best'], state['frame'] = tilt, renderer.render().copy()

    from contact_selection.sim.candidate_generator import Candidate
    outcome, arrays = evaluate_rollout(model, data, Candidate(**candidate), config_from_saved(scene['controller']),
                                       config['feasibility'], step_callback=capture)
    rotations = arrays['object_rotation_world']
    tilt = np.degrees(Rotation.from_matrix(rotations @ rotations[0].T).magnitude())
    t = arrays['diagnostic_time']
    keep = slice(None, None, max(1, len(t) // 600))
    return {'feasible': outcome['feasible'], 'failure_modes': outcome['failure_modes'],
            't': t[keep] - t[0], 'tilt': tilt[keep], 'frame': state['frame'], 'peak_tilt': float(tilt.max())}


def scenario_scene(roots: list[Path], scenario: str, obj: str) -> Path:
    for root in roots:
        found = sorted(root.glob(f'**/{scenario}/{obj}_trial_01/scene.json'))
        if found:
            return found[0].parent
    raise FileNotFoundError(f'No {scenario} scene for {obj} under {roots}')


def plot(results: dict, scenario: str, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    surface, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    good, bad, neutral = '#0ca30c', '#d03b3b', '#a3a29c'
    plt.rcParams.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'font.size': 9,
                         'axes.edgecolor': grid, 'xtick.color': muted, 'ytick.color': muted,
                         'axes.labelcolor': muted, 'axes.spines.top': False, 'axes.spines.right': False})
    keys = list(SELECTORS)
    fig, axes = plt.subplots(len(OBJECTS), len(keys) + 1, figsize=(17, 2.9 * len(OBJECTS)),
                             gridspec_kw={'width_ratios': [1] * len(keys) + [1.35]})
    for row, obj in enumerate(OBJECTS):
        for col, key in enumerate(keys):
            ax = axes[row, col]
            ax.set_xticks([]), ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            r = results[obj][key]
            if 'skipped' in r:
                ax.set_facecolor('#f0efec')
                ax.text(0.5, 0.5, r['skipped'], ha='center', va='center', color=muted, wrap=True,
                        transform=ax.transAxes)
                badge, colour = 'no pick', neutral
            else:
                ax.imshow(r['frame'])
                badge, colour = ('PASS', good) if r['feasible'] else ('FAIL', bad)
                ax.text(0.02, 0.04, f"peak tilt {r['peak_tilt']:.1f}°", transform=ax.transAxes,
                        color='white', fontsize=8, bbox=dict(facecolor=ink, alpha=0.55, lw=0, pad=2))
            ax.text(0.98, 0.96, badge, transform=ax.transAxes, ha='right', va='top', color='white',
                    fontweight='bold', fontsize=9, bbox=dict(facecolor=colour, lw=0, pad=3))
            if row == 0:
                ax.set_title(SELECTORS[key][0], fontsize=10, color=SELECTORS[key][1], fontweight='bold')
            if col == 0:
                ax.set_ylabel(obj, fontsize=11, color=ink, rotation=0, ha='right', va='center', labelpad=10)
        ax = axes[row, -1]
        for key in keys:
            r = results[obj][key]
            if 'skipped' not in r:
                ax.plot(r['t'], r['tilt'], color=SELECTORS[key][1], lw=2 if key == 'hardware' else 1.5,
                        ls='-' if r['feasible'] else (0, (4, 2)))
        ax.grid(color=grid, lw=0.5)
        ax.set_ylabel('tilt (°)')
        if row == len(OBJECTS) - 1:
            ax.set_xlabel('time (s)')
        if row == 0:
            ax.set_title('Object tilt  (solid = pass, dashed = fail)', fontsize=10, color=ink)
    fig.suptitle(f'Heuristic vs learned selector, simulated at table friction {scenario.split("mu_")[-1].replace("p", ".")}'
                 '  ·  side view at peak tilt  ·  wrist orientation held constant', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('roots', type=Path, nargs='+', help='Suite/probe folders holding the scenario datasets')
    parser.add_argument('--scenario', default='*mu_0p15', help='Dataset folder pattern (default: *mu_0p15)')
    parser.add_argument('--model15', type=Path, required=True)
    parser.add_argument('--model16', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--output', type=Path, default=FIGURES / 'sim_compare.png')
    args = parser.parse_args()
    models = {'logistic15': json.loads(args.model15.read_text()), 'logistic16': json.loads(args.model16.read_text())}
    jobs, slots, results = [], [], {obj: {} for obj in OBJECTS}
    for obj in OBJECTS:
        scene_dir = scenario_scene(args.roots, args.scenario, obj)
        for key, (candidate, reason) in picks(scene_dir, models).items():
            if candidate is None:
                results[obj][key] = {'skipped': reason}
            else:
                jobs.append((str(scene_dir), candidate.to_dict()))
                slots.append((obj, key))
        print(f'{obj}: {scene_dir.parent.name}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for (obj, key), result in zip(slots, pool.map(run, jobs)):
            results[obj][key] = result
            print(f"  {obj:10} {key:11} feasible={result['feasible']} peak={result['peak_tilt']:.1f}° "
                  f"{result['failure_modes']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(results, args.scenario.strip('*'), args.output)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
