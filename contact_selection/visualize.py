"""Offline spatial diagnostics and deterministic viewer replay, separate from training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from contact_selection.dataset import read_records, write_json


def summarize(directory: Path) -> dict:
    """Include empty candidate sets in scene coverage and oracle denominators."""
    records = read_records(directory / 'rollouts.jsonl')
    grouped = defaultdict(list)
    for record in records:
        grouped[record['candidate_set_id']].append(record)
    split_counts = {split: {'feasible': 0, 'infeasible': 0} for split in ('train', 'validation', 'test')}
    for record in records:
        split_counts[record['object_split']]['feasible' if record['feasible'] else 'infeasible'] += 1
    scenes = []
    for path in sorted(directory.glob('*/scene.json')):
        manifest = json.loads(path.read_text())
        rows = grouped[manifest['candidate_set_id']]
        labels = np.array([r['feasible'] for r in rows], dtype=bool)
        # Current heuristic: centre of the geom AABB top, approximated by the
        # nearest valid candidate if centre was geometrically rejected.
        heuristic = min(rows, key=lambda r: np.linalg.norm(r['candidate']['press_offset_xy'])) if rows else None
        reference_rows = [r for r in rows if r.get('is_reference_contact', False)]
        scenes.append({'scene': manifest['candidate_set_id'], 'object': manifest['object_name'],
                       'simulation_preset': manifest.get('simulation_preset', {'name': 'legacy'}),
                       'reference_contact_included': bool(manifest.get('reference_candidate_indices', [])),
                       'reference_contact_success': bool(all(r['feasible'] for r in reference_rows)) if reference_rows else None,
                       'candidates': len(rows), 'feasible': int(labels.sum()),
                       'expected_candidates': len(manifest['candidates']),
                       'complete': len(rows) == len(manifest['candidates']),
                       'oracle_success': bool(labels.any()),
                       'random_expected_success': float(labels.mean()) if len(labels) else 0.0,
                       'heuristic_success': bool(heuristic['feasible']) if heuristic else False,
                       'mixed_labels': bool(labels.any() and not labels.all()),
                       'scene_rejection': manifest['geometry'].get('scene_rejection'),
                       'failure_counts': dict(Counter(f for r in rows for f in r['failure_modes']))})
    report = {'scenes': scenes, 'rollouts': len(records),
              'split_label_counts': split_counts,
              'oracle_scene_success': float(np.mean([s['oracle_success'] for s in scenes])) if scenes else None,
              'random_scene_success': float(np.mean([s['random_expected_success'] for s in scenes])) if scenes else None,
              'heuristic_scene_success': float(np.mean([s['heuristic_success'] for s in scenes])) if scenes else None,
              'training_gate': 'not_validated',
              'gate_reason': 'Inspect mixed labels, repeatability, and object coverage before training; a pilot alone is insufficient.'}
    if not all(split_counts['train'].values()):
        report.update(training_gate='blocked_single_class_training_data',
                      gate_reason='Training objects do not contain both feasibility classes. Do not move held-out candidates into training to hide this.')
    write_json(directory / 'summary.json', report)
    return report


def plot(directory: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    records = read_records(directory / 'rollouts.jsonl')
    for path in sorted(directory.glob('*/scene.json')):
        scene = json.loads(path.read_text())
        rows = [r for r in records if r['candidate_set_id'] == scene['candidate_set_id']]
        fig, axes = plt.subplots(1, 4, figsize=(19, 4), constrained_layout=True)
        for ax in axes:
            for vertices in scene['geometry']['hulls']:
                xy = np.unique(np.array(vertices)[:, :2], axis=0)
                outline = xy[ConvexHull(xy).vertices]
                ax.fill(outline[:, 0], outline[:, 1], color='lightgray', alpha=0.4)
            pivot = scene['geometry']['pivot']
            ax.axvline(pivot[0], color='black', linestyle=':', label='configured pivot X')
            ax.set(xlabel='world X (m)', ylabel='world Y (m)', aspect='equal')
        if rows:
            xy = np.array([r['candidate']['position'][:2] for r in rows])
            labels = [r['feasible'] for r in rows]
            axes[0].scatter(*xy.T, c=['tab:green' if f else 'tab:red' for f in labels], s=70)
            for row, (x, y) in zip(rows, xy):
                axes[0].annotate(str(row['candidate']['index']), (x, y), xytext=(4, 4), textcoords='offset points')
            heuristic = int(np.argmin([np.linalg.norm(r['candidate']['press_offset_xy']) for r in rows]))
            axes[0].scatter(*xy[heuristic], marker='*', s=240, facecolors='none', edgecolors='black', label='centre heuristic')
            reference_xy = [r['candidate']['position'][:2] for r in rows if r.get('is_reference_contact', False)]
            if reference_xy:
                axes[0].scatter(*np.asarray(reference_xy).T, marker='D', s=160, facecolors='none',
                                edgecolors='tab:blue', linewidths=1.5, label='demo reference')
            for ax, metric, title in zip(axes[1:], ['max_intended_tip_deg', 'arc_contact_fraction', 'max_off_axis_deg'],
                                         ['Intended tip (degrees)', 'ARC ball-contact fraction', 'Off-axis rotation (degrees)']):
                values = [r['metrics'][metric] for r in rows]
                limits = {'vmin': 0, 'vmax': 1} if metric == 'arc_contact_fraction' else {}
                scatter = ax.scatter(*xy.T, c=values, cmap='viridis', s=70, **limits)
                fig.colorbar(scatter, ax=ax)
                ax.set_title(title)
            axes[0].set_title('Actual feasibility: green=yes, red=no')
            axes[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.30))
        else:
            axes[0].set_title('No compatible candidates')
            axes[1].text(0.02, 0.5, scene['geometry'].get('scene_rejection', 'all proposals rejected').replace('_', '\n'),
                         transform=axes[1].transAxes)
        preset = scene.get('simulation_preset', {}).get('name', 'legacy')
        fig.suptitle(f"{scene['candidate_set_id']} — {preset}")
        fig.savefig(path.parent / 'contacts.png', dpi=160, bbox_inches='tight')
        plt.close(fig)


def replay(scene_path: Path, candidate_index: int, show_viewer: bool = False):
    """Replay an explicitly chosen pilot candidate from its saved MJB and full state."""
    import mujoco
    from contact_selection.candidate_generator import Candidate
    from contact_selection.rollout_evaluator import evaluate_rollout
    from parameter_estimation.controllers.press_pull_fsm import PressPullConfig

    scene = json.loads(scene_path.read_text())
    config = json.loads((scene_path.parent.parent / 'config.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(scene_path.parent / 'model.mjb'))
    data = mujoco.MjData(model)
    with np.load(scene_path.parent / 'initial_state.npz') as saved:
        mujoco.mj_setState(model, data, saved['state'], scene['state_spec'])
    mujoco.mj_forward(model, data)
    candidate = Candidate(**next(c for c in scene['candidates'] if c['index'] == candidate_index))
    callback = None
    viewer = None
    viewer_data = None
    if show_viewer:
        import mujoco.viewer
        viewer_data = mujoco.MjData(model)
        mujoco.mj_copyData(viewer_data, model, data)
        viewer = mujoco.viewer.launch_passive(model, viewer_data)

        def callback(m, d, c):
            if not viewer.is_running():
                raise KeyboardInterrupt('Viewer closed')
            mujoco.mj_copyData(viewer_data, m, d)
            if round(d.time / m.opt.timestep) % 20 == 0:
                viewer.sync()
    try:
        return evaluate_rollout(model, data, candidate, PressPullConfig(**scene['controller']),
                                config['feasibility'], step_callback=callback)
    finally:
        if viewer is not None:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--replay-scene', help='Candidate-set directory name')
    parser.add_argument('--candidate', type=int, default=0)
    parser.add_argument('--show-viewer', action='store_true')
    args = parser.parse_args()
    if args.replay_scene:
        result, _ = replay(args.directory / args.replay_scene / 'scene.json', args.candidate, args.show_viewer)
        print(result)
    else:
        plot(args.directory)
        print(json.dumps(summarize(args.directory), indent=2))


if __name__ == '__main__':
    main()
