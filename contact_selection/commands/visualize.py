"""Offline spatial diagnostics and deterministic viewer replay, separate from training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from contact_selection.sim.dataset import read_records, write_json


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
                       'simulation_preset': manifest['simulation_preset'],
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
              'label_scope': 'contact_with_controller_configuration',
              'split_label_counts': split_counts,
              'oracle_scene_success': float(np.mean([s['oracle_success'] for s in scenes])) if scenes else None,
              'random_scene_success': float(np.mean([s['random_expected_success'] for s in scenes])) if scenes else None,
              'heuristic_scene_success': float(np.mean([s['heuristic_success'] for s in scenes])) if scenes else None,
              'training_gate': 'not_validated',
              'gate_reason': 'Inspect mixed labels, repeatability, and object coverage before training.'}
    if not all(split_counts['train'].values()):
        report.update(training_gate='blocked_single_class_training_data',
                      gate_reason='Training objects do not contain both feasibility classes. Do not move held-out candidates into training to hide this.')
    write_json(directory / 'summary.json', report)
    return report


def initial_com_world(scene_path: Path, scene: dict) -> np.ndarray:
    """Read the actual COM; a mesh bounding-box center need not be its COM."""
    if 'com_world_m' in scene['geometry']:
        return np.asarray(scene['geometry']['com_world_m'])
    import mujoco
    model = mujoco.MjModel.from_binary_path(str(scene_path.parent / 'model.mjb'))
    data = mujoco.MjData(model)
    with np.load(scene_path.parent / 'initial_state.npz') as state:
        mujoco.mj_setState(model, data, state['state'], scene['state_spec'])
    mujoco.mj_forward(model, data)
    return data.xipos[int(model.site_bodyid[model.site('site:obj_frame').id])].copy()


def plot(directory: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    records = read_records(directory / 'rollouts.jsonl')
    for path in sorted(directory.glob('*/scene.json')):
        scene = json.loads(path.read_text())
        rows = [r for r in records if r['candidate_set_id'] == scene['candidate_set_id']]
        com = initial_com_world(path, scene)
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
        hulls = [np.asarray(vertices) for vertices in scene['geometry']['hulls']]
        vertices = np.concatenate(hulls)
        center = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
        extent = np.max(np.abs(vertices[:, :2] - center), axis=0) * 1.25
        extent = np.maximum(extent, 0.01)
        for ax in axes[:3]:
            for vertices in scene['geometry']['hulls']:
                xy = np.unique(np.array(vertices)[:, :2] - center, axis=0)
                outline = xy[ConvexHull(xy).vertices]
                ax.fill(outline[:, 0], outline[:, 1], color='lightgray', alpha=0.4)
            pivot = scene['geometry']['pivot']
            ax.axvline(pivot[0] - center[0], color='black', linestyle=':', label='configured pivot X')
            ax.set(xlabel='X from object center (m)', ylabel='Y from object center (m)',
                   aspect='equal', xlim=(-extent[0], extent[0]), ylim=(-extent[1], extent[1]))
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
        if rows:
            xy = np.array([r['candidate']['position'][:2] for r in rows]) - center
            labels = [r['feasible'] for r in rows]
            axes[0].scatter(*xy.T, c=['tab:green' if f else 'tab:red' for f in labels], s=70)
            for row, (x, y) in zip(rows, xy):
                axes[0].annotate(str(row['candidate']['index']), (x, y), xytext=(4, 4), textcoords='offset points')
            heuristic = int(np.argmin([np.linalg.norm(r['candidate']['press_offset_xy']) for r in rows]))
            axes[0].scatter(*xy[heuristic], marker='*', s=240, facecolors='none', edgecolors='black', label='centre heuristic')
            reference_xy = [r['candidate']['position'][:2] for r in rows if r.get('is_reference_contact', False)]
            if reference_xy:
                axes[0].scatter(*(np.asarray(reference_xy) - center).T, marker='D', s=160, facecolors='none',
                                edgecolors='tab:blue', linewidths=1.5, label='demo reference')
            for ax, metric, title in zip(axes[1:3], ['max_intended_tip_deg', 'arc_contact_fraction'],
                                         ['Intended tip (degrees)', 'ARC time touching object (%)']):
                values = np.array([r['metrics'][metric] for r in rows], dtype=float)
                contact = metric == 'arc_contact_fraction'
                valid = np.array([r['metrics'].get('arc_ticks', 1) > 0 for r in rows]) if contact else np.ones(len(rows), dtype=bool)
                if contact:
                    values *= 100
                limits = {'vmin': 0, 'vmax': 100} if contact else {}
                scatter = ax.scatter(*xy[valid].T, c=values[valid], cmap='viridis', s=70, **limits)
                if not valid.all():
                    ax.scatter(*xy[~valid].T, marker='x', color='gray', label='ARC not reached')
                    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.30))
                bar = fig.colorbar(scatter, ax=ax, shrink=0.8)
                if contact:
                    bar.set_ticks([0, 25, 50, 75, 100])
                    observed = values[valid]
                    if len(observed):
                        title += f"\nObserved: {observed.min():.1f}–{observed.max():.1f}%"
                    ax.text(0.5, 1.20, 'Touching includes sliding; not a grip score',
                            transform=ax.transAxes, ha='center', fontsize=8)
                ax.set_title(title, fontsize=10)
            # Compare Y offsets at fixed X instead of hiding the relationship in a spatial color map.
            off_axis = np.array([r['metrics']['max_off_axis_deg'] if r['metrics'].get('arc_ticks', 1) else np.nan for r in rows])
            y_from_com = xy[:, 1] + center[1] - com[1]
            x_groups = np.round(xy[:, 0], 6)
            for x in np.unique(x_groups):
                indices = np.flatnonzero(x_groups == x)
                indices = indices[np.argsort(xy[indices, 1])]
                if len(indices) > 1:
                    axes[3].plot(y_from_com[indices], off_axis[indices], color='gray', alpha=0.4, linewidth=1)
            scatter = axes[3].scatter(y_from_com, off_axis, c=xy[:, 0], cmap='coolwarm', s=45, zorder=3)
            fig.colorbar(scatter, ax=axes[3], shrink=0.8, label='Contact X offset (m)')
            axes[3].axvline(0, color='black', linestyle=':', linewidth=1)
            axes[3].set(xlabel='Contact Y − initial COM Y (m)', ylabel='Peak ARC-onset-relative rotation (degrees)',
                        title='Measured off-axis rotation (world X/Z)',
                        xlim=(-extent[1] - abs(center[1] - com[1]), extent[1] + abs(center[1] - com[1])), ylim=(0, None))
            axes[3].text(0.5, -0.30, 'Same-X contacts joined; table + robot response',
                         transform=axes[3].transAxes, ha='center', fontsize=8)
            axes[0].set_title('Configured trial: green=pass, red=fail')
            axes[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.30))
        else:
            axes[0].set_title('No compatible candidates')
            axes[1].text(0.02, 0.5, scene['geometry'].get('scene_rejection', 'all proposals rejected').replace('_', '\n'),
                         transform=axes[1].transAxes)
        preset = scene.get('simulation_preset', {}).get('name', 'legacy preset unspecified')
        fig.suptitle(f"{scene['candidate_set_id']} — {preset}\nPlot origin: initial geometry center at world ({center[0]:.3f}, {center[1]:.3f}) m; initial COM Y={com[1]:.3f} m")
        fig.savefig(path.parent / 'contacts.png', dpi=160, bbox_inches='tight')
        plt.close(fig)


def replay(scene_path: Path, candidate_index: int, show_viewer: bool = False,
           video_path: Path | None = None, *, saved_collisions: bool = False):
    """Replay a selected candidate from its saved MJB and full state."""
    import mujoco
    from contact_selection.sim.candidate_generator import Candidate
    from contact_selection.sim.rollout_evaluator import evaluate_rollout
    from contact_selection.sim.controller import config_from_saved

    scene = json.loads(scene_path.read_text())
    config = json.loads((scene_path.parent.parent / 'config.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(scene_path.parent / 'model.mjb'))
    if not saved_collisions:
        from contact_selection.sim.scene import disable_adapter_object_collisions
        disable_adapter_object_collisions(model)
    data = mujoco.MjData(model)
    with np.load(scene_path.parent / 'initial_state.npz') as saved:
        mujoco.mj_setState(model, data, saved['state'], scene['state_spec'])
    mujoco.mj_forward(model, data)
    candidate = Candidate(**next(c for c in scene['candidates'] if c['index'] == candidate_index))
    from contact_selection.sim.video import DemoDisplay

    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Recording {scene['candidate_set_id']}, contact {candidate_index} "
              f"at {candidate.position} → {video_path} (real-time playback)", flush=True)
    controller_config = dict(scene['controller'])
    output = video_path.parent if video_path is not None else scene_path.parent
    with DemoDisplay(model, data, output, video_path is not None, show_viewer,
                     video_path=video_path, playback_speed=1.0) as display:
        result, arrays = evaluate_rollout(
            model, data, candidate, config_from_saved(controller_config), config['feasibility'],
            step_callback=display if video_path is not None or show_viewer else None)
    if video_path is not None:
        if display.frames == 0:
            video_path.unlink(missing_ok=True)
        result.update(video_frames=display.frames, video_speedup=1.0,
                      scene=scene['candidate_set_id'], candidate=candidate.to_dict(),
                      controller=controller_config, source_scene=str(scene_path.resolve()),
                      collision_policy='saved_model' if saved_collisions else 'adapter_object_disabled',
                      label_scope='contact_with_controller_configuration')
        write_json(video_path.with_suffix('.json'), result)
        video_status = str(video_path.resolve()) if display.frames else "No video: approach IK failed before simulation; see " + str(video_path.with_suffix(".json"))
        print(f"Video: {video_status}\nFeasible: {result['feasible']}; "
              f"failures: {', '.join(result['failure_modes']) or 'none'}", flush=True)
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--replay-scene', help='Candidate-set directory name')
    parser.add_argument('--candidate', type=int, default=0)
    parser.add_argument('--show-viewer', action='store_true')
    args = parser.parse_args()
    if args.replay_scene:
        scene_path = args.directory / args.replay_scene / 'scene.json'
        result, _ = replay(scene_path, args.candidate, args.show_viewer,
                           scene_path.parent / f'candidate_{args.candidate:03d}_no_adapter_collision.mp4')
        print(result)
    else:
        plot(args.directory)
        print(json.dumps(summarize(args.directory), indent=2))


if __name__ == '__main__':
    main()
