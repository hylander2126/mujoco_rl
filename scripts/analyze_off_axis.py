#!/usr/bin/env python3
"""Run/cache matched box Y-offset trials and write an explanatory comparison figure."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/contact-selection-mpl')


def main():
    import json
    import hashlib
    from dataclasses import replace
    import mujoco
    import numpy as np
    from contact_selection.candidate_generator import Candidate
    from contact_selection.dataset import write_json
    from contact_selection.off_axis import plot_diagnostic
    from contact_selection.rollout_evaluator import evaluate_rollout
    from parameter_estimation.controllers.press_pull_fsm import PressPullConfig
    from parameter_estimation.scene import disable_adapter_object_collisions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/contact_selection/archive_pre_centering_20260930/box_mu_0p50')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/contact_selection/y_offset_diagnostic')
    args = parser.parse_args()
    scene_path = args.source / 'box_trial_01/scene.json'
    scene = json.loads(scene_path.read_text())
    config = json.loads((args.source / 'config.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'source_sha256': {
        name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
        for name in ('config.json', 'box_trial_01/scene.json', 'box_trial_01/model.mjb',
                     'box_trial_01/initial_state.npz')},
        'collision_policy': 'adapter_object_disabled', 'contact_x_m': 0.58}
    provenance_path = args.output / 'source.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        parser.error('Cached trials use a different source. Choose a fresh --output directory.')
    write_json(provenance_path, provenance)
    write_json(args.output / 'config.json', config)
    records = []
    for center_y, offsets in [(0.08, [-.044, 0, .044]), (0.0, [-.044, -.022, 0, .022, .044])]:
        model = mujoco.MjModel.from_binary_path(str(scene_path.parent / 'model.mjb'))
        disable_adapter_object_collisions(model)
        data = mujoco.MjData(model)
        with np.load(scene_path.parent / 'initial_state.npz') as saved:
            mujoco.mj_setState(model, data, saved['state'], scene['state_spec'])
        pid = int(model.site_bodyid[model.site('site:obj_frame').id])
        adr = int(model.jnt_qposadr[int(model.body_jntadr[pid])])
        old_y = data.qpos[adr + 1]
        model.body_pos[pid, 1] = center_y
        model.qpos_spring[adr + 1] = center_y
        data.qpos[adr + 1] = model.qpos0[adr + 1] = center_y
        mujoco.mj_forward(model, data)
        scene_id = f'box_y_{center_y:.2f}'.replace('.', 'p')
        folder = args.output / scene_id
        folder.mkdir(exist_ok=True)
        candidates = []
        for index, dy in enumerate(offsets):
            c = Candidate(**scene['candidates'][1])
            c = replace(c, index=index, position=[.58, center_y + dy, .35],
                        object_position=[0, dy, .15], press_offset_xy=[0, dy])
            candidates.append(c.to_dict())
            name = f'center_{center_y:.2f}_offset_{dy:+.3f}'
            result_path = args.output / f'{name}.json'
            array_path = args.output / f'{name}.npz'
            if result_path.exists() and array_path.exists():
                result = json.loads(result_path.read_text())
            else:
                result, arrays = evaluate_rollout(model, data, c, PressPullConfig(**scene['controller']), config['feasibility'])
                result.update(center_y=center_y, y_offset=dy, candidate=c.to_dict(),
                              collision_policy='adapter_object_disabled')
                write_json(result_path, result)
                np.savez_compressed(array_path, **arrays)
                print(name, result['feasible'], flush=True)
            records.append({**result, 'candidate': c.to_dict(), 'candidate_set_id': scene_id,
                            'trajectory': array_path.name, 'scene_manifest': f'{scene_id}/scene.json'})
        manifest = json.loads(json.dumps(scene))
        delta = np.array([0, center_y - old_y, 0])
        for key in ('hulls', 'bounds', 'pivot'):
            manifest['geometry'][key] = (np.asarray(manifest['geometry'][key]) + delta).tolist()
        manifest['geometry']['requested'] = len(candidates)
        manifest['geometry']['generated'] = len(candidates)
        manifest['geometry']['com_world_m'] = data.xipos[pid].tolist()
        manifest.update(candidate_set_id=scene_id, candidates=candidates, reference_candidate_indices=[],
                        reference_contact=None, initial_qpos=data.qpos.tolist(), initial_qvel=data.qvel.tolist())
        manifest['physical_parameters'].update(collision_policy='adapter_object_disabled',
                                              geom_contype=model.geom_contype.tolist(),
                                              geom_conaffinity=model.geom_conaffinity.tolist())
        manifest['simulation_preset']['parameters']['object_y_m'] = center_y
        state = np.empty(mujoco.mj_stateSize(model, scene['state_spec']))
        mujoco.mj_getState(model, data, state, scene['state_spec'])
        manifest['state_sha256'] = hashlib.sha256(state.tobytes()).hexdigest()
        np.savez_compressed(folder / 'initial_state.npz', state=state)
        mujoco.mj_saveModel(model, str(folder / 'model.mjb'))
        write_json(folder / 'scene.json', manifest)
    (args.output / 'rollouts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    plot_diagnostic(args.output)
    from contact_selection.visualize import plot
    plot(args.output)
    print(f'Figure: {args.output / "off_axis_comparison.png"}')
    print(f'Watch a centered trial: .venv/bin/python scripts/replay_contact.py {args.output} --scene box_y_0p00 --candidate 4')


if __name__ == '__main__':
    main()
