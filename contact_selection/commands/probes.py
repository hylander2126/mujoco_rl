#!/usr/bin/env python3
"""Rerun the reported sensitivity probes using fresh, COM-centered suite snapshots."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/contact-selection-mpl')


def main():
    import hashlib
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import mujoco
    import numpy as np
    from contact_selection.sim.dataset import write_json, append_record, content_id, read_records
    from contact_selection.commands.generate import _evaluate_saved_candidate
    from contact_selection.commands.visualize import plot, summarize
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    specs = []
    for mu, mass in [(.14, 1), (.16, 1), (.15, .9), (.15, 1.1)]:
        specs.append((f'box_mu_{mu:g}_mass_{mass:g}', 'box_mu_0p15', 'box_trial_01', [0,1,2,4],
                      {'ground_friction': mu, 'mass_scale': mass}, 'boundary_probe'))
    for obj, mus in [('heart', [.45,.55]), ('L', [.23,.27])]:
        for mu in mus:
            specs.append((f'{obj}_mu_{mu:g}', 'heart_l_mu_0p50', f'{obj}_trial_01', [0,1],
                          {'ground_friction':mu, 'object_friction':mu}, 'mesh_robustness_probe'))
    for mu, finger in [(.1,2),(.17,2),(.5,.2)]:
        specs.append((f'box_mu_{mu:g}_finger_{finger:g}', 'box_mu_0p50', 'box_trial_01', [0,1,2],
                      {'ground_friction':mu, 'finger_friction':finger}, 'small_friction_probe'))
    jobs, entries = [], []
    for name, source_name, scene_id, indices, overrides, group in specs:
        source = args.suite / source_name
        source_scene = source / scene_id
        manifest = json.loads((source_scene/'scene.json').read_text())
        config = json.loads((source/'config.json').read_text())
        folder = args.suite / 'probes' / name
        folder.mkdir(parents=True, exist_ok=False)
        scene_folder = folder/scene_id
        scene_folder.mkdir()
        model=mujoco.MjModel.from_binary_path(str(source_scene/'model.mjb'))
        data=mujoco.MjData(model)
        with np.load(source_scene/'initial_state.npz') as a: state=a['state'].copy()
        pid=int(model.site_bodyid[model.site('site:obj_frame').id])
        mass=overrides.get('mass_scale',1.0)
        model.body_mass[pid]*=mass; model.body_inertia[pid]*=mass
        model.geom_friction[model.geom('table').id,0]=overrides['ground_friction']
        if 'object_friction' in overrides:
            model.geom_friction[model.geom_bodyid==pid,0]=overrides['object_friction']
        if 'finger_friction' in overrides:
            model.geom_friction[model.geom('push_ball_col').id,0]=overrides['finger_friction']
        mujoco.mj_setConst(model,data)
        mujoco.mj_setState(model,data,state,manifest['state_spec']);mujoco.mj_forward(model,data)
        config['simulation_preset']['parameters'].update({k:v for k,v in overrides.items() if k!='mass_scale'})
        config['randomization']['mass_scale']=[mass,mass]
        config['probe_candidate_indices']=indices
        write_json(folder/'config.json',config)
        manifest['config_id']=content_id(config)
        manifest['simulation_preset']['parameters'].update({k:v for k,v in overrides.items() if k!='mass_scale'})
        manifest['randomization']['mass_scale']=mass
        manifest['physical_parameters'].update(mass_kg=float(model.body_mass[pid]),inertia=model.body_inertia[pid].tolist(),
                                               geom_friction=model.geom_friction.tolist())
        manifest['candidates']=[c for c in manifest['candidates'] if c['index'] in indices]
        manifest['geometry']['requested']=manifest['geometry']['generated']=len(indices)
        manifest['source_dataset']=source_name
        manifest['probe_overrides']=overrides
        write_json(scene_folder/'scene.json',manifest)
        np.savez_compressed(scene_folder/'initial_state.npz',state=state)
        mujoco.mj_saveModel(model,str(scene_folder/'model.mjb'))
        base_rows={r['candidate']['index']:r for r in read_records(source/'rollouts.jsonl') if r['candidate_set_id']==scene_id}
        for c in manifest['candidates']:
            jobs.append((str(scene_folder),c,config['feasibility']))
            entries.append((folder,manifest,c,base_rows[c['index']],group,overrides))
    reports={group:[] for group in ['boundary_probe','mesh_robustness_probe','small_friction_probe']}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for outcome,entry in zip(pool.map(_evaluate_saved_candidate,jobs),entries):
            folder,manifest,c,old,group,overrides=entry
            row={**old,**outcome,'config_id':manifest['config_id'],
                 'simulation_preset':manifest['simulation_preset'],'physical_parameters':manifest['physical_parameters'],
                 'randomization':manifest['randomization']}
            append_record(folder/'rollouts.jsonl',row)
            reports[group].append({'object':manifest['object_name'],**overrides,'candidate_index':c['index'],
                                   'position':c['position'],'dataset':str(folder.relative_to(args.suite)),**outcome})
            print(folder.name,c['index'],outcome['feasible'],outcome['failure_modes'],flush=True)
    for group,rows in reports.items():write_json(args.suite/f'{group}.json',rows)
    for folder in sorted((args.suite/'probes').iterdir()):plot(folder);summarize(folder)
    print('Completed',len(jobs),'sensitivity rollouts.',flush=True)


if __name__=='__main__':main()
