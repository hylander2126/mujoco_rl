"""Reproducible scene-grouped experiments, deliberately preceding ML training."""
import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import platform
import subprocess

import mujoco
import numpy as np

from util.paths import CONTACT_SELECTION_OUTPUTS, dated
from contact_selection.candidate_generator import generate_candidates
from contact_selection.dataset import append_record, content_id, write_json
from contact_selection.features import extract_features
from contact_selection.rollout_evaluator import evaluate_rollout
from contact_selection.scene import OBJECTS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('config') / 'box_mu_0p50.json'


def prepare_experiment_scene(object_id: int, preset: dict):
    """Load validated box or exploratory mesh contact physics.

    Returns model, reset data, base controller config, and an optional reference
    contact. Candidate generation still validates that reference geometrically.
    """
    if preset.get('name') == 'arc_grip':
        from contact_selection.physics import prepare_arc_grip
        return prepare_arc_grip(object_id, preset.get('parameters', {}))
    if preset.get('name') != 'box_grip':
        raise ValueError(f"Unknown simulation preset: {preset.get('name')}")
    if object_id != 0:
        raise ValueError('box_grip supports only object 0 (box)')
    from contact_selection.box import BoxDemoConfig, prepare_box
    options = BoxDemoConfig(**preset.get('parameters', {}))
    model, data, reference, controller, metadata = prepare_box(options, verbose=False)
    return model, data, controller, reference, {'name': 'box_grip', 'parameters': metadata['preset']}


def set_initial_object_position(model, data, xyz) -> None:
    """Set a free payload's world translation before saving the reset state."""
    position = np.asarray(xyz, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Initial object position must be three finite coordinates')
    payload = int(model.site_bodyid[model.site('site:obj_frame').id])
    if model.body_jntnum[payload] != 1:
        raise ValueError('Object position override requires one payload free joint')
    joint = int(model.body_jntadr[payload])
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Object position override requires a payload free joint')
    adr = int(model.jnt_qposadr[joint])
    data.qpos[adr:adr + 3] = position
    mujoco.mj_forward(model, data)


def _evaluate_saved_candidate(job):
    """Worker loads immutable snapshots; scene XML construction stays in the parent."""
    import json
    from contact_selection.candidate_generator import Candidate
    from contact_selection.controller import PressPullConfig
    scene_dir, candidate_dict, thresholds = job
    scene_dir = Path(scene_dir)
    manifest = json.loads((scene_dir / 'scene.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(scene_dir / 'model.mjb'))
    data = mujoco.MjData(model)
    with np.load(scene_dir / 'initial_state.npz') as saved:
        mujoco.mj_setState(model, data, saved['state'], manifest['state_spec'])
    mujoco.mj_forward(model, data)
    candidate = Candidate(**candidate_dict)
    outcome, arrays = evaluate_rollout(model, data, candidate, PressPullConfig(**manifest['controller']), thresholds)
    np.savez_compressed(scene_dir / f'candidate_{candidate.index:03d}.npz', **arrays)
    return outcome


def _candidate_outcomes(model, data, candidates, cfg, thresholds, scene_dir, workers):
    if workers == 1:
        for candidate in candidates:
            outcome, arrays = evaluate_rollout(model, data, candidate, cfg, thresholds)
            yield candidate, outcome, arrays
    elif candidates:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        jobs = [(str(scene_dir), c.to_dict(), thresholds) for c in candidates]
        with ProcessPoolExecutor(max_workers=min(workers, len(candidates)),
                                 mp_context=get_context('spawn')) as pool:
            for candidate, outcome in zip(candidates, pool.map(_evaluate_saved_candidate, jobs)):
                yield candidate, outcome, None


def generate(config: dict, output: Path, workers: int = 1) -> None:
    """Fresh output directory only; never mix reruns or silently overwrite trials."""
    if workers < 1:
        raise ValueError('workers must be positive')
    preset = config["simulation_preset"]
    if preset.get("name") == "box_grip" and any(i != 0 for i in config["objects"]):
        raise ValueError("box_grip supports only object 0 (box)")
    output.mkdir(parents=True, exist_ok=False)
    config_id = content_id(config)
    write_json(output / 'config.json', config)
    source_paths = sorted(set(ROOT.glob('contact_selection/*.py')) |
                          set(ROOT.glob('parameter_estimation/controllers/*.py')) |
                          set(ROOT.glob('parameter_estimation/*.py')) |
                          set(ROOT.glob('mujoco_irb120/robot/controllers/*.py')))
    write_json(output / 'provenance.json', {
        'python': platform.python_version(), 'mujoco': mujoco.__version__, 'numpy': np.__version__,
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}})
    records_path = output / 'rollouts.jsonl'
    records_path.touch()
    for object_id in config['objects']:
        for repeat in range(config['repeats']):
            seed = int(np.random.SeedSequence([config['seed'], object_id, repeat]).generate_state(1)[0])
            rng = np.random.default_rng(seed)
            name = OBJECTS[object_id]
            scene_id = f'{name}_trial_{repeat + 1:02d}'
            scene_dir = output / scene_id
            scene_dir.mkdir()
            model, data, base_cfg, reference, resolved_preset = prepare_experiment_scene(object_id, preset)
            initial_object_position = config.get('initial_object_position_by_object', {}).get(name)
            if initial_object_position is not None and reference is not None:
                raise ValueError('Initial object position cannot move a fixed reference contact')
            mujoco.mj_forward(model, data)
            pid = int(model.site_bodyid[model.site('site:obj_frame').id])
            randomization = {key: float(rng.uniform(*bounds)) for key, bounds in config['randomization'].items()}
            if set(randomization) != {'mass_scale', 'force_scale'}:
                raise ValueError('Supported randomization: mass_scale and force_scale')
            if min(randomization.values()) <= 0:
                raise ValueError('Mass and force scales must be positive')
            model.body_mass[pid] *= randomization['mass_scale']
            model.body_inertia[pid] *= randomization['mass_scale']
            mujoco.mj_setConst(model, data)
            # mj_setConst resets qpos; apply the experimental pose afterwards.
            if initial_object_position is not None:
                set_initial_object_position(model, data, initial_object_position)
            mujoco.mj_forward(model, data)
            requested_initial_position = initial_object_position
            if config.get('center_com_y', False):
                position = data.xpos[pid].copy()
                position[1] -= data.xipos[pid, 1]
                set_initial_object_position(model, data, position)
                initial_object_position = position.tolist()
            controller_options = {**config['controller'], **config.get('controller_by_object', {}).get(name, {})}
            cfg = replace(base_cfg, **controller_options)
            cfg.force_ref_n *= randomization['force_scale']
            if cfg.adaptive_retry:
                raise ValueError('Candidate comparison runs single attempts; disable adaptive_retry')
            if cfg.force_ref_n >= cfg.force_hard_limit_n:
                raise ValueError('Randomized force reference exceeds controller safety limit')
            candidates, geometry = generate_candidates(
                model, data, config['candidates'], config['geometry'], cfg,
                reference_points_xy=[reference.position[:2]] if reference is not None else None)
            geometry['com_world_m'] = data.xipos[pid].tolist()
            reference_indices = [c.index for c in candidates if reference is not None
                                 and np.allclose(c.position, reference.position, rtol=0, atol=1e-7)]
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(model, state_spec))
            mujoco.mj_getState(model, data, state, state_spec)
            np.savez_compressed(scene_dir / 'initial_state.npz', state=state)
            mujoco.mj_saveModel(model, str(scene_dir / 'model.mjb'))
            physical = {'mass_kg': float(model.body_mass[pid]), 'com_body_m': model.body_ipos[pid],
                        'inertia': model.body_inertia[pid], 'geom_friction': model.geom_friction,
                        'geom_solref': model.geom_solref, 'geom_solimp': model.geom_solimp,
                        'geom_condim': model.geom_condim, 'geom_priority': model.geom_priority,
                        'geom_contype': model.geom_contype, 'geom_conaffinity': model.geom_conaffinity,
                        'collision_policy': 'adapter_object_disabled',
                        'timestep': model.opt.timestep, 'cone': int(model.opt.cone),
                        'solver': int(model.opt.solver), 'impratio': model.opt.impratio,
                        'noslip_iterations': model.opt.noslip_iterations,
                        'iterations': model.opt.iterations, 'tolerance': model.opt.tolerance,
                        'noslip_tolerance': model.opt.noslip_tolerance}
            manifest = {'schema_version': 2, 'object_id': object_id, 'object_name': name,
                        'object_split': config['splits'][name], 'random_seed': seed,
                        'candidate_set_id': scene_id, 'config_id': config_id,
                        'simulation_preset': resolved_preset,
                        'initial_object_position': initial_object_position,
                        'requested_initial_object_position': requested_initial_position,
                        'center_com_y': config.get('center_com_y', False),
                        'reference_candidate_indices': reference_indices,
                        'reference_contact': reference.to_dict() if reference is not None else None,
                        'randomization': randomization, 'physical_parameters': physical,
                        'initial_qpos': data.qpos, 'initial_qvel': data.qvel,
                        'state_spec': int(state_spec), 'state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
                        'controller': asdict(cfg), 'geometry': geometry,
                        'candidates': [c.to_dict() for c in candidates]}
            write_json(scene_dir / 'scene.json', manifest)
            print(f'{scene_id}: {len(candidates)} candidates; {geometry.get("scene_rejection", "geometry accepted")}', flush=True)
            for candidate, outcome, arrays in _candidate_outcomes(
                    model, data, candidates, cfg, config['feasibility'], scene_dir, workers):
                trajectory = f'{scene_id}/candidate_{candidate.index:03d}.npz'
                if arrays is not None:
                    np.savez_compressed(output / trajectory, **arrays)
                record = {k: manifest[k] for k in ('object_id', 'object_name', 'object_split', 'random_seed',
                          'candidate_set_id', 'config_id', 'simulation_preset', 'initial_object_position',
                          'randomization', 'physical_parameters', 'state_sha256')}
                record.update(candidate=candidate.to_dict(), is_reference_contact=candidate.index in reference_indices,
                              features=extract_features(candidate, geometry),
                              scene_manifest=f'{scene_id}/scene.json', trajectory=trajectory, **outcome)
                append_record(records_path, record)
                print(f'  {candidate.index}: feasible={outcome["feasible"]}, '
                      f'tip={outcome["metrics"]["max_intended_tip_deg"]:.2f} deg, '
                      f'failures={outcome["failure_modes"]}', flush=True)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='Experiment configuration (default: box_mu_0p50.json)')
    parser.add_argument('--output', type=Path,
                        help='Dataset directory (default: sweeps/YYYY-MM-DD_CONFIG)')
    parser.add_argument('--objects', type=int, nargs='+')
    parser.add_argument('--candidates', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--workers', type=int, default=1, help='Independent rollout processes; scene preparation remains serial')
    args = parser.parse_args()
    args.output = args.output or CONTACT_SELECTION_OUTPUTS / 'sweeps' / dated(args.config.stem)
    config = json.loads(args.config.read_text())
    for key in ('objects', 'candidates', 'repeats'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    print('Generating a full numerical dataset (no videos). To watch a saved contact, use\n'
          '  .venv/bin/python -m contact_selection replay RUN --candidate INDEX', flush=True)
    generate(config, args.output, workers=args.workers)
    print(f'Data saved. Record one contact with:\n'
          f'  .venv/bin/python -m contact_selection replay {args.output} --candidate 0', flush=True)


if __name__ == '__main__':
    main()
