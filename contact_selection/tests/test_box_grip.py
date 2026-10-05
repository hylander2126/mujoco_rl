"""Verify that contact tests use the working demo settings, not a second copy."""
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from contact_selection.candidate_generator import generate_candidates
from contact_selection.dataset import read_records
from contact_selection.generate import generate, prepare_experiment_scene
from contact_selection.box import BoxDemoConfig, prepare_box

CONFIG = json.loads((Path(__file__).parents[1] / 'config/box_mu_0p50.json').read_text())


def test_preset_matches_demo_and_contains_reference():
    model, data, cfg, reference, preset = prepare_experiment_scene(0, CONFIG['simulation_preset'])
    demo_model, demo_data, demo_ref, demo_cfg, _ = prepare_box(BoxDemoConfig(), verbose=False)
    assert asdict(cfg) == asdict(demo_cfg)
    assert reference.to_dict() == demo_ref.to_dict()
    for key in ('geom_friction', 'geom_priority', 'geom_solref', 'geom_solimp', 'geom_condim'):
        np.testing.assert_array_equal(getattr(model, key), getattr(demo_model, key))
    np.testing.assert_array_equal(data.qpos, demo_data.qpos)
    assert model.opt.cone == demo_model.opt.cone
    assert model.opt.impratio == demo_model.opt.impratio
    assert model.opt.noslip_iterations == demo_model.opt.noslip_iterations
    assert preset['parameters'] == asdict(BoxDemoConfig())
    candidates, _ = generate_candidates(model, data, 2, CONFIG['geometry'], cfg,
                                        reference_points_xy=[reference.position[:2]])
    assert len(candidates) == 2
    np.testing.assert_allclose(candidates[0].position, demo_ref.position, atol=1e-12)
    assert candidates[1].press_offset_xy == pytest.approx([0, 0])


def test_reference_still_must_pass_geometry_filters():
    model, data, cfg, _, _ = prepare_experiment_scene(0, CONFIG['simulation_preset'])
    candidates, geometry = generate_candidates(model, data, 2, CONFIG['geometry'], cfg,
                                               reference_points_xy=[[5, 5]])
    assert len(candidates) == 2
    assert geometry['rejected']['surface_or_edge_clearance'] >= 1
    assert all(c.position[0] < 1 for c in candidates)


def test_preset_is_box_only_and_unknown_preset_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='only object 0'):
        generate({**CONFIG, 'objects': [10]}, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    with pytest.raises(ValueError):
        prepare_experiment_scene(0, {'name': 'typo'})


def test_dataset_persists_resolved_preset_and_shared_reset(tmp_path, monkeypatch):
    snapshots = []
    def evaluator(model, data, candidate, cfg, thresholds):
        assert model.opt.noslip_iterations == 10
        assert not cfg.rotate_with_arc and cfg.arc_force_drop_fraction == .1
        assert cfg.max_normal_speed == .005 and cfg.force_ref_n == 5
        snapshots.append(data.qpos.copy())
        # Stub only expensive physics: verify real assembly/serialization wiring.
        return {'feasible': False, 'failure_modes': ['test_stub'],
                'metrics': {'max_intended_tip_deg': 0}}, {'t_hist': np.array([0.0])}
    monkeypatch.setattr('contact_selection.generate.evaluate_rollout', evaluator)
    output = tmp_path / 'run'
    generate({**CONFIG, 'candidates': 2}, output)
    rows = read_records(output / 'rollouts.jsonl')
    assert len(rows) == 2
    assert rows[0]['candidate_set_id'] == 'box_trial_01'
    assert rows[0]['random_seed'] == 481830384
    assert rows[0]['is_reference_contact'] and not rows[1]['is_reference_contact']
    assert rows[0]['state_sha256'] == rows[1]['state_sha256']
    np.testing.assert_array_equal(*snapshots)
    assert rows[0]['physical_parameters']['noslip_iterations'] == 10
    assert rows[0]['simulation_preset']['parameters'] == asdict(BoxDemoConfig())
    manifest = json.loads((output / rows[0]['scene_manifest']).read_text())
    assert manifest['reference_candidate_indices'] == [0]
    loaded = mujoco.MjModel.from_binary_path(str(output / rows[0]['candidate_set_id'] / 'model.mjb'))
    assert loaded.opt.impratio == 10
    assert loaded.geom_friction[loaded.geom('table').id, 0] == .5


@pytest.mark.parametrize('center_y', [0.0, 0.08])
def test_centered_box_survives_reset_and_preserves_local_contact(center_y):
    model, data, candidate, _, _ = prepare_box(BoxDemoConfig(object_y_m=center_y), verbose=False)
    payload = int(model.site_bodyid[model.site('site:obj_frame').id])
    assert data.xipos[payload, 1] == pytest.approx(center_y)
    assert candidate.position[1] == pytest.approx(center_y)
    assert candidate.object_position[1] == pytest.approx(0.0)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    assert data.xipos[payload, 1] == pytest.approx(center_y)
