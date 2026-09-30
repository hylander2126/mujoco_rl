import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from contact_selection.candidate_generator import generate_candidates, upper_surface, collision_hulls
from contact_selection.dataset import append_record, read_records
from contact_selection.features import FEATURE_NAMES, extract_features
from contact_selection.rollout_evaluator import label_feasibility
from contact_selection.controller import PressPullConfig
from contact_selection.scene import load_environment

CONFIG = json.loads((Path(__file__).parents[1] / 'config/box_mu_0p50.json').read_text())


@pytest.fixture(scope='module')
def box_candidates():
    model, data = load_environment(0)
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    candidates, report = generate_candidates(model, data, 5, CONFIG['geometry'], PressPullConfig(verbose=False))
    np.testing.assert_array_equal(data.qpos, before)
    return model, data, candidates, report


def test_candidates_on_surface_and_deterministic(box_candidates):
    model, data, candidates, report = box_candidates
    assert len(candidates) == 5
    lo, hi = np.array(report['bounds'])
    for c in candidates:
        assert np.isclose(c.position[2], hi[2])
        assert np.all(np.array(c.position[:2]) > lo[:2])
        assert np.all(np.array(c.position[:2]) < hi[:2])
        assert c.approach_error_m <= CONFIG['geometry']['approach_tolerance_m']
    again, _ = generate_candidates(model, data, 5, CONFIG['geometry'], PressPullConfig(verbose=False))
    assert [c.to_dict() for c in candidates] == [c.to_dict() for c in again]
    hulls = collision_hulls(model, data, model.body('payload').id)
    assert upper_surface(hulls, hi[:2] + 1, 0.006, 0.95) is None


def test_features(box_candidates):
    _, _, candidates, report = box_candidates
    features = extract_features(candidates[0], report)
    assert list(features) == FEATURE_NAMES
    assert features['normalized_height'] == pytest.approx(1)
    assert features['width_m'] == pytest.approx(0.1)
    assert 'mass' not in features and 'feasible' not in features


def good_metrics():
    return dict(numerically_stable=True, contact_established=True, arc_contact_fraction=1,
                max_intended_tip_deg=3, max_pivot_drift_m=0, max_off_axis_deg=0,
                force_limit_margin_n=1, min_joint_margin_rad=0.1, unintended_collision=False,
                completed=True, done=True)


@pytest.mark.parametrize('key,value,reason', [
    ('max_intended_tip_deg', 0, 'insufficient_intended_rotation'),
    ('max_intended_tip_deg', -3, 'insufficient_intended_rotation'),
    ('arc_contact_fraction', 0.5, 'contact_loss'),
    ('max_pivot_drift_m', 0.02, 'unintended_pivot_or_sliding'),
    ('max_off_axis_deg', 10, 'unintended_rotation'),
    ('force_limit_margin_n', -1, 'force_limit_exceeded'),
    ('min_joint_margin_rad', -0.1, 'joint_limit'),
    ('unintended_collision', True, 'unintended_collision'),
    ('numerically_stable', False, 'simulation_instability'),
])
def test_labeling(key, value, reason):
    m = good_metrics()
    assert label_feasibility(m, CONFIG['feasibility'], 2) == (True, [])
    m[key] = value
    ok, reasons = label_feasibility(m, CONFIG['feasibility'], 2)
    assert not ok and reason in reasons


def test_abort_overrides_completion():
    ok, reasons = label_feasibility(good_metrics(), CONFIG['feasibility'], 2, 'lost contact during ARC')
    assert not ok and 'contact_loss' in reasons


def test_nan_cannot_be_feasible():
    metrics = good_metrics()
    metrics['max_intended_tip_deg'] = float('nan')
    assert label_feasibility(metrics, CONFIG['feasibility'], 2) == (False, ['simulation_instability'])


def test_full_state_snapshot_round_trip(box_candidates, tmp_path):
    model, data, _, _ = box_candidates
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    path = tmp_path / 'scene.mjb'
    mujoco.mj_saveModel(model, str(path))
    restored_model = mujoco.MjModel.from_binary_path(str(path))
    restored = mujoco.MjData(restored_model)
    mujoco.mj_setState(restored_model, restored, state, spec)
    actual = np.empty_like(state)
    mujoco.mj_getState(restored_model, restored, actual, spec)
    np.testing.assert_array_equal(actual, state)


def test_json_serialization(tmp_path):
    path = tmp_path / 'rollouts.jsonl'
    append_record(path, {'candidate_set_id': 'same', 'index': np.int64(1), 'missing': np.nan})
    append_record(path, {'candidate_set_id': 'same', 'feasible': np.bool_(False)})
    assert read_records(path) == [{'candidate_set_id': 'same', 'index': 1, 'missing': None},
                                  {'candidate_set_id': 'same', 'feasible': False}]


def test_invalid_mesh_pivot_is_scene_rejection():
    model, data = load_environment(10)
    mujoco.mj_forward(model, data)
    candidates, report = generate_candidates(model, data, 5, CONFIG['geometry'], PressPullConfig(verbose=False))
    assert not candidates
    assert report['scene_rejection'] == 'pivot_site_incompatible_with_near_x_support_edge'


def test_explicit_pivot_override_uses_existing_controller_option():
    model, data = load_environment(10)
    mujoco.mj_forward(model, data)
    original_site = model.site_pos.copy()
    cfg = PressPullConfig(verbose=False, arc_center_xz=(0.5699898114345493, 0.05))
    candidates, report = generate_candidates(model, data, 3, CONFIG['geometry'], cfg)
    assert candidates
    assert report['pivot'][0] == cfg.arc_center_xz[0]
    np.testing.assert_array_equal(model.site_pos, original_site)


def test_summary_preserves_empty_scenes_and_training_gate(tmp_path):
    from contact_selection.dataset import write_json
    from contact_selection.visualize import summarize
    for name, candidates in [('positive', [{}]), ('empty', [])]:
        (tmp_path / name).mkdir()
        write_json(tmp_path / name / 'scene.json', {
            'candidate_set_id': name, 'object_name': name, 'candidates': candidates,
            'geometry': {}, 'simulation_preset': {'name': 'box_grip'}})
    append_record(tmp_path / 'rollouts.jsonl', {
        'candidate_set_id': 'positive', 'object_split': 'validation', 'feasible': True,
        'candidate': {'press_offset_xy': [0, 0]}, 'failure_modes': []})
    report = summarize(tmp_path)
    assert report['oracle_scene_success'] == 0.5
    assert report['random_scene_success'] == 0.5
    assert report['training_gate'] == 'blocked_single_class_training_data'
    assert report['split_label_counts']['train']['feasible'] == 0
