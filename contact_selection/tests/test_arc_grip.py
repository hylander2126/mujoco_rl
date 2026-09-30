"""Check the explicitly exploratory mesh contact setup."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from contact_selection.generate import generate, prepare_experiment_scene, set_initial_object_position
from contact_selection.physics import ArcGripConfig


@pytest.mark.parametrize('object_id', [10, 11])
def test_arc_grip_applies_both_sides_of_table_contact(object_id):
    model, data, controller, reference, preset = prepare_experiment_scene(
        object_id, {'name': 'arc_grip', 'parameters': {'ground_friction': 0.15,
                                                     'object_friction': 0.15}})
    payload = model.site_bodyid[model.site('site:obj_frame').id]
    payload_geoms = np.flatnonzero(model.geom_bodyid == payload)
    assert len(payload_geoms) > 0
    np.testing.assert_allclose(model.geom_friction[payload_geoms, 0], 0.15)
    assert model.geom_friction[model.geom('table').id, 0] == 0.15
    assert model.geom_friction[model.geom('push_ball_col').id, 0] == 2.0
    assert model.geom_priority[model.geom('push_ball_col').id] == 1
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.noslip_iterations == 10
    assert controller.rotate_with_arc and controller.arc_force_drop_fraction == 0.1
    assert controller.force_ref_n == 5 and controller.max_normal_speed == 0.005
    assert reference is None and preset['name'] == 'arc_grip'
    assert preset['parameters']['object_friction'] == 0.15


def test_arc_grip_rejects_invalid_physics():
    with pytest.raises(ValueError, match='Friction'):
        ArcGripConfig(object_friction=-0.1)
    with pytest.raises(ValueError, match='Press force'):
        ArcGripConfig(press_force_n=15)


def test_initial_object_position_updates_free_joint():
    model, data, _, _, _ = prepare_experiment_scene(13, {'name': 'arc_grip'})
    before = data.site_xpos[model.site('site:obj_frame').id].copy()
    set_initial_object_position(model, data, [0.58562788, 0, 0.05])
    after = data.site_xpos[model.site('site:obj_frame').id].copy()
    np.testing.assert_allclose(after - before, [-0.41437212, 0, 0], atol=1e-7)
    with pytest.raises(ValueError, match='three finite'):
        set_initial_object_position(model, data, [0, float('nan'), 0])


def test_saved_pose_survives_setconst_and_generates_contacts(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).parents[1] / 'config/monitor_soda_mu_0p50.json').read_text())
    config['objects'] = [13]
    config['candidates'] = 12  # Soda needs the denser proposal grid to find its two valid points.

    def evaluator(model, data, candidate, controller, thresholds):
        return {'feasible': False, 'failure_modes': ['test_stub'],
                'metrics': {'max_intended_tip_deg': 0}}, {'t_hist': np.array([0.0])}

    monkeypatch.setattr('contact_selection.generate.evaluate_rollout', evaluator)
    output = tmp_path / 'run'
    generate(config, output)
    scene = json.loads(next(output.glob('*/scene.json')).read_text())
    assert len(scene['candidates']) == 2
    assert scene['requested_initial_object_position'] == config['initial_object_position_by_object']['soda']
    assert scene['geometry']['com_world_m'][1] == pytest.approx(0, abs=1e-12)
    model = mujoco.MjModel.from_binary_path(str(output / scene['candidate_set_id'] / 'model.mjb'))
    payload = model.site_bodyid[model.site('site:obj_frame').id]
    adr = model.jnt_qposadr[model.body_jntadr[payload]]
    np.testing.assert_allclose(scene['initial_qpos'][adr:adr + 3], scene['initial_object_position'])
