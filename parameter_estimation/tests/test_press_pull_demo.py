from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from parameter_estimation.controllers.press_pull_fsm import PressPullConfig, PressPullFSM
from parameter_estimation.press_pull_demo import BoxDemoConfig, prepare_box
from parameter_estimation.scene import load_environment


def test_default_controller_keeps_legacy_behavior():
    cfg = PressPullConfig()
    assert not cfg.rotate_with_arc
    assert cfg.arc_force_drop_fraction is None


def test_box_preset_does_not_modify_default_scene():
    model, data, candidate, cfg, metadata = prepare_box(BoxDemoConfig(), verbose=False)
    pivot = metadata['pivot_world_m']
    assert candidate.position[0] - pivot[0] == pytest.approx(0.006)
    assert candidate.position[1:] == pytest.approx([0.0, 0.35])
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.noslip_iterations == 10
    assert model.geom_priority[model.geom('push_ball_col').id] == 1
    assert metadata['mass_kg'] == pytest.approx(0.663)
    assert not cfg.rotate_with_arc and cfg.arc_force_drop_fraction == 0.1
    untouched, _ = load_environment(0)
    assert untouched.opt.noslip_iterations == 0
    assert untouched.geom_friction[untouched.geom('table').id, 0] == 0
    assert untouched.geom_priority[untouched.geom('push_ball_col').id] == 0


@pytest.mark.parametrize('kwargs', [
    {'press_force_n': 0}, {'press_force_n': 15}, {'ground_friction': -1},
    {'force_drop_fraction': 1}, {'finger_friction': float('nan')}, {'timestep': 0},
])
def test_invalid_demo_parameters(kwargs):
    with pytest.raises(ValueError):
        BoxDemoConfig(**kwargs)


@pytest.mark.parametrize('inset', [-0.01, 0, 0.06])
def test_contact_must_be_in_near_half_of_top_face(inset):
    with pytest.raises(ValueError):
        prepare_box(BoxDemoConfig(edge_inset_m=inset), verbose=False)


@pytest.mark.parametrize('rotating', [False, True])
def test_wrist_twist_preserves_requested_ball_velocity(rotating):
    calls = []
    offset = np.array([0.18, 0.01, 0.02])

    def pose(which):
        out = np.eye(4)
        if which == 'ball':
            out[:3, 3] = offset
        return out

    fsm = PressPullFSM.__new__(PressPullFSM)
    fsm.cfg = PressPullConfig(rotate_with_arc=rotating)
    fsm._hold_rotation = None
    fsm.irb = SimpleNamespace(get_site_pose=pose,
                             apply_cartesian_keyboard_ctrl=lambda v, **kw: calls.append((v, kw)))
    wy = -0.04 if rotating else 0
    desired = np.array([-0.008, 0, -0.001])
    fsm._command(*desired, wy=wy)
    twist, options = calls[0]
    np.testing.assert_allclose(twist[3:] + np.cross(twist[:3], offset), desired, atol=1e-12)
    assert not options['maintain_orientation']


def test_peak_drop_does_not_exit_during_ramp_and_exits_after_sweep():
    # Exercise the real FSM's exit branch with controlled sensor readings.
    from mujoco_irb120.robot.controllers.robot import controller
    model, data = load_environment(0)
    mujoco.mj_forward(model, data)
    cfg = PressPullConfig(arc_force_drop_fraction=0.1, verbose=False)
    fsm = PressPullFSM(controller(model, data), model, data, cfg)
    fsm.state = 'ARC'
    fsm._arc_start_angle = 0.0
    fsm._arc_center_x = fsm._arc_center_z = 0.0
    fsm._arc_end_angle = -1.0
    fsm._arc_peak_force = 4.0
    fsm._n_fx_low_stable = 1
    fsm._record = lambda *args: None
    fsm._arc_step = lambda *args: None
    fsm._arc_fx_flipped = lambda *args: False
    fsm._check_lost_contact = lambda *args: False
    fsm.object_tip_angle_deg = lambda: 10
    fsm._ball_xz = lambda: (0.0, 1.0)
    force = np.array([0.2, 0, 0, 0, 0, 0])
    fsm._world_wrench = lambda: (force, force)
    fsm._current_arc_angle = lambda *args: -0.01
    fsm.step()
    assert fsm.state == 'ARC'
    fsm._current_arc_angle = lambda *args: -0.15
    fsm.step()
    assert fsm.state == 'LULL'
    assert '10% of peak' in fsm.arc_exit_reason


@pytest.mark.parametrize('pitch', [-15, 0, 5, 10, 15])
def test_finger_pitch_preserves_ball_target(pitch):
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    from mujoco_irb120.robot.controllers.robot import controller
    model, data, _, cfg, _ = prepare_box(BoxDemoConfig(), verbose=False)
    irb = controller(model, data)
    home_rotation = irb.FK()[:3, :3].copy()
    fsm = PressPullFSM(irb, model, data, replace(cfg, finger_pitch_deg=pitch,
                                               press_offset_xy=(0.044, -0.044)))
    top = fsm.object_top_center()
    target = top + np.array([0.044, -0.044, cfg.approach_clearance_m])
    fsm.move_to_pre_squash()
    np.testing.assert_allclose(irb.get_site_pose('ball')[:3, 3], target, atol=0.002)
    actual_rotation = irb.FK()[:3, :3]
    desired_rotation = Rotation.from_euler('y', pitch, degrees=True).as_matrix() @ home_rotation
    error = Rotation.from_matrix(actual_rotation @ desired_rotation.T).magnitude()
    assert error < np.deg2rad(0.1)


def test_unreachable_pitch_is_reported_as_controller_failure():
    from dataclasses import replace
    from contact_selection.rollout_evaluator import evaluate_rollout
    model, data, candidate, cfg, _ = prepare_box(BoxDemoConfig(object_y_m=0.08), verbose=False)
    candidate = replace(candidate, position=[0.624, 0.036, 0.35], press_offset_xy=[0.044, -0.044])
    thresholds = {'min_arc_contact_fraction': 0.9, 'max_pivot_drift_m': 0.01,
                  'max_off_axis_deg': 3, 'joint_limit_tolerance_rad': 0.01}
    result, _ = evaluate_rollout(model, data, candidate, replace(cfg, finger_pitch_deg=30), thresholds)
    assert not result['feasible']
    assert 'unreachable' in result['failure_modes']
    assert result['finger_pitch_deg'] == 30
    assert result['label_scope'] == 'contact_with_controller_configuration'
    assert not result['collision_events']
    assert result['metrics']['arc_ticks'] == 0


@pytest.mark.parametrize('y', [0.0, 0.08])
def test_box_y_center_survives_constant_recomputation(y):
    model, data, candidate, _, _ = prepare_box(BoxDemoConfig(object_y_m=y), verbose=False)
    pid = int(model.site_bodyid[model.site('site:obj_frame').id])
    assert candidate.position[1] == pytest.approx(y)
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    assert data.xpos[pid, 1] == pytest.approx(y)
    assert data.xipos[pid, 1] == pytest.approx(y)


def test_static_orientation_corrects_drift_without_changing_ball_velocity():
    from scipy.spatial.transform import Rotation
    calls = []
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('y', 2, degrees=True).as_matrix()
    offset = np.array([0.18, 0, 0.01])
    def site(which):
        out = pose.copy()
        if which == 'ball':
            out[:3, 3] += offset
        return out
    fsm = PressPullFSM.__new__(PressPullFSM)
    fsm.cfg = PressPullConfig()
    fsm._hold_rotation = np.eye(3)
    fsm.irb = SimpleNamespace(get_site_pose=site,
        apply_cartesian_keyboard_ctrl=lambda v, **kw: calls.append(v))
    desired = np.array([-0.008, 0, -0.001])
    fsm._command(*desired, wy=0.5)  # Static mode ignores arc angular feedforward.
    twist = calls[0]
    assert -0.15 <= twist[1] < 0
    np.testing.assert_allclose(twist[3:] + np.cross(twist[:3], offset), desired)
    np.testing.assert_array_equal(fsm._hold_rotation, np.eye(3))
