from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from parameter_estimation.controllers.press_pull_fsm import PressPullConfig, PressPullFSM
from parameter_estimation.press_pull_demo import BoxDemoConfig, prepare_box
from parameter_estimation.scene import load_environment


def test_default_controller_keeps_legacy_behavior():
    cfg = PressPullConfig()
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
    assert cfg.arc_force_drop_fraction == 0.1
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


def test_command_holds_orientation_and_preserves_ball_velocity():
    calls = []
    offset = np.array([0.18, 0.01, 0.02])

    def pose(which):
        out = np.eye(4)
        if which == 'ball':
            out[:3, 3] = offset
        return out

    fsm = PressPullFSM.__new__(PressPullFSM)
    fsm.cfg = PressPullConfig()
    fsm._hold_rotation = None
    fsm.irb = SimpleNamespace(get_site_pose=pose,
                             apply_cartesian_keyboard_ctrl=lambda v, **kw: calls.append((v, kw)))
    desired = np.array([-0.008, 0, -0.001])
    fsm._command(*desired)
    twist, options = calls[0]
    np.testing.assert_allclose(twist[:3], 0, atol=1e-12)  # already at the held orientation
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


def test_pre_squash_places_ball_at_home_orientation():
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    from mujoco_irb120.robot.controllers.robot import controller
    model, data, _, cfg, _ = prepare_box(BoxDemoConfig(), verbose=False)
    irb = controller(model, data)
    home_rotation = irb.FK()[:3, :3].copy()
    fsm = PressPullFSM(irb, model, data, replace(cfg, press_offset_xy=(0.044, -0.044)))
    top = fsm.object_top_center()
    target = top + np.array([0.044, -0.044, cfg.approach_clearance_m])
    fsm.move_to_pre_squash()
    np.testing.assert_allclose(irb.get_site_pose('ball')[:3, 3], target, atol=0.002)
    actual_rotation = irb.FK()[:3, :3]
    error = Rotation.from_matrix(actual_rotation @ home_rotation.T).magnitude()
    assert error < np.deg2rad(0.1)


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
    fsm._command(*desired)
    twist = calls[0]
    assert -0.15 <= twist[1] < 0
    np.testing.assert_allclose(twist[3:] + np.cross(twist[:3], offset), desired)
    np.testing.assert_array_equal(fsm._hold_rotation, np.eye(3))
