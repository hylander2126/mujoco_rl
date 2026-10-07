"""Independent checks against the ROS fixed-chain dimensions and measured mass."""
import json

import mujoco
import numpy as np
import pytest

from parameter_estimation.scene import ROBOT_ASSETS, load_environment


def test_hardware_tool_chain_and_inertia():
    model, data = load_environment(0)
    mujoco.mj_forward(model, data)
    flange = model.site('site:tool0').id
    rotation = data.site_xmat[flange].reshape(3, 3)
    def local(site):
        return rotation.T @ (data.site_xpos[model.site(site).id] - data.site_xpos[flange])
    # 10 + 21.5 + 21 + 33.3244 mm sensor stack; CAD Y offset maps to Z.
    np.testing.assert_allclose(local('site:sensor'), [0.0858244, 0, -0.000500516], atol=1e-6)
    np.testing.assert_allclose(local('site:ball_center'), [0.1724756, 0, -0.000500516], atol=2e-6)
    assert np.linalg.norm(local('site:ball_center') - local('site:sensor')) == pytest.approx(0.0866512)
    assert model.geom('push_ball_col').size[0] == pytest.approx(0.01325)
    pusher = model.body('pusher_link')
    assert pusher.mass[0] == pytest.approx(0.066)
    assert np.linalg.norm(pusher.ipos) == pytest.approx(0.0275)
    np.testing.assert_allclose(model.geom('tool_stack_col').size[:2], [0.045, 0.0429])
    # All six imported CAD parts are present, while sphere contact stays analytic.
    for name in ['adapter_robot', 'adapter_sensor', 'sensor_body', 'sensor_plate', 'ball', 'pusher_body']:
        assert model.mesh('hardware_' + name).id >= 0


def test_snapshot_hashes():
    import hashlib
    root = ROBOT_ASSETS / 'hardware_tool'
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_unloaded_gravity_compensation_uses_measured_com():
    from mujoco_irb120.robot.controllers.robot import controller
    from parameter_estimation.controllers.press_pull_fsm import PressPullFSM, PressPullConfig
    model, data = load_environment(0)
    irb = controller(model, data)
    fsm = PressPullFSM(irb, model, data, PressPullConfig(verbose=False))
    fsm.move_to_pre_squash()
    for _ in range(300):
        mujoco.mj_step(model, data)
    # No tare should be needed to cancel a stationary finger's own weight.
    np.testing.assert_allclose(irb.ft_get_reading(grav_comp=True, apply_bias=False), 0, atol=1e-3)
