"""Explicit exploratory press-pull physics for non-box collision geometry."""
from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np

from contact_selection.controller import PressPullConfig
from contact_selection.scene import load_environment


@dataclass(frozen=True)
class ArcGripConfig:
    press_force_n: float = 5.0
    ground_friction: float = 0.5
    object_friction: float = 0.5
    finger_friction: float = 2.0
    impratio: float = 10.0
    noslip_iterations: int = 10
    max_normal_speed: float = 0.005
    force_drop_fraction: float = 0.1
    timestep: float = 0.001

    def __post_init__(self):
        values = (self.press_force_n, self.ground_friction, self.object_friction,
                  self.finger_friction, self.impratio, self.max_normal_speed,
                  self.force_drop_fraction, self.timestep)
        if not np.isfinite(values).all():
            raise ValueError('Arc grip parameters must be finite')
        if not 0 < self.press_force_n < PressPullConfig().force_hard_limit_n:
            raise ValueError('Press force must be positive and below the hard limit')
        if min(self.ground_friction, self.object_friction, self.finger_friction) < 0:
            raise ValueError('Friction must be nonnegative')
        if self.impratio <= 0 or self.max_normal_speed <= 0 or self.timestep <= 0:
            raise ValueError('Solver ratio, speed and timestep must be positive')
        if self.noslip_iterations < 0 or not 0 < self.force_drop_fraction < 1:
            raise ValueError('Invalid solver iterations or force-drop fraction')


def prepare_arc_grip(object_id: int, parameters: dict):
    """Load a fresh scene and apply a recorded, uncalibrated contact setup.

    Set both sides of the object/table pair: with equal geom priority MuJoCo
    uses the larger sliding friction value at their contact. This preset does
    not infer a pivot or mark any contact as a known-good reference.
    """
    options = ArcGripConfig(**parameters)
    model, data = load_environment(object_id)
    payload = int(model.site_bodyid[model.site('site:obj_frame').id])
    ball = model.geom('push_ball_col').id
    table = model.geom('table').id
    model.opt.timestep = options.timestep
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = options.impratio
    model.opt.noslip_iterations = options.noslip_iterations
    model.geom_friction[ball, 0] = options.finger_friction
    model.geom_priority[ball] = 1
    model.geom_friction[table, 0] = options.ground_friction
    for gid in np.flatnonzero(model.geom_bodyid == payload):
        model.geom_friction[gid, 0] = options.object_friction
    mujoco.mj_forward(model, data)
    controller = replace(PressPullConfig(verbose=False),
                         force_ref_n=options.press_force_n,
                         max_normal_speed=options.max_normal_speed,
                         rotate_with_arc=True,
                         arc_force_drop_fraction=options.force_drop_fraction)
    return model, data, controller, None, {'name': 'arc_grip', 'parameters': asdict(options)}
