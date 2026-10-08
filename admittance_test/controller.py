from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_irb120.robot.controllers.robot import controller as IRBController


@dataclass(frozen=True)
class AdmittanceConfig:
    """Cartesian admittance settings for the demo."""

    target_force_n: float = 2.0
    virtual_mass_kg: float = 0.15
    damping_ns_m: float = 8.0
    orientation_kp: float = 12.0
    orientation_kv: float = 1.5
    max_linear_speed_m_s: float = 0.08
    max_angular_speed_rad_s: float = 0.5
    max_joint_speed_rad_s: float = 1.0

    def __post_init__(self) -> None:
        if self.virtual_mass_kg <= 0 or self.damping_ns_m < 0:
            raise ValueError("Virtual mass must be positive and damping nonnegative")
        if self.target_force_n < 0 or self.max_linear_speed_m_s <= 0:
            raise ValueError("Force and speed settings must be nonnegative and positive")


class AdmittanceController:
    """Force-to-motion controller around the repository's IRB120 wrapper.

    MuJoCo's scene uses position actuators, so the Cartesian admittance velocity
    is integrated into a joint-position target before it is sent to ``ctrl``.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 config: AdmittanceConfig | None = None) -> None:
        self.model = model
        self.data = data
        self.irb = IRBController(model, data)
        self.config = config or AdmittanceConfig()
        self.q_target = data.qpos[self.irb.joint_idx].copy()
        self.R_desired = self.irb.get_site_pose("ee")[:3, :3].copy()
        self.last_force_world = np.zeros(3)
        self.last_velocity = np.zeros(6)

    def set_target_force(self, target_force_n: float) -> None:
        """Change the +X force target while the simulation is running."""
        if target_force_n < 0:
            raise ValueError("Target force must be nonnegative")
        self.config = replace(self.config, target_force_n=float(target_force_n))

    def reset(self) -> None:
        """Reset the integrated target and hold the current tool orientation."""
        mujoco.mj_forward(self.model, self.data)
        self.q_target = self.data.qpos[self.irb.joint_idx].copy()
        self.R_desired = self.irb.get_site_pose("ee")[:3, :3].copy()
        self.last_force_world.fill(0.0)
        self.last_velocity.fill(0.0)

    def _force_world(self) -> np.ndarray:
        wrench_sensor = self.irb.ft_get_reading(grav_comp=True, apply_bias=False)
        sensor_id = self.irb.ft_site
        sensor_rotation = self.data.site_xmat[sensor_id].reshape(3, 3)
        return sensor_rotation @ wrench_sensor[:3]

    def step(self) -> dict[str, np.ndarray | float]:
        """Advance the admittance command by one simulation timestep."""
        mujoco.mj_forward(self.model, self.data)
        dt = self.model.opt.timestep
        J = self.irb.get_jacobian()
        velocity = J @ self.data.qvel[self.irb.joint_dof_idx]
        force_world = self._force_world()

        force_error = np.array([
            self.config.target_force_n - force_world[0],
            -force_world[1],
            -force_world[2],
        ])
        acceleration = (force_error - self.config.damping_ns_m * velocity[3:]) / self.config.virtual_mass_kg
        linear_velocity = velocity[3:].copy()
        linear_velocity[0] += acceleration[0] * dt
        linear_velocity[1:] = 0.0
        linear_velocity = np.clip(
            linear_velocity,
            -self.config.max_linear_speed_m_s,
            self.config.max_linear_speed_m_s,
        )

        current_rotation = self.irb.get_site_pose("ee")[:3, :3]
        orientation_error = Rotation.from_matrix(self.R_desired @ current_rotation.T).as_rotvec()
        angular_velocity = (
            self.config.orientation_kp * orientation_error
            - self.config.orientation_kv * velocity[:3]
        )
        angular_velocity = np.clip(
            angular_velocity,
            -self.config.max_angular_speed_rad_s,
            self.config.max_angular_speed_rad_s,
        )

        twist = np.concatenate([angular_velocity, linear_velocity])
        q_dot = np.linalg.pinv(J) @ twist
        max_q_dot = np.max(np.abs(q_dot))
        if max_q_dot > self.config.max_joint_speed_rad_s:
            q_dot *= self.config.max_joint_speed_rad_s / max_q_dot
        self.q_target = np.clip(
            self.q_target + q_dot * dt,
            self.irb.q_min,
            self.irb.q_max,
        )
        self.data.ctrl[self.irb.joint_idx] = self.q_target

        self.last_force_world = force_world
        self.last_velocity = velocity
        return {
            "force_world": force_world.copy(),
            "tool_position": self.irb.get_site_pose("ee")[:3, 3].copy(),
            "tool_velocity": velocity.copy(),
            "command_velocity": twist.copy(),
            "time": float(self.data.time),
        }
