from __future__ import annotations

import mujoco
import numpy as np


class GravityCompController:
    """Apply model bias torques so the robot can be perturbed in free space."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.joint_names = tuple(f"joint_{index}" for index in range(1, 7))
        self.dof_ids = np.array([model.joint(name).dofadr for name in self.joint_names])
        self.actuator_ids = np.array([model.actuator(name).id for name in self.joint_names])
        self.tool_site = model.site("site:tool0").id

    def step(self) -> dict[str, np.ndarray | float]:
        """Update gravity compensation and return a small diagnostic sample."""
        mujoco.mj_forward(self.model, self.data)
        bias_torque = -np.asarray(self.data.qfrc_bias[self.dof_ids]).reshape(-1).copy()
        self.data.ctrl[self.actuator_ids] = bias_torque
        return {
            "time": float(self.data.time),
            "tool_position": self.data.site_xpos[self.tool_site].copy(),
            "joint_torque": bias_torque,
        }
