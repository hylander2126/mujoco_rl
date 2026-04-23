"""
env.py
------
MuJoCo environment wrapper for the IRB120 tray-transport / behavior-cloning task.

The robot carries a welded tray on its FT sensor. The task is to transport
a box (object 0) placed on the tray.

Designed for behavior cloning: reset(), step(joint_targets), and observation
collection. No reward shaping — the caller is responsible for defining loss.

Action
------
Task-space position targets, 4D shape (dx, dy, droll, dpitch) (2D trans + 2D orient).

Observation (OBS_DIM = 24)
--------------------------
Index   Description                         Units / Notes
------  ----------------------------------  ----------------------------
  0:4   Tray pose (4dof) proprioception     m, rads
  4:8   Tray velocities (4dof)              m/s, rad/s, world frame
 8:11   Object pose wrt tray (and z rot)    m, rads, tray frame
11:14   Object velocities wrt tray          m/s, rad/s, tray frame
14:18   Goal relative pose wrt tray         m, rads, tray frame
18:26   F/T reading (instant)               N, N·m, sensor frame

Usage
-----
    env = IRB120Env()
    obs, info = env.reset()
    for step in demo:
        obs, done, info = env.step(step["q_target"])
        dataset.append((obs, step["q_target"]))
    env.close()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from mujoco_irb120.util.load_obj_in_env import load_environment
from mujoco_irb120.controllers.controllers import PositionController, VelocityController

# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------
_REPO_ROOT   = Path(__file__).resolve().parents[4]
_PARAMS_FILE = _REPO_ROOT / "src" / "mujoco_irb120" / "assets" / "object_params.json"

OBS_DIM = 24   # see module docstring for layout

# Joint limits (radians) — mirrored from robot.xml for clipping
_Q_MIN = np.array([-2.87979, -1.91986, -1.22173, -2.79252, -2.09440, -3.14200])
_Q_MAX = np.array([ 2.87979,  1.91986,  1.91986,  2.79252,  2.09440,  3.14200])


class IRB120Env:
    """
    IRB120 tray-robot environment for behavior cloning.

    Parameters
    ----------
    object_id : int
        Object to load. Only 0 (box) is currently supported with the tray robot.
    controller_type : str
        "position" or "velocity". Velocity is recommended for BC (direct dx tray targets).
    max_sim_time : float
        Episode truncation wall-clock sim time in seconds.
    render_mode : str or None
        "human" opens the MuJoCo passive viewer; None runs headless.
    seed : int or None
        RNG seed (for future domain randomisation).
    """

    def __init__(
        self,
        object_id: int = 0,
        controller_type: str = "position",
        max_sim_time: float = 30.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.object_id       = object_id
        self.controller_type = controller_type
        self.max_sim_time    = max_sim_time
        self.render_mode     = render_mode
        self.np_random       = np.random.default_rng(seed)

        with open(_PARAMS_FILE) as f:
            p = json.load(f)["objects"][str(object_id)]
        self.mass_gt   = float(p["mass_gt"])
        self.com_gt    = np.array(p["com_gt_onshape"]) - np.array(p["com_gt_offset"])
        self._init_xyz = np.array(p["init_xyz"])

        self.model:  Optional[mujoco.MjModel] = None
        self.data:   Optional[mujoco.MjData]  = None
        self.irb:    Optional[PositionController] = None
        self._viewer = None

        self._obj_body_id: int = -1
        self._episode_steps = 0
        self._total_steps   = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:  # noqa: ARG002
        """
        Rebuild the MuJoCo model, place the robot at the init pose, bias the
        F/T sensor, and return the first observation.
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._close_viewer()

        self.model, self.data = load_environment(
            num=self.object_id,
            launch_viewer=False,
            controller_type=self.controller_type,
        )
        assert self.model is not None, "load_environment() returned None."

        self.irb = (VelocityController if self.controller_type == "velocity"
                    else PositionController)(self.model, self.data)

        self._patch_irb_for_tray()

        T_init = self.irb.FK()
        T_init[:3, 3] = self._init_xyz
        try:
            q_init = self.irb.IK(T_init, method=2, damping=0.5, max_iters=1000)
        except RuntimeError:
            print("[IRB120Env] IK did not converge at reset; using home pose.")
            q_init = self.data.qpos[self.irb.joint_idx].copy()
        self.irb.set_pose(q=q_init)
        self.irb.ft_bias(n_samples=200)

        self._episode_steps = 0

        if self.render_mode == "human":
            self._open_viewer()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        """
        Apply joint-position targets and advance one physics step.

        Parameters
        ----------
        action : np.ndarray shape (6,)
            Desired joint angles in radians. Clipped to joint limits.
        """
        assert self.model is not None, "Call reset() before step()."

        q_target = np.clip(np.asarray(action, dtype=float).reshape(6), _Q_MIN, _Q_MAX)
        self.irb.set_pos_ctrl(q_target, check_ellipsoid=False)
        mujoco.mj_step(self.model, self.data)

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        self._episode_steps += 1
        self._total_steps   += 1

        return self._get_obs(), self.data.time >= self.max_sim_time, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            if self._viewer is not None:
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            renderer.close()
            return pixels
        return None

    def close(self):
        self._close_viewer()

    def save_episode(self, path: str, obs_list: list, action_list: list):
        """Save a collected episode to .npz for offline BC training."""
        np.savez(
            path,
            observations = np.stack(obs_list,    axis=0).astype(np.float32),
            actions      = np.stack(action_list, axis=0).astype(np.float32),
            object_id    = np.array([self.object_id]),
            mass_gt      = np.array([self.mass_gt]),
            com_gt       = self.com_gt,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Build the flat observation vector (OBS_DIM = 24).

        NOTE: the obs layout in the module docstring is the intended design;
        update this method once tray-frame kinematics are fully defined.
        Currently returns the raw sensor quantities that feed into those terms:

            [ 0: 6]  q         — joint positions             [rad]
            [ 6:12]  qdot      — joint velocities            [rad/s]
            [12:18]  ft        — F/T in sensor frame         [N, N·m]
            [18:21]  ee_pos    — tool0 flange xyz            [m]
            [21:24]  box_pos   — box xyz in world frame      [m]
        """
        mujoco.mj_forward(self.model, self.data)

        q       = self.data.qpos[self.irb.joint_idx].copy().astype(float)
        qdot    = self.data.qvel[self.irb.joint_dof_idx].copy().astype(float)
        ft      = self.irb.ft_get_reading(grav_comp=True, apply_bias=True, flip_sign=True)
        ee_pos  = self.data.site_xpos[self.irb.ee_site].copy()
        box_pos = self.data.xpos[self._obj_body_id].copy()

        obs = np.concatenate([q, qdot, ft, ee_pos, box_pos]).astype(np.float32)
        assert obs.shape == (OBS_DIM,), f"Obs shape {obs.shape} != {OBS_DIM}"
        return obs

    def _get_info(self) -> dict:
        return {
            "object_id":     self.object_id,
            "mass_gt":       self.mass_gt,
            "com_gt":        self.com_gt.tolist(),
            "sim_time":      float(self.data.time) if self.data else 0.0,
            "episode_steps": self._episode_steps,
        }

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _patch_irb_for_tray(self):
        """
        Fix up Robot attributes that reference sites/bodies that no longer
        exist in the tray robot (no pusher, no ball, no fingertip, no obj_frame).
        """
        self._obj_body_id        = self.model.body("payload").id
        self.irb.payload_body_id = self._obj_body_id
        self.irb.pusher_body_id  = -1
        self.irb.ball_geom_id    = None
        # Gravity-comp mass should reflect the tray, not a pusher link
        self.irb.grav_mass = float(np.asarray(self.model.body("tray").mass).flat[0])
        # fingertip site is used in grav-comp torque offset — point it at sensor site
        self.irb.fingertip_site  = self.irb.ft_site

    def _open_viewer(self):
        try:
            import mujoco.viewer as mjv
            self._viewer = mjv.launch_passive(self.model, self.data)
        except Exception as e:
            print(f"[IRB120Env] Could not open viewer: {e}")

    def _close_viewer(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"IRB120Env(object_id={self.object_id}, "
            f"controller={self.controller_type}, "
            f"max_sim_time={self.max_sim_time})"
        )
