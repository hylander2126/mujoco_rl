"""VLA-specific IRB120 simulation environment.

This environment is intentionally separate from the RL environment. It keeps
the same light reset/step/render shape, but its defaults are chosen for VLA
data collection:

- start from a named home joint pose by default;
- return proprioceptive state for behavior cloning;
- render RGB frames at configurable image size;
- keep a domain-randomization hook that can grow beyond starting pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from mujoco_irb120.controllers.controllers import PositionController, VelocityController
from mujoco_irb120.VLA.hw1_constants import (
    HW1_BIN_SITE_BY_COLOR,
    HW1_CUBE_COLORS,
    HW1_CUBE_RGBA,
    HW1_HOME_Q,
)
from mujoco_irb120.util.load_obj_in_env import load_vla_binsort_environment

OBS_DIM = 24

_Q_MIN = np.array([-2.87979, -1.91986, -1.22173, -2.79252, -2.09440, -3.14200])
_Q_MAX = np.array([2.87979, 1.91986, 1.91986, 2.79252, 2.09440, 3.14200])


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Optional VLA domain-randomization settings.

    All fields are disabled by default. The broader categories are here on
    purpose: pose is only one kind of variation we will eventually want.
    """

    enabled: bool = False
    home_joint_noise_std: float = 0.0
    object_position_noise_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_position_noise_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    light_position_noise_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_noise_std: float = 0.0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "DomainRandomizationConfig":
        if not values:
            return cls()
        kwargs = dict(values)
        for key in (
            "object_position_noise_xyz",
            "camera_position_noise_xyz",
            "light_position_noise_xyz",
        ):
            if key in kwargs:
                kwargs[key] = tuple(float(v) for v in kwargs[key])
        return cls(**kwargs)


class VLAIRB120Env:
    """Simulation-only environment for VLA data collection and evaluation."""

    def __init__(
        self,
        object_id: int = 0,
        controller_type: str = "position",
        max_sim_time: float = 30.0,
        render_mode: Optional[str] = None,
        image_height: int = 128,
        image_width: int = 128,
        camera_name: str = "vla_cam",
        cube_color: str = "random",
        home_q: Optional[np.ndarray] = None,
        domain_randomization: Optional[DomainRandomizationConfig | dict] = None,
        seed: Optional[int] = None,
    ):
        self.object_id = object_id
        self.controller_type = controller_type
        self.max_sim_time = max_sim_time
        self.render_mode = render_mode
        self.image_height = image_height
        self.image_width = image_width
        self.camera_name = camera_name
        self.cube_color_mode = cube_color
        self.cube_color = "red"
        self.home_q = np.asarray(HW1_HOME_Q if home_q is None else home_q, dtype=np.float32).reshape(6)
        self.np_random = np.random.default_rng(seed)

        if isinstance(domain_randomization, DomainRandomizationConfig):
            self.domain_randomization = domain_randomization
        else:
            self.domain_randomization = DomainRandomizationConfig.from_dict(domain_randomization)

        self.mass_gt = 0.08
        self.com_gt = np.zeros(3, dtype=float)

        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.irb: Optional[PositionController] = None
        self._viewer = None
        self._renderer: Optional[mujoco.Renderer] = None
        self._obj_body_id = -1
        self._cube_joint_id = -1
        self._cube_geom_id = -1
        self._tray_site_id = -1
        self._bin_site_ids: dict[str, int] = {}
        self._episode_steps = 0
        self._total_steps = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Rebuild the scene, apply home/default randomized state, and observe."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._close_viewer()
        self._close_renderer()

        self.model, self.data = load_vla_binsort_environment(
            launch_viewer=False,
            controller_type=self.controller_type,
        )
        assert self.model is not None and self.data is not None, "load_vla_binsort_environment() returned None."

        self.irb = (VelocityController if self.controller_type == "velocity" else PositionController)(
            self.model,
            self.data,
        )
        self._obj_body_id = self._find_object_body_id()
        self._cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "sort_cube_free")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sort_cube_geom")
        self._tray_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site:tray_center")
        self._bin_site_ids = {
            color: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for color, site_name in HW1_BIN_SITE_BY_COLOR.items()
        }

        q_home = self._sample_home_q(options)
        self._set_robot_joint_pose(q_home)
        self.cube_color = self._sample_cube_color(options)
        self._set_cube_color(self.cube_color)
        self._populate_cube_on_tray()
        self._apply_domain_randomization()
        self.irb.ft_bias(n_samples=200)

        self._episode_steps = 0
        if self.render_mode == "human":
            self._open_viewer()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        assert self.model is not None and self.data is not None, "Call reset() before step()."

        q_target = np.clip(np.asarray(action, dtype=float).reshape(6), _Q_MIN, _Q_MAX)
        if self.domain_randomization.enabled and self.domain_randomization.action_noise_std > 0.0:
            q_target += self.np_random.normal(0.0, self.domain_randomization.action_noise_std, size=6)
            q_target = np.clip(q_target, _Q_MIN, _Q_MAX)

        self.irb.set_pos_ctrl(q_target, check_ellipsoid=False)
        mujoco.mj_step(self.model, self.data)

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        self._episode_steps += 1
        self._total_steps += 1
        return self._get_obs(), self.data.time >= self.max_sim_time, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            if self._viewer is not None:
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            return self.capture_image()
        return None

    def capture_image(self) -> np.ndarray:
        """Render an RGB frame from the configured VLA camera."""
        assert self.model is not None and self.data is not None, "Call reset() before rendering."
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.image_height,
                width=self.image_width,
            )
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def close(self) -> None:
        self._close_viewer()
        self._close_renderer()

    def _sample_home_q(self, options: Optional[dict]) -> np.ndarray:
        q_home = self.home_q.copy()
        if options and "home_q" in options:
            q_home = np.asarray(options["home_q"], dtype=np.float32).reshape(6)

        if self.domain_randomization.enabled and self.domain_randomization.home_joint_noise_std > 0.0:
            q_home += self.np_random.normal(
                0.0,
                self.domain_randomization.home_joint_noise_std,
                size=6,
            ).astype(np.float32)
        return np.clip(q_home, _Q_MIN, _Q_MAX)

    def _sample_cube_color(self, options: Optional[dict]) -> str:
        color = options.get("cube_color") if options else self.cube_color_mode
        if color == "random":
            return str(self.np_random.choice(HW1_CUBE_COLORS))
        if color not in HW1_CUBE_COLORS:
            raise ValueError(f"cube_color must be one of {HW1_CUBE_COLORS} or 'random', got {color!r}")
        return str(color)

    def _set_robot_joint_pose(self, q: np.ndarray) -> None:
        self.data.qpos[self.irb.joint_idx] = q
        self.data.qvel[self.irb.joint_dof_idx] = 0.0
        self.irb.set_pos_ctrl(q, check_ellipsoid=False)
        mujoco.mj_forward(self.model, self.data)
        self.irb.kb_q_des = q.copy()
        self.irb.kb_goal_pose = None

    def _set_cube_color(self, color: str) -> None:
        if self._cube_geom_id < 0:
            raise RuntimeError("Could not find sort_cube_geom.")
        self.model.geom_rgba[self._cube_geom_id] = HW1_CUBE_RGBA[color]

    def _populate_cube_on_tray(self) -> None:
        if self._cube_joint_id < 0:
            raise RuntimeError("Could not find sort_cube_free joint.")
        if self._tray_site_id < 0:
            raise RuntimeError("Could not find site:tray_center.")

        mujoco.mj_forward(self.model, self.data)
        qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        tray_pos = self.data.site_xpos[self._tray_site_id].copy()
        tray_mat = self.data.site_xmat[self._tray_site_id].reshape(3, 3).copy()
        tray_normal = tray_mat[:, 2]
        cube_half_extent = 0.025
        tray_half_thickness = 0.015
        cube_pos = tray_pos + tray_normal * (cube_half_extent + tray_half_thickness + 0.004)

        self.data.qpos[qpos_adr : qpos_adr + 3] = cube_pos
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.model.jnt_dofadr[self._cube_joint_id] : self.model.jnt_dofadr[self._cube_joint_id] + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _apply_domain_randomization(self) -> None:
        if not self.domain_randomization.enabled:
            mujoco.mj_forward(self.model, self.data)
            return

        self._randomize_free_body_position(self._obj_body_id, self.domain_randomization.object_position_noise_xyz)
        self._randomize_named_position(
            name=self.camera_name,
            obj_type=mujoco.mjtObj.mjOBJ_CAMERA,
            noise_xyz=self.domain_randomization.camera_position_noise_xyz,
        )
        self._randomize_named_position(
            name="light0",
            obj_type=mujoco.mjtObj.mjOBJ_LIGHT,
            noise_xyz=self.domain_randomization.light_position_noise_xyz,
        )
        mujoco.mj_forward(self.model, self.data)

    def _randomize_free_body_position(self, body_id: int, noise_xyz: tuple[float, float, float]) -> None:
        noise_scale = np.asarray(noise_xyz, dtype=float)
        if not np.any(noise_scale):
            return
        jnt_id = int(self.model.body_jntadr[body_id])
        if jnt_id < 0:
            return
        qpos_adr = int(self.model.jnt_qposadr[jnt_id])
        if self.model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[qpos_adr : qpos_adr + 3] += self.np_random.uniform(-noise_scale, noise_scale)

    def _randomize_named_position(
        self,
        name: str,
        obj_type: mujoco.mjtObj,
        noise_xyz: tuple[float, float, float],
    ) -> None:
        noise_scale = np.asarray(noise_xyz, dtype=float)
        if not np.any(noise_scale):
            return
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            return
        delta = self.np_random.uniform(-noise_scale, noise_scale)
        if obj_type == mujoco.mjtObj.mjOBJ_CAMERA:
            self.model.cam_pos[obj_id] += delta
        elif obj_type == mujoco.mjtObj.mjOBJ_LIGHT:
            self.model.light_pos[obj_id] += delta

    def _get_obs(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)

        q = self.data.qpos[self.irb.joint_idx].copy().astype(float)
        qdot = self.data.qvel[self.irb.joint_dof_idx].copy().astype(float)
        ft = self.irb.ft_get_reading(grav_comp=True, apply_bias=True, flip_sign=True)
        ee_pos = self.data.site_xpos[self.irb.ee_site].copy()
        obj_pos = self.data.xpos[self._obj_body_id].copy()

        obs = np.concatenate([q, qdot, ft, ee_pos, obj_pos]).astype(np.float32)
        assert obs.shape == (OBS_DIM,), f"Obs shape {obs.shape} != {OBS_DIM}"
        return obs

    def get_bin_position(self, color: str) -> np.ndarray:
        if color not in self._bin_site_ids:
            raise ValueError(f"Unknown bin color: {color}")
        return self.data.site_xpos[self._bin_site_ids[color]].copy()

    def get_cube_position(self) -> np.ndarray:
        return self.data.xpos[self._obj_body_id].copy()

    def success(self, color: Optional[str] = None) -> bool:
        target_color = color or self.cube_color
        cube_xy = self.get_cube_position()[:2]
        bin_xy = self.get_bin_position(target_color)[:2]
        in_xy = np.linalg.norm(cube_xy - bin_xy) < 0.11
        low_enough = self.get_cube_position()[2] < 0.18
        return bool(in_xy and low_enough)

    def reward(self) -> float:
        return 1.0 if self.success() else 0.0

    def _get_info(self) -> dict:
        return {
            "env": "VLAIRB120Env",
            "task": "hw1_bin_sort",
            "object_id": self.object_id,
            "cube_color": self.cube_color,
            "instruction": f"sort the {self.cube_color} object into the {self.cube_color} bin",
            "mass_gt": self.mass_gt,
            "com_gt": self.com_gt.tolist(),
            "home_q": self.home_q.tolist(),
            "reward": self.reward(),
            "success": self.success(),
            "domain_randomization": self.domain_randomization.enabled,
            "sim_time": float(self.data.time) if self.data else 0.0,
            "episode_steps": self._episode_steps,
        }

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer as mjv

            self._viewer = mjv.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self._use_fixed_viewer_camera(self._viewer, self.camera_name)
        except Exception as e:
            print(f"[VLAIRB120Env] Could not open viewer: {e}")

    def _use_fixed_viewer_camera(self, viewer, camera_name: str) -> None:
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            print(f"[VLAIRB120Env] Could not find camera {camera_name!r}; viewer will use free camera.")
            return
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

    def _close_viewer(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _find_object_body_id(self) -> int:
        candidate_names = ("sort_cube", "payload", "box_base")
        for name in candidate_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        raise RuntimeError(f"Could not find object body. Tried: {candidate_names}")

    def __enter__(self) -> "VLAIRB120Env":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"VLAIRB120Env(object_id={self.object_id}, "
            f"controller={self.controller_type}, "
            f"home_q={self.home_q.tolist()}, "
            f"domain_randomization={self.domain_randomization.enabled})"
        )
