"""VLA-specific IRB120 simulation environment.

This environment owns the simulation loop for VLA data collection:

- start from a named home joint pose by default;
- return proprioceptive state for behavior cloning;
- render RGB frames at configurable image size;
- keep a domain-randomization hook that can grow beyond starting pose.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

# MuJoCo's default GLFW backend needs X11. Prefer EGL automatically on SSH and
# other headless sessions; callers can still explicitly select another backend.
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from environment.scene import load_hw1_scene
from robot.controllers.robot import PositionController
from task import BinSortTaskSpec, HW1_TASK, swap_bin_colors

OBS_DIM = 24

_Q_MIN = np.array([-2.87979, -1.91986, -1.22173, -2.79252, -2.09440, -3.14200])
_Q_MAX = np.array([2.87979, 1.91986, 1.91986, 2.79252, 2.09440, 3.14200])
_WAYLAND_WINDOW_POSITION_WARNING = ".*Wayland: The platform does not provide the window position.*"


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
        max_sim_time: float = 30.0,
        render_mode: Optional[str] = None,
        image_height: int = 128,
        image_width: int = 128,
        cube_color: str = "random",
        home_q: Optional[np.ndarray] = None,
        task: BinSortTaskSpec = HW1_TASK,
        domain_randomization: Optional[DomainRandomizationConfig | dict] = None,
        seed: Optional[int] = None,
    ):
        self._base_task = task
        self.task = task
        self.max_sim_time = max_sim_time
        self.render_mode = render_mode
        self.image_height = image_height
        self.image_width = image_width
        self.camera_name = task.camera_name
        self.cube_color_mode = cube_color
        self.cube_color = "red"
        self.swap_bins = False
        self.home_q = np.asarray(task.home_q if home_q is None else home_q, dtype=np.float32).reshape(6)
        self.ft_bias_enabled = False
        self.ft_bias_samples = 0
        self.np_random = np.random.default_rng(seed)

        if isinstance(domain_randomization, DomainRandomizationConfig):
            self.domain_randomization = domain_randomization
        else:
            self.domain_randomization = DomainRandomizationConfig.from_dict(domain_randomization)

        self.mass_gt = 0.0
        self.com_gt = np.zeros(3, dtype=float)

        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.irb: Optional[PositionController] = None
        self._viewer = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._obj_body_id = -1
        self._cube_joint_id = -1
        self._cube_geom_id = -1
        self._tray_geom_id = -1
        self._tray_site_id = -1
        self._bin_site_ids: dict[str, int] = {}
        self._episode_steps = 0
        self._total_steps = 0
        self._success_hold_time = 0.0
        self._last_done_reason = "not_done"

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

        self.swap_bins = self._sample_swap_bins(options)
        self.task = swap_bin_colors(self._base_task) if self.swap_bins else self._base_task

        self.model, self.data = load_hw1_scene(
            task=self.task,
        )

        self.irb = PositionController(self.model, self.data)
        self._obj_body_id = self._find_object_body_id()
        self._cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "sort_cube_free")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sort_cube_geom")
        self._tray_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tray_geom")
        self._tray_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site:tray_center")
        self._bin_site_ids = {
            color: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for color, site_name in self.task.bin_site_by_color.items()
        }
        self.mass_gt = float(self.model.body_mass[self._obj_body_id])
        self.com_gt = self.model.body_ipos[self._obj_body_id].copy().astype(float)

        q_home = self._sample_home_q(options)
        self._set_robot_joint_pose(q_home)
        self.cube_color = self._sample_cube_color(options)
        self._set_cube_color(self.cube_color)
        self._populate_cube_on_tray()
        self._apply_domain_randomization()

        self._episode_steps = 0
        self._success_hold_time = 0.0
        self._last_done_reason = "not_done"
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
        done = self._episode_done()
        return self._get_obs(), done, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            if self._viewer is not None:
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            return self.capture_image()
        return None

    def capture_image(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Render an RGB frame from the configured VLA camera."""
        assert self.model is not None and self.data is not None, "Call reset() before rendering."
        height = self.image_height if height is None else height
        width = self.image_width if width is None else width
        renderer_key = (height, width)
        if renderer_key not in self._renderers:
            self._renderers[renderer_key] = mujoco.Renderer(
                self.model,
                height=height,
                width=width,
            )
        renderer = self._renderers[renderer_key]
        renderer.update_scene(self.data, camera=self.camera_name)
        return renderer.render()

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
            return str(self.np_random.choice(self.task.colors))
        if color not in self.task.colors:
            raise ValueError(f"cube_color must be one of {self.task.colors} or 'random', got {color!r}")
        return str(color)

    def _sample_swap_bins(self, options: Optional[dict]) -> bool:
        value = options.get("swap_bins", False) if options else False
        if value == "random":
            return bool(self.np_random.choice([False, True]))
        return bool(value)

    def _set_robot_joint_pose(self, q: np.ndarray) -> None:
        self.data.qpos[self.irb.joint_idx] = q
        self.data.qvel[self.irb.joint_dof_idx] = 0.0
        self.irb.set_pos_ctrl(q, check_ellipsoid=False)
        mujoco.mj_forward(self.model, self.data)

    def _set_cube_color(self, color: str) -> None:
        if self._cube_geom_id < 0:
            raise RuntimeError("Could not find sort_cube_geom.")
        self.model.geom_rgba[self._cube_geom_id] = self.task.cube_rgba[color]

    def _bias_force_torque_without_consuming_episode(self) -> None:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ctrl = self.data.ctrl.copy()
        time = float(self.data.time)

        self.irb.ft_bias(n_samples=200)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        self.data.time = time
        mujoco.mj_forward(self.model, self.data)

    def _populate_cube_on_tray(self) -> None:
        if self._cube_joint_id < 0:
            raise RuntimeError("Could not find sort_cube_free joint.")
        if self._cube_geom_id < 0:
            raise RuntimeError("Could not find sort_cube_geom.")
        if self._tray_geom_id < 0:
            raise RuntimeError("Could not find tray_geom.")
        if self._tray_site_id < 0:
            raise RuntimeError("Could not find site:tray_center.")

        mujoco.mj_forward(self.model, self.data)
        qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        tray_pos = self.data.site_xpos[self._tray_site_id].copy()
        tray_mat = self.data.site_xmat[self._tray_site_id].reshape(3, 3).copy()
        tray_normal = tray_mat[:, 2]
        cube_half_extent = float(np.max(self.model.geom_size[self._cube_geom_id]))
        tray_half_thickness = float(self.model.geom_size[self._tray_geom_id][2])
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
        in_xy = np.linalg.norm(cube_xy - bin_xy) < self.task.success_bin_radius
        low_enough = self.get_cube_position()[2] < self.task.success_cube_max_z
        return bool(in_xy and low_enough)

    def reward(self) -> float:
        return 1.0 if self.success() else 0.0

    def _cube_speed(self) -> float:
        if self._cube_joint_id < 0:
            return float("inf")
        qvel_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
        return float(np.linalg.norm(self.data.qvel[qvel_adr : qvel_adr + 3]))

    def _stable_success(self) -> bool:
        return self.success() and self._cube_speed() < self.task.success_cube_max_speed

    def _episode_done(self) -> bool:
        if self._stable_success():
            self._success_hold_time += float(self.model.opt.timestep)
        else:
            self._success_hold_time = 0.0

        if self._success_hold_time >= self.task.success_hold_seconds:
            self._last_done_reason = "success"
            return True

        if self.data.time >= self.max_sim_time:
            self._last_done_reason = "timeout"
            return True

        self._last_done_reason = "not_done"
        return False

    def _get_info(self) -> dict:
        return {
            "env": "VLAIRB120Env",
            "task": "hw1_bin_sort",
            "cube_color": self.cube_color,
            "swap_bins": self.swap_bins,
            "instruction": self.task.instruction_template.format(color=self.cube_color),
            "mass_gt": self.mass_gt,
            "com_gt": self.com_gt.tolist(),
            "home_q": self.home_q.tolist(),
            "ft_bias_enabled": self.ft_bias_enabled,
            "ft_bias_samples": self.ft_bias_samples,
            "reward": self.reward(),
            "success": self.success(),
            "stable_success": self._stable_success(),
            "success_hold_time": self._success_hold_time,
            "done_reason": self._last_done_reason,
            "domain_randomization": self.domain_randomization.enabled,
            "sim_time": float(self.data.time) if self.data else 0.0,
            "episode_steps": self._episode_steps,
        }

    def _open_viewer(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_WAYLAND_WINDOW_POSITION_WARNING,
                    category=Warning,
                    module=r"glfw(\.|$).*",
                )
                import mujoco.viewer as mjv

                self._viewer = mjv.launch_passive(
                    self.model,
                    self.data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            self._use_debug_viewer_camera(self._viewer)
        except Exception as e:
            print(f"[VLAIRB120Env] Could not open viewer: {e}")

    def _use_debug_viewer_camera(self, viewer) -> None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.50, 0.0, 0.30])
        viewer.cam.distance = 1.85
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -35.0

    def _close_viewer(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def _close_renderer(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def _find_object_body_id(self) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sort_cube")
        if body_id < 0:
            raise RuntimeError("Could not find required body 'sort_cube'.")
        return body_id

    def __enter__(self) -> "VLAIRB120Env":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"VLAIRB120Env(home_q={self.home_q.tolist()}, "
            f"domain_randomization={self.domain_randomization.enabled})"
        )
