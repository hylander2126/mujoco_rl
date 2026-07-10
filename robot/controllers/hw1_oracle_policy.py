from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from task import BinSortTaskSpec, HW1_TASK


class HW1BinSortExpert:
    """Ground-truth scripted expert for HW1 bin sorting.

    The expert uses MuJoCo state and robot IK directly. It is not meant to be
    deployed; it is a demonstration generator for behavior cloning.
    """

    def __init__(
        self,
        env,
        cube_color: str | None = None,
        task: BinSortTaskSpec = HW1_TASK,
    ):
        self.env = env
        self.task = task
        self.cube_color = self._read_cube_color(cube_color)
        self.move_duration = task.pre_drop_seconds
        self.tilt_duration = task.tip_seconds
        self.hold_duration = task.drop_hold_seconds
        self.return_duration = task.return_home_seconds
        self.dt = float(env.model.opt.timestep)
        self.step_idx = 0

        self.q_home = env.home_q.copy() if hasattr(env, "home_q") else task.home_q.copy()
        self.start_q = env.data.qpos[env.irb.joint_idx].copy().astype(float)
        self.start_tray_normal = env.data.site_xmat[env._tray_site_id].reshape(3, 3)[:, 2].copy()
        print(f"[HW1 expert] Home pose (cartesian): {np.round(env.irb.FK()[:3, 3], 2)}")

        self.pre_drop_xyz = self.task.pre_drop_xyz_by_color[self.cube_color]

        print(f"[HW1 expert] cube_color={self.cube_color}; solving tray-align IK once...")
        # Keep the tray level (same normal as home) while moving to the
        # pre-drop position, instead of only constraining position.
        self.pre_drop_q = self.env.irb.solve_tray_pose_ik(
            self.pre_drop_xyz, self.start_tray_normal, home_q=self.q_home,
        )
        self.drop_q = self._apply_tip_offset(self.pre_drop_q)
        print("[HW1 expert] Tray-align IK waypoints ready.")
        self.return_start_q = self.drop_q.copy()

    def select_action(self) -> np.ndarray:
        t = self.step_idx * self.dt
        self.step_idx += 1

        if t < self.move_duration:
            alpha = t / self.move_duration
            return self._interpolate_q(self.start_q, self.pre_drop_q, alpha)

        if t < self.move_duration + self.tilt_duration:
            alpha = (t - self.move_duration) / self.tilt_duration
            return self._interpolate_q(self.pre_drop_q, self.drop_q, alpha)

        if t < self.move_duration + self.tilt_duration + self.hold_duration:
            return self.drop_q.astype(np.float32)

        if t < self.move_duration + self.tilt_duration + self.hold_duration + self.return_duration:
            alpha = (
                t
                - self.move_duration
                - self.tilt_duration
                - self.hold_duration
            ) / self.return_duration
            return self._interpolate_q(self.return_start_q, self.q_home, alpha)

        return self.q_home.astype(np.float32)

    def _interpolate_q(self, q_a: np.ndarray, q_b: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        q = (1.0 - alpha) * q_a + alpha * q_b
        return q.astype(np.float32)

    def _read_cube_color(self, cube_color: str | None) -> str:
        color = cube_color or getattr(self.env, "cube_color", None)
        if color not in self.task.pre_drop_xyz_by_color:
            raise ValueError(
                f"HW1BinSortExpert expected cube color in {tuple(self.task.pre_drop_xyz_by_color)}, got {color!r}"
            )
        return str(color)

    def _apply_tip_offset(self, pre_drop_q: np.ndarray) -> np.ndarray:
        """Tip the tray by rotating its normal about the axis tangent to the
        radial-out-from-base direction, so the tray's downhill side points
        from the pre-drop pose straight out toward the bin."""
        bin_xy = np.asarray(self.task.bin_xy_by_color[self.cube_color], dtype=float).reshape(2)
        radial_norm = np.linalg.norm(bin_xy)
        if radial_norm < 1e-6:
            return pre_drop_q.astype(np.float32)
        radial = bin_xy / radial_norm
        tangent = np.array([-radial[1], radial[0], 0.0])

        tip_normal = Rotation.from_rotvec(self.task.tip_tilt_rad * tangent).apply(self.start_tray_normal)

        # Seed from the pre-drop config first (the tip is a small rotation
        # away, so this converges fast); solve_tray_pose_ik falls back to
        # its own seed sweep if that doesn't converge.
        return self.env.irb.solve_tray_pose_ik(
            self.pre_drop_xyz, tip_normal, home_q=self.q_home, seed_hint=pre_drop_q,
        )
