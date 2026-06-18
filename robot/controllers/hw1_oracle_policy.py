from __future__ import annotations

import numpy as np

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
        self.start_T = env.irb.FK().copy()
        self.start_q = env.data.qpos[env.irb.joint_idx].copy().astype(float)
        print(f"[HW1 expert] Home pose (cartesian): {np.round(self.start_T[:3, 3], 2)}")

        self.pre_drop_T = self._make_pose(task.pre_drop_xyz_by_color[self.cube_color])

        self.pre_drop_q, self.drop_q = self._solve_ik_waypoints()
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

    def _make_pose(self, xyz: np.ndarray) -> np.ndarray:
        T = self.start_T.copy()
        T[:3, 3] = xyz
        return T

    def _solve_ik_waypoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve Cartesian position waypoints into joint targets once per episode."""
        print(f"[HW1 expert] cube_color={self.cube_color}; solving position-only IK once...")

        pre_drop_q = self.env.irb.position_only_IK(
            self.pre_drop_T[:3, 3],
            damping=0.5,
            max_iters=250,
            tol=1e-2,
        ).astype(np.float32)

        drop_q = self._apply_tip_offset(pre_drop_q)
        print("[HW1 expert] Position IK waypoint ready.")
        return pre_drop_q, drop_q

    def _apply_tip_offset(self, pre_drop_q: np.ndarray) -> np.ndarray:
        drop_q = pre_drop_q.copy()
        wrist_roll_idx = 5
        drop_q[wrist_roll_idx] += self.task.tray_tip_rad_by_color[self.cube_color]
        drop_q = np.clip(drop_q, self.env.irb.q_min, self.env.irb.q_max)
        return drop_q.astype(np.float32)
