from __future__ import annotations

import numpy as np

from mujoco_irb120.VLA.hw1_constants import (
    HW1_DROP_HOLD_SECONDS,
    HW1_HOME_Q,
    HW1_PRE_DROP_SECONDS,
    HW1_PRE_DROP_XYZ_BY_COLOR,
    HW1_RETURN_HOME_SECONDS,
    HW1_TIP_SECONDS,
    HW1_TRAY_TIP_RAD_BY_COLOR,
)


def _rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


class HW1BinSortExpert:
    """Ground-truth scripted expert for HW1 bin sorting.

    The expert uses MuJoCo state and robot IK directly. It is not meant to be
    deployed; it is a demonstration generator for behavior cloning.
    """

    def __init__(
        self,
        env,
        cube_color: str | None = None,
    ):
        self.env = env
        self.cube_color = self._read_cube_color(cube_color)
        self.move_duration = HW1_PRE_DROP_SECONDS
        self.tilt_duration = HW1_TIP_SECONDS
        self.hold_duration = HW1_DROP_HOLD_SECONDS
        self.return_duration = HW1_RETURN_HOME_SECONDS
        self.dt = float(env.model.opt.timestep)
        self.step_idx = 0

        self.q_home = env.home_q.copy() if hasattr(env, "home_q") else HW1_HOME_Q.copy()
        self.start_T = env.irb.FK().copy()
        self.start_q = env.data.qpos[env.irb.joint_idx].copy().astype(float)

        self.pre_drop_T = self._make_pose(HW1_PRE_DROP_XYZ_BY_COLOR[self.cube_color])

        self.drop_T = self.pre_drop_T.copy()
        self.drop_T[:3, :3] = self.pre_drop_T[:3, :3] @ _rot_x(
            HW1_TRAY_TIP_RAD_BY_COLOR[self.cube_color]
        )

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
        if color not in HW1_PRE_DROP_XYZ_BY_COLOR:
            raise ValueError(
                f"HW1BinSortExpert expected cube color in {tuple(HW1_PRE_DROP_XYZ_BY_COLOR)}, got {color!r}"
            )
        return str(color)

    def _make_pose(self, xyz: np.ndarray) -> np.ndarray:
        T = self.start_T.copy()
        T[:3, 3] = xyz
        return T

    def _solve_ik_waypoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve Cartesian waypoints into joint targets once per episode.

        This is the only place the HW1 state machine calls robot IK:
        - `pre_drop_T` is the pose above the selected bin.
        - `drop_T` is the same pose with tray tilt applied.

        Runtime control then interpolates between these solved joint targets.
        """
        print(f"[HW1 expert] cube_color={self.cube_color}; solving IK waypoints once...")
        pre_drop_q = self._ik_to("pre_drop", self.pre_drop_T, fallback=self.start_q)
        drop_q = self._ik_to("drop", self.drop_T, fallback=pre_drop_q)
        print("[HW1 expert] IK waypoints ready.")
        return pre_drop_q, drop_q

    def _ik_to(self, name: str, target_T: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        try:
            q = self.env.irb.IK(target_T, method=2, damping=0.5, max_iters=250)
        except RuntimeError:
            print(f"[HW1 expert] IK failed for {name} waypoint {target_T.tolist()}.")
            q = fallback
        return q.astype(np.float32)
