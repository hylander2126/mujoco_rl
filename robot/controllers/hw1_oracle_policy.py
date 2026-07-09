from __future__ import annotations

import mujoco
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
        self.start_T = env.irb.FK().copy()
        self.start_q = env.data.qpos[env.irb.joint_idx].copy().astype(float)
        self.start_tray_normal = env.data.site_xmat[env._tray_site_id].reshape(3, 3)[:, 2].copy()
        print(f"[HW1 expert] Home pose (cartesian): {np.round(self.start_T[:3, 3], 2)}")

        self.pre_drop_xyz = self._pre_drop_xyz()

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

    def _pre_drop_xyz(self) -> np.ndarray:
        if not getattr(self.env, "randomize_bins", False):
            return self.task.pre_drop_xyz_by_color[self.cube_color]

        bin_xy = np.asarray(self.task.bin_xy_by_color[self.cube_color], dtype=float).reshape(2)
        radius = float(np.linalg.norm(bin_xy))
        if radius < 1e-6:
            return self.task.pre_drop_xyz_by_color[self.cube_color]

        inward_offset = 0.23
        pre_radius = max(0.18, radius - inward_offset)
        pre_xy = bin_xy * (pre_radius / radius)
        return np.array([pre_xy[0], pre_xy[1], 0.55], dtype=float)

    def _solve_ik_waypoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve Cartesian waypoints into joint targets once per episode."""
        print(f"[HW1 expert] cube_color={self.cube_color}; solving tray-align IK once...")

        # Keep the tray level (same normal as home) while moving to the
        # pre-drop position, instead of only constraining position.
        pre_drop_q = self._solve_tray_align_ik_multistart(self.pre_drop_xyz, self.start_tray_normal)

        drop_q = self._apply_tip_offset(pre_drop_q)
        print("[HW1 expert] Tray-align IK waypoints ready.")
        return pre_drop_q, drop_q

    def _solve_tray_align_ik_multistart(
        self,
        target_xyz: np.ndarray,
        target_normal: np.ndarray,
        seeds: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        q0 = self.env.data.qpos[self.env.irb.joint_idx].copy()
        qvel0 = self.env.data.qvel.copy()
        ctrl0 = self.env.data.ctrl.copy()

        seeds = self._ik_seed_candidates(target_xyz) if seeds is None else seeds
        best_seed_q: np.ndarray | None = None
        best_err_norm = np.inf
        try:
            for seed_q in seeds:
                self.env.data.qpos[self.env.irb.joint_idx] = np.clip(seed_q, self.env.irb.q_min, self.env.irb.q_max)
                self.env.data.qvel[:] = 0.0
                mujoco.mj_forward(self.env.model, self.env.data)
                try:
                    return self.env.irb.tray_align_IK(
                        target_xyz,
                        target_normal,
                        damping=0.7,
                        step_size=0.5,
                        max_iters=600,
                        tol=1e-2,
                    ).astype(np.float32)
                except RuntimeError as exc:
                    err_norm = float(str(exc).split("err_norm=")[1].split(" ")[0])
                    if err_norm < best_err_norm:
                        best_err_norm = err_norm
                        best_seed_q = seed_q

            # No seed reached tight convergence (e.g. the target sits at the
            # edge of the reachable set while staying level). Fall back to
            # the closest solution rather than crashing the episode.
            self.env.data.qpos[self.env.irb.joint_idx] = np.clip(best_seed_q, self.env.irb.q_min, self.env.irb.q_max)
            self.env.data.qvel[:] = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            q = self.env.irb.tray_align_IK(
                target_xyz,
                target_normal,
                damping=0.7,
                step_size=0.5,
                max_iters=600,
                tol=1e-2,
                best_effort=True,
            ).astype(np.float32)
            print(
                f"[HW1 expert] WARNING: tray-align IK best-effort fallback for target "
                f"{np.round(target_xyz, 3)} (residual err_norm={best_err_norm:.4f})"
            )
            return q
        finally:
            self.env.data.qpos[self.env.irb.joint_idx] = q0
            self.env.data.qvel[:] = qvel0
            self.env.data.ctrl[:] = ctrl0
            mujoco.mj_forward(self.env.model, self.env.data)

    def _ik_seed_candidates(self, target_xyz: np.ndarray) -> list[np.ndarray]:
        q_min = self.env.irb.q_min
        q_max = self.env.irb.q_max
        q_mid = 0.5 * (q_min + q_max)
        base_angle = float(np.arctan2(target_xyz[1], target_xyz[0]))

        candidates = []
        for q in (self.start_q, self.q_home, q_mid):
            seed = np.asarray(q, dtype=float).copy()
            seed[0] = base_angle
            candidates.append(seed)

        shoulder_elbow = [
            (-0.8, 0.9),
            (-0.4, 0.7),
            (0.0, 0.5),
            (0.4, 0.3),
            (-1.1, 1.2),
        ]
        for shoulder, elbow in shoulder_elbow:
            seed = self.q_home.copy().astype(float)
            seed[0] = base_angle
            seed[1] = shoulder
            seed[2] = elbow
            candidates.append(seed)

        rng = np.random.default_rng(0)
        for _ in range(8):
            seed = rng.uniform(q_min, q_max)
            seed[0] = base_angle + rng.uniform(-0.4, 0.4)
            candidates.append(seed)

        return candidates

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

        # Try the pre-drop config first (the tip is a small rotation away,
        # so this converges fast) and fall back to the full seed sweep.
        seeds = [pre_drop_q] + self._ik_seed_candidates(self.pre_drop_xyz)
        drop_q = self._solve_tray_align_ik_multistart(self.pre_drop_xyz, tip_normal, seeds=seeds)
        return drop_q.astype(np.float32)
