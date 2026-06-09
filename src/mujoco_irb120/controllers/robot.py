import mujoco
from scipy.spatial.transform import Rotation as Robj
import numpy as np
from mujoco_irb120.util.helper_fns import *


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model          = model
        self.data           = data
        self.joint_names    = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.joint_idx      = np.array([model.joint(name).qposadr for name in self.joint_names]).flatten()
        self.joint_dof_idx  = np.array([model.joint(name).dofadr  for name in self.joint_names]).flatten()
        self.stop           = False
        self.q_min          = model.jnt_range[self.joint_idx, 0]
        self.q_max          = model.jnt_range[self.joint_idx, 1]

        self.J              = np.zeros((6, 6))
        self.J_pinv         = np.zeros((6, 6))
        self.T              = np.eye(4)
        self.R_desired      = np.eye(3)
        self.traj_coeffs    = np.zeros((6, 3))
        self.traj_duration  = 0.0
        self.traj_start_time = 0

        self.a_margin       = 1.22 * 0.98
        self.c_margin       = 1.74 * 0.98

        self.error_history  = []
        self.prev_error     = np.inf

        self.ee_site        = model.site('site:tool0').id
        self.table_site     = model.site('site:table').id
        self.obj_frame_site = model.site('site:obj_frame').id
        self.tray_site      = model.site('site:tray_center').id
        self.payload_body_id = int(self.model.site_bodyid[self.obj_frame_site])

        self.f_adr          = self._sensor_address("force_sensor")
        self.t_adr          = self._sensor_address("torque_sensor")
        self.ft_site        = model.site('site:sensor').id
        self.ft_bias_val    = np.zeros(6)
        self.grav_mass       = 0.5 # defined in robot.xml as mass of tray

    def _sensor_address(self, name):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise RuntimeError(
                f"Required sensor {name!r} is missing from the MuJoCo model. "
                "The VLA scene template should define force_sensor and torque_sensor at site:sensor."
            )
        return int(self.model.sensor_adr[sensor_id])

    def FK(self):
        mujoco.mj_forward(self.model, self.data)
        R_curr = self.data.site_xmat[self.ee_site].reshape(3, 3)
        p_curr = self.data.site_xpos[self.ee_site].reshape(3, 1)
        self.T[:3, :3] = R_curr
        self.T[:3, 3] = p_curr.flatten()
        return self.T

    def get_site_pose(self, which='ee'):
        site_map = {
            'ee':     self.ee_site,
            'sensor': self.ft_site,
        }
        if which not in site_map:
            raise ValueError(f"which must be one of {set(site_map)}")
        sid = site_map[which]
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3).copy()
        T[:3,  3] = self.data.site_xpos[sid].copy()
        return T

    def get_jacobian(self, set_pinv=True):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)
        J_pos = jacp[:, self.joint_dof_idx]
        J_rot = jacr[:, self.joint_dof_idx]
        self.J = np.vstack([J_rot, J_pos]).squeeze()
        if set_pinv:
            self.J_pinv = np.linalg.pinv(self.J)
        return self.J

    def ft_bias(self, n_samples=200):
        print(f"Biasing F/T sensor with {n_samples} static samples...")
        samples = []

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        for _ in range(n_samples):
            mujoco.mj_step(self.model, self.data)
            samples.append(self.ft_get_reading(grav_comp=True, apply_bias=False))

        self.ft_bias_val = np.mean(np.asarray(samples, dtype=float), axis=0)
        print(f"Force offset: {np.round(self.ft_bias_val, 3)}")
        return self.ft_bias_val.copy()

    def ft_get_reading(self, grav_comp=True, apply_bias=True, flip_sign=True):
        f_S = np.asarray(self.data.sensordata[self.f_adr:self.f_adr + 3], dtype=float).copy()
        t_S = np.asarray(self.data.sensordata[self.t_adr:self.t_adr + 3], dtype=float).copy()

        if grav_comp:
            R_BS = self.data.site_xmat[self.ft_site].reshape(3, 3)
            g_S = R_BS.T @ np.asarray(self.model.opt.gravity, dtype=float)
            weight_S = self.grav_mass * g_S
            f_S -= weight_S
            r_B = self.data.site_xpos[self.tray_site] - self.data.site_xpos[self.ft_site]
            r_S = R_BS.T @ r_B
            t_S -= np.cross(r_S, weight_S)

        w = np.concatenate([f_S, t_S])
        w *= -1 if flip_sign else 1
        return w - self.ft_bias_val if apply_bias else w

    def set_pose(self, q=np.zeros((6, 1))):
        if not self.is_in_ellipsoid():
            print("Warning: Desired pose is outside the manipulability ellipsoid.")
            return
        self.data.qpos[self.joint_idx] = q
        self.data.qvel[self.joint_dof_idx] = 0.0
        self.set_pos_ctrl(np.asarray(q).flatten(), check_ellipsoid=False)
        mujoco.mj_fwdPosition(self.model, self.data)
        self.error_history = []

    def IK(self, T_des, method=2, max_iters=500, tol=1e-3, damping=0.1, step_size=0.5):
        if T_des.shape != (4, 4):
            raise ValueError("T_des must be a 4x4 homogeneous transform matrix.")

        workspace = self.check_workspace_pose(T_des)
        if not workspace["within_bounds"]:
            print(
                "[Robot.IK] Requested pose is outside the coarse workspace ellipsoid: "
                f"r2={workspace['ellipsoid_r2']:.4f}, pos={np.round(workspace['position'], 4).tolist()}"
            )

        q_original = self.data.qpos[self.joint_idx].copy()
        q = q_original.copy()
        prev_error = np.inf
        xi_e_space = np.full(6, np.nan)
        delta_q = np.full(6, np.nan)
        final_iter = -1

        try:
            for iter_idx in range(max_iters):
                final_iter = iter_idx
                self.data.qpos[self.joint_idx] = q
                mujoco.mj_fwdPosition(self.model, self.data)
                self.FK()

                T_e = np.linalg.inv(self.T) @ T_des
                xi_e = ht2screw(T_e)
                xi_e_space = twistbody2space(xi_e, self.T)
                self.error_history.append(xi_e_space)

                if np.linalg.norm(xi_e_space) < tol:
                    return q

                if np.linalg.norm(xi_e_space) > prev_error:
                    damping *= 0.5
                prev_error = np.linalg.norm(xi_e_space)

                self.get_jacobian()

                if method == 2:
                    J_update = self.J.T @ np.linalg.pinv((self.J @ self.J.T) + (damping**2 * np.eye(6))).real
                elif method == 1:
                    J_update = self.J_pinv.real
                elif method == 3:
                    J_update = damping * self.J.T
                else:
                    raise ValueError("Invalid method. Choose 1, 2, or 3.")

                delta_q = J_update @ xi_e_space
                q += delta_q.flatten()
                q = np.clip(q, self.q_min, self.q_max)

            raise RuntimeError(
                self._format_ik_failure_report(
                    T_des=T_des,
                    q_start=q_original,
                    q_final=q,
                    xi_e_space=xi_e_space,
                    delta_q=delta_q,
                    iter_idx=final_iter,
                    max_iters=max_iters,
                    tol=tol,
                    damping=damping,
                    method=method,
                )
            )

        finally:
            self.data.qpos[self.joint_idx] = q_original
            mujoco.mj_fwdPosition(self.model, self.data)

    def position_only_IK(
        self,
        p_des,
        max_iters=250,
        tol=1e-3,
        damping=0.1,
        step_size=0.5,
        restore_original=True,
    ):
        """Solve IK using only end-effector position error.

        Find joint angles `q` such that `site:tool0` reaches `p_des`, while
        ignoring end-effector orientation.

        Parameters
        ----------
        p_des : array-like, shape (3,)
            Desired world-frame position for `site:tool0`.
        max_iters : int
            Maximum number of IK iterations.
        tol : float
            Stop when Euclidean position error is below this many meters.
        damping : float
            Damped least-squares stabilizer.
        step_size : float
            Scales the joint update each iteration.
        restore_original : bool
            If True, restore `data.qpos` before returning, matching `IK()`.
            If False, leave the robot at the solved pose for debugging.

        Returns
        -------
        np.ndarray, shape (6,)
            Solved joint position target.
        """
        p_des = np.asarray(p_des, dtype=float).reshape(3)
        q_original = self.data.qpos[self.joint_idx].copy()
        q = q_original.copy()
        p_curr = np.full(3, np.nan)
        pos_error = np.full(3, np.nan)
        pos_error_norm = np.inf
        delta_q = np.full(6, np.nan)
        J_pos = np.full((3, 6), np.nan)

        try:
            for iter_idx in range(max_iters):
                self.data.qpos[self.joint_idx] = q
                mujoco.mj_fwdPosition(self.model, self.data)

                p_curr = self.data.site_xpos[self.ee_site].copy()
                pos_error = p_des - p_curr
                pos_error_norm = np.linalg.norm(pos_error)

                if pos_error_norm < tol:
                    return q.copy()

                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)
                J_pos = jacp[:, self.joint_dof_idx] # (3, 6)

                # Damped least squares update
                A = J_pos @ J_pos.T
                A.flat[::4] += damping * damping
                delta_q = J_pos.T @ np.linalg.solve(A, pos_error) # (6,) linalg solve faster than inv
                q += step_size * delta_q
                q = np.clip(q, self.q_min, self.q_max)

            raise RuntimeError(
                self._format_position_ik_failure_report(
                    p_des=p_des,
                    p_final=p_curr,
                    pos_error=pos_error,
                    pos_error_norm=pos_error_norm,
                    q_start=q_original,
                    q_final=q,
                    delta_q=delta_q,
                    J_pos=J_pos,
                    iter_idx=iter_idx,
                    max_iters=max_iters,
                    tol=tol,
                    damping=damping,
                    step_size=step_size,
                )
            )

        finally:
            if restore_original:
                self.data.qpos[self.joint_idx] = q_original
                mujoco.mj_fwdPosition(self.model, self.data)

    def _format_position_ik_failure_report(
        self,
        p_des,
        p_final,
        pos_error,
        pos_error_norm,
        q_start,
        q_final,
        delta_q,
        J_pos,
        iter_idx,
        max_iters,
        tol,
        damping,
        step_size,
    ):
        limit_margin_low = q_final - self.q_min
        limit_margin_high = self.q_max - q_final
        limit_hits = [
            name
            for name, low, high in zip(self.joint_names, limit_margin_low, limit_margin_high)
            if low < 1e-4 or high < 1e-4
        ]

        T_target_position_only = np.eye(4)
        T_target_position_only[:3, 3] = p_des
        target_workspace = self.check_workspace_pose(T_target_position_only)

        T_final_position_only = np.eye(4)
        T_final_position_only[:3, 3] = p_final
        final_workspace = self.check_workspace_pose(T_final_position_only)

        return "\n".join(
            [
                "Position-only IK did not converge.",
                f"  iterations={iter_idx + 1}/{max_iters}, tol={tol:.3e}, damping={damping:.3e}, step_size={step_size:.3f}",
                f"  target_pos={np.round(p_des, 4).tolist()}",
                f"  target_coarse_workspace_r2={target_workspace['ellipsoid_r2']:.4f} ({'inside' if target_workspace['within_bounds'] else 'outside'} coarse bounds)",
                f"  final_pos={np.round(p_final, 4).tolist()}",
                f"  final_coarse_workspace_r2={final_workspace['ellipsoid_r2']:.4f} ({'inside' if final_workspace['within_bounds'] else 'outside'} coarse bounds)",
                f"  pos_error={np.round(pos_error, 4).tolist()}, pos_error_norm={pos_error_norm:.4f} m",
                f"  q_start={np.round(q_start, 4).tolist()}",
                f"  q_final={np.round(q_final, 4).tolist()}",
                f"  q_min={np.round(self.q_min, 4).tolist()}",
                f"  q_max={np.round(self.q_max, 4).tolist()}",
                f"  joints_at_limits={limit_hits if limit_hits else 'none'}",
                f"  last_delta_q={np.round(delta_q, 5).tolist()}, last_delta_q_norm={np.linalg.norm(delta_q):.5f}",
                f"  J_pos_rank={np.linalg.matrix_rank(J_pos)}, J_pos_cond={np.linalg.cond(J_pos):.4e}",
            ]
        )

    def _format_ik_failure_report(
        self,
        T_des,
        q_start,
        q_final,
        xi_e_space,
        delta_q,
        iter_idx,
        max_iters,
        tol,
        damping,
        method,
    ):
        self.data.qpos[self.joint_idx] = q_final
        mujoco.mj_fwdPosition(self.model, self.data)
        T_final = self.FK().copy()

        target_pos = T_des[:3, 3]
        final_pos = T_final[:3, 3]
        start_pos = self._fk_at_q(q_start)[:3, 3]
        pos_err = target_pos - final_pos

        R_err = T_final[:3, :3].T @ T_des[:3, :3]
        rotvec_err = Robj.from_matrix(R_err).as_rotvec()

        limit_margin_low = q_final - self.q_min
        limit_margin_high = self.q_max - q_final
        limit_hits = [
            name
            for name, low, high in zip(self.joint_names, limit_margin_low, limit_margin_high)
            if low < 1e-4 or high < 1e-4
        ]

        p = final_pos.flatten()
        target_workspace = self.check_workspace_pose(T_des)
        final_workspace = self.check_workspace_pose(T_final)

        return "\n".join(
            [
                "IK did not converge.",
                f"  method={method}, iterations={iter_idx + 1}/{max_iters}, tol={tol:.3e}, final_damping={damping:.3e}",
                f"  target_pos={np.round(target_pos, 4).tolist()}",
                f"  target_coarse_workspace_r2={target_workspace['ellipsoid_r2']:.4f} ({'inside' if target_workspace['within_bounds'] else 'outside'} coarse bounds)",
                f"  start_pos={np.round(start_pos, 4).tolist()}",
                f"  final_pos={np.round(final_pos, 4).tolist()}",
                f"  pos_error={np.round(pos_err, 4).tolist()}, pos_error_norm={np.linalg.norm(pos_err):.4f} m",
                f"  rotvec_error={np.round(rotvec_err, 4).tolist()}, rot_error_norm={np.linalg.norm(rotvec_err):.4f} rad",
                f"  twist_error={np.round(xi_e_space, 4).tolist()}, twist_error_norm={np.linalg.norm(xi_e_space):.4f}",
                f"  last_delta_q={np.round(delta_q, 5).tolist()}, last_delta_q_norm={np.linalg.norm(delta_q):.5f}",
                f"  q_start={np.round(q_start, 4).tolist()}",
                f"  q_final={np.round(q_final, 4).tolist()}",
                f"  q_min={np.round(self.q_min, 4).tolist()}",
                f"  q_max={np.round(self.q_max, 4).tolist()}",
                f"  joints_at_limits={limit_hits if limit_hits else 'none'}",
                f"  final_coarse_workspace_r2={final_workspace['ellipsoid_r2']:.4f} ({'inside' if final_workspace['within_bounds'] else 'outside'} coarse bounds)",
            ]
        )

    def _fk_at_q(self, q):
        q_current = self.data.qpos[self.joint_idx].copy()
        self.data.qpos[self.joint_idx] = q
        mujoco.mj_fwdPosition(self.model, self.data)
        T = self.FK().copy()
        self.data.qpos[self.joint_idx] = q_current
        mujoco.mj_fwdPosition(self.model, self.data)
        return T

    def set_pos_ctrl(self, q_desired, check_ellipsoid=True):
        raise NotImplementedError

    def get_surface_pos(self):
        mujoco.mj_forward(self.model, self.data)
        tab_global_pos = self.data.site_xpos[self.table_site].flatten()
        tab_dims = self.model.geom('table').size
        surface_pos = tab_global_pos + np.array([0, 0, tab_dims[2]]).flatten()
        return surface_pos

    def is_in_ellipsoid(self):
        self.FK()
        workspace = self.check_workspace_pose(self.T)
        if not workspace["within_bounds"]:
            print('Robot is outside the manipulability ellipsoid.')
            self.stop = True
            return False
        return True

    def check_workspace_pose(self, pose) -> dict:
        """Check whether a pose is inside the coarse geometric workspace envelope.

        This is not a joint-limit-aware reachability test. It is a fast first
        filter that says whether a Cartesian point lies inside the broad
        ellipsoid used elsewhere in this project. IK can still fail inside this
        envelope because of joint limits, singularities, current seed, or
        orientation constraints.
        """
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("pose must be a 4x4 homogeneous transform matrix.")
        p = pose[:3, 3].flatten()
        r2 = ((p[0] ** 2 + p[1] ** 2) / self.a_margin**2) + (p[2] ** 2 / self.c_margin**2)
        return {
            "within_bounds": bool(r2 <= 1.0),
            "ellipsoid_r2": float(r2),
            "position": p.copy(),
            "a_margin": float(self.a_margin),
            "c_margin": float(self.c_margin),
        }

    def get_payload_pose(self, payload_site='site:obj_frame', out='T', degrees=False, frame='world'):
        sid = self.model.site(payload_site).id
        Rw = self.data.site_xmat[sid].reshape(3, 3)
        p = self.data.site_xpos[sid].flatten()

        if out == 'T':
            T_payload = np.eye(4)
            T_payload[:3, :3] = Rw
            T_payload[:3, 3] = p
            return T_payload

        if out == 'R':
            return Rw

        if out == 'p':
            return p

        if out == 'quat':
            return Robj.from_matrix(Rw).as_quat()

        if out == 'rpy':
            R_eval = np.eye(3) if frame == 'body' else Rw
            return Robj.from_matrix(R_eval).as_euler('xyz', degrees=degrees)

        raise ValueError("Output must be one of 'T', 'R', 'p', 'rpy', 'quat'.")


class PositionController(Robot):
    def set_pos_ctrl(self, q_desired, check_ellipsoid=True):
        if check_ellipsoid and not self.is_in_ellipsoid():
            return
        self.data.ctrl[:] = np.asarray(q_desired).flatten()
        mujoco.mj_forward(self.model, self.data)
