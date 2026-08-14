"""Press-and-pull (squash / arc / unarc) state machine for the MuJoCo IRB120.

This is a simulation port of the real-robot controller
`irb120_ws/src/irb120_ros2/irb120_control/irb120_control/arc_static.py`
(with the escalating-force retry from `adaptive_press.py` folded in as an
option). It reproduces the same phase sequence, the same exit conditions, and
the same force regulation, so rollouts recorded here are directly comparable to
hardware logs.

    SQUASH -> LULL -> ARC -> LULL -> UNARC -> RETRACT -> DONE

  SQUASH  descend at a fixed speed until the measured normal force reaches the
          reference, then hand off.
  LULL    hold still briefly so the contact settles. Also the junction that
          decides whether the next phase is ARC or UNARC.
  ARC     sweep the contact point along a circular arc about the object's
          tipping edge while a PI controller regulates the radial (into-object)
          force. Exits when the tangential force dies away or flips sign --
          both of which mean the object has gone over its balance point -- or
          at a hard angle cap as a safety fallback.
  UNARC   reverse the sweep back to the starting angle, easing the object down.
  RETRACT lift straight up, away from the object.

Why the tangential-force exit matters: the point where tangential force crosses
zero *is* theta*, the object's balance angle, which is what the parameter
estimator ultimately wants. The controller stops there rather than pushing the
object all the way over.

Differences from the hardware version, and why
----------------------------------------------
1. Pose source. Hardware tracks the TF frame `finger_ball_center`; here we read
   the `site:ball_center` site. NOT the site behind `Robot.FK()`, which is
   `site:fingertip` and sits ~0.18 m back along the rod -- using it would put
   the arc center in the wrong place by that distance.
2. Force frame. Hardware subscribes to `/netft_data_transformed`, already
   rotated into the world frame. `controller.ft_get_reading()` returns the
   wrench in the *sensor* frame, so we rotate it with the sensor site's
   orientation before doing any radial/tangential projection.
3. Commanding. Hardware publishes twists to MoveIt Servo. Here we integrate
   Cartesian velocity into joint position targets via
   `apply_cartesian_keyboard_ctrl`, because the MuJoCo actuators are
   position-controlled -- feeding velocities to `set_vel_ctrl` would be
   interpreted as joint angles.
4. Arc center. Hardware hard-codes `ARC_CENTER = (0.61, 0, 0)` for its table
   setup. Here the pivot is read from the object's `site:obj_frame`, which sits
   on the tipping edge and is the same frame `com_gt` is measured from
   in `object_params.json`. That keeps the arc geometry and the ground-truth
   CoM in one consistent frame.
5. Approach. Hardware plans to a stored `pre_squash` pose with MoveIt. Here we
   solve IK and set the joint configuration directly, since the sim can be
   placed exactly and the approach motion is not part of the experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from parameter_estimation.controllers.force_controller import PIDForceController
from parameter_estimation.controllers.motion_geometry import (
    arc_angle_xz,
    arc_velocity_xz,
    clamp,
    radial_force_xz,
    tangent_force_xz,
)

# Shared with the hardware logs and with estimate_params.py. Do not renumber:
# downstream phase segmentation keys off these integers.
STATE_IDS = {
    "SQUASH": 1,
    "LULL": 2,
    "ARC": 3,
    "UNARC": 4,
    "RETRACT": 5,
    "DONE": 6,
}


@dataclass
class PressPullConfig:
    """Tunables, defaulted to the hardware values in arc_static.py.

    Speeds are in m/s at real-robot scale. `speed_scale` multiplies all of them
    at once: a full rollout at hardware speed is ~60 s of sim time, which is
    fine headless but slow to watch. Scaling up trades physical fidelity for
    wall-clock -- contact transients get less time to settle -- so keep it at
    1.0 for anything you intend to fit parameters to.
    """

    # --- force regulation -------------------------------------------------
    force_ref_n: float = 5.0
    force_hard_limit_n: float = 15.0
    contact_detect_n: float = 0.25
    contact_stable_samples: int = 1

    kp_force: float = 0.00035
    ki_force: float = 0.000005
    kd_force: float = 0.0
    max_normal_speed: float = 0.006
    force_deadband_n: float = 0.25
    force_filter_alpha: float = 0.12
    force_output_slew_rate: float = 0.02

    unarc_force_augment_speed: float = 0.004
    unarc_force_augment_softness_n: float = 0.75

    # --- speeds -----------------------------------------------------------
    descend_speed: float = 0.005
    arc_tangential_speed: float = 0.008
    retract_speed: float = 0.008
    arc_tangential_ramp_sec: float = 2.0
    speed_scale: float = 1.0

    # --- arc geometry and exit conditions ---------------------------------
    arc_max_angle_deg: float = -23.0
    arc_center_xz: tuple[float, float] | None = None  # None -> read site:obj_frame
    arc_fx_sign_deadband_n: float = 0.08
    arc_fx_sign_min_sweep_deg: float = 5.0
    arc_fx_sign_min_samples: int = 20
    arc_fx_flip_stable_samples: int = 5
    arc_fx_low_thresh_n: float = 0.1
    arc_fx_low_stable_samples: int = 5

    # --- timeouts ---------------------------------------------------------
    squash_timeout_sec: float = 30.0
    arc_timeout_sec: float = 30.0
    unarc_timeout_sec: float = 30.0
    lull_wait_sec: float = 1.0
    retract_duration_sec: float = 3.0

    # --- lost contact -----------------------------------------------------
    lost_contact_force_thresh_n: float = 0.3
    lost_contact_steps: int = 20

    # Minimum object rotation for an ARC to count as a real tip. Without this,
    # a finger that slips and slides across the object's top face looks like a
    # success: tangential force collapses exactly as it does at the true
    # balance point, so the ARC exit test cannot tell the two apart from force
    # alone. Hardware resolves the same ambiguity with its vision stream
    # (`arc_static._on_detection`, which publishes object pitch); in sim the
    # equivalent signal is the object's ground-truth pose. This is outcome
    # labelling only -- it never enters the control law.
    min_tip_angle_deg: float = 2.0

    # Rate the sample-count thresholds above were tuned at. Every `*_samples` /
    # `*_steps` value is a count of control ticks, not a duration, so running
    # the sim at its native 1000 Hz would make each debounce 10x shorter than
    # on hardware -- short enough that a stick-slip transient reads as the
    # object reaching its balance point and ARC exits almost immediately.
    # PressPullFSM rescales those counts by (actual rate / this).
    hardware_control_hz: float = 100.0

    # --- approach ---------------------------------------------------------
    approach_clearance_m: float = 0.02
    # Where on the top face to press, relative to its centre. Not cosmetic: the
    # press point's distance from the tipping edge is a lever arm in the
    # restoring moment (W*com_offset + N*press_offset), so pressing nearer the
    # pivot makes the object tip at a lower force. For the box, moving from the
    # centre (0.05 m out) to 0.02 m out drops the tangential force needed at
    # 5 N from ~1.84 N to ~1.36 N -- the difference between slipping and
    # tipping. Too near the edge and the finger slides off it instead.
    press_offset_xy: tuple[float, float] = (0.0, 0.0)

    # --- adaptive retry (from adaptive_press.py) --------------------------
    # When the finger slips off mid-ARC, retry the whole sequence with a
    # higher squash force instead of giving up. The force ladder is itself
    # informative: how much normal force it took to keep contact through the
    # tip is a cheap proxy for the object's mass and CoM height, and it is
    # available before the estimator can fit anything.
    adaptive_retry: bool = False
    force_scale_factor: float = 1.25
    force_ref_max_n: float = 13.0

    verbose: bool = True


class PressPullFSM:
    """Drives one press-and-pull rollout. Call `step()` once per sim step."""

    def __init__(self, irb, model, data, config: PressPullConfig | None = None):
        self.irb = irb
        self.model = model
        self.data = data
        self.cfg = config or PressPullConfig()

        # The hardware loop runs at a fixed 100 Hz; here the control rate is the
        # physics rate. Passing the true rate keeps the PID's integral and
        # derivative terms equivalent to the tuned hardware behaviour.
        self.control_hz = 1.0 / float(model.opt.timestep)

        self._force_ctrl = PIDForceController(
            kp=self.cfg.kp_force,
            ki=self.cfg.ki_force,
            kd=self.cfg.kd_force,
            force_ref_n=self.cfg.force_ref_n,
            max_normal_speed=self.cfg.max_normal_speed,
            control_hz=self.control_hz,
            deadband_n=self.cfg.force_deadband_n,
            measurement_filter_alpha=self.cfg.force_filter_alpha,
            output_slew_rate=self.cfg.force_output_slew_rate,
        )

        # Debounce counts are tick counts tuned at `hardware_control_hz`. Convert
        # them to the same wall-clock durations at this loop's actual rate.
        self._tick_scale = max(1.0, self.control_hz / self.cfg.hardware_control_hz)
        self._n_contact_stable = self._ticks(self.cfg.contact_stable_samples)
        self._n_lost_contact = self._ticks(self.cfg.lost_contact_steps)
        self._n_fx_sign_min = self._ticks(self.cfg.arc_fx_sign_min_samples)
        self._n_fx_flip_stable = self._ticks(self.cfg.arc_fx_flip_stable_samples)
        self._n_fx_low_stable = self._ticks(self.cfg.arc_fx_low_stable_samples)
        if self.cfg.verbose and self._tick_scale > 1.0:
            print(f"[fsm] control rate {self.control_hz:.0f} Hz vs tuned "
                  f"{self.cfg.hardware_control_hz:.0f} Hz -> debounce counts scaled "
                  f"{self._tick_scale:.0f}x "
                  f"(lost-contact {self.cfg.lost_contact_steps}->{self._n_lost_contact} ticks)")

        self._ft_site = model.site("site:sensor").id
        self._reset_run_state()

        # Rollout log. Names match shove_simulation.py's npz keys so the
        # existing notebooks can load these rollouts unchanged; the arc/state
        # channels are additions the hardware estimator needs for phase
        # segmentation.
        self.log: dict[str, list] = {
            "t_hist": [], "w_hist": [], "w_world_hist": [], "quat_hist": [],
            "ball_pose_hist": [], "sens_pose_hist": [], "obj_pose_hist": [],
            "con_bool_hist": [], "state_id_hist": [], "arc_angle_hist": [],
            "f_radial_hist": [], "f_tangent_hist": [], "force_ref_hist": [],
        }
        self.attempts: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def _ticks(self, hardware_samples: int) -> int:
        """Convert a hardware tick count into this loop's equivalent count."""
        return max(1, int(round(hardware_samples * self._tick_scale)))

    def _reset_run_state(self) -> None:
        self.state = "SQUASH"
        self.done = False
        self.completed = False          # True only if UNARC finished cleanly
        self.abort_reason: str | None = None
        self._state_start_time = float(self.data.time)
        self._contact_count = 0
        self._contact_felt = False
        self._lost_contact_count = 0
        self._lull_next = "ARC"
        self._arc_center_x: float | None = None
        self._arc_center_z: float | None = None
        self._arc_start_angle: float | None = None
        self._arc_end_angle = math.radians(self.cfg.arc_max_angle_deg)
        self._arc_fx_pos_count = 0
        self._arc_fx_neg_count = 0
        self._arc_fx_flip_count = 0
        self._arc_fx_majority_sign: int | None = None
        self._arc_fx_low_count = 0
        self._force_y_ref: float | None = None
        self.tipped = False
        self.max_tip_deg = 0.0
        self.arc_exit_angle_rad = float("nan")
        self.arc_exit_reason: str | None = None
        self._obj_rot0: np.ndarray | None = None

    def object_tip_angle_deg(self) -> float:
        """Object rotation away from its pose at ARC onset, in degrees.

        Total rotation magnitude rather than a single Euler axis, so it works
        for objects that tip about something other than +Y. Outcome labelling
        only -- see `PressPullConfig.min_tip_angle_deg`.
        """
        Rw = self.irb.get_payload_pose(out="R")
        if self._obj_rot0 is None:
            return 0.0
        dR = self._obj_rot0.T @ Rw
        cos_theta = (np.trace(dR) - 1.0) / 2.0
        return math.degrees(math.acos(float(np.clip(cos_theta, -1.0, 1.0))))

    def object_top_center(self) -> np.ndarray:
        """World-frame centre of the payload's top face, from its geom AABBs.

        `model.geom_aabb` is a local axis-aligned box per geom (centre + half
        extents), which is exact for primitive box geoms and a tight fit for the
        meshed objects. Good enough to aim the descent -- SQUASH finds the real
        surface by force, not by this estimate.
        """
        mujoco.mj_forward(self.model, self.data)
        pid = self.irb.payload_body_id
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] != pid:
                continue
            aabb = self.model.geom_aabb[gid].reshape(2, 3)
            centre_local, half = aabb[0], aabb[1]
            R = self.data.geom_xmat[gid].reshape(3, 3)
            centre_world = self.data.geom_xpos[gid] + R @ centre_local
            # Rotate the box's half-extents into world axes.
            half_world = np.abs(R) @ half
            lo = np.minimum(lo, centre_world - half_world)
            hi = np.maximum(hi, centre_world + half_world)
        return np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, hi[2]])

    def pivot_xz(self) -> tuple[float, float]:
        """Arc centre: the object's tipping edge.

        Read from `site:obj_frame`, which the object XMLs place on the bottom
        edge nearest the robot -- the edge the object rotates about when the
        finger drags its top toward -X. This is the same origin that
        `object_params.json`'s `com_gt` is expressed in, so the arc and
        the ground-truth CoM share a frame.
        """
        if self.cfg.arc_center_xz is not None:
            return self.cfg.arc_center_xz
        p = self.irb.get_payload_pose(site="site:obj_frame", out="p")
        return float(p[0]), float(p[2])

    def move_to_pre_squash(self) -> np.ndarray:
        """Place the ball contact point just above the object's top face."""
        top = self.object_top_center()
        target_ball = np.array([
            top[0] + self.cfg.press_offset_xy[0],
            top[1] + self.cfg.press_offset_xy[1],
            top[2] + self.cfg.approach_clearance_m,
        ])

        # IK solves for the fingertip site, but we want to place the ball. With
        # orientation held fixed the two are a constant offset apart, so solve
        # for the fingertip pose that puts the ball where we want it.
        T_home = self.irb.FK().copy()
        ball_offset = self.irb.get_site_pose("ball")[:3, 3] - T_home[:3, 3]

        T_target = T_home.copy()
        T_target[:3, 3] = target_ball - ball_offset
        q = self.irb.IK(T_target, method=2, damping=0.5, max_iters=1000)
        self.irb.set_pose(q=q)

        if self.cfg.verbose:
            reached = self.irb.get_site_pose("ball")[:3, 3]
            err = np.linalg.norm(reached - target_ball)
            print(f"[pre-squash] object top {np.round(top, 4)}  "
                  f"ball target {np.round(target_ball, 4)}  "
                  f"reached {np.round(reached, 4)}  err {err * 1000:.1f} mm")
        return q

    # ------------------------------------------------------------------ #
    #  Sensing
    # ------------------------------------------------------------------ #

    def _world_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (wrench_sensor_frame, wrench_world_frame), each [f(3), t(3)].

        `ft_get_reading` reports in the sensor frame; the arc's radial and
        tangential projections are defined in world XZ, so rotate first. The
        hardware controller gets this for free from `/netft_data_transformed`.
        """
        w_S = np.asarray(self.irb.ft_get_reading(grav_comp=True, apply_bias=True), dtype=float)
        R_BS = self.data.site_xmat[self._ft_site].reshape(3, 3)
        w_B = np.concatenate([R_BS @ w_S[:3], R_BS @ w_S[3:]])
        return w_S, w_B

    def _ball_xz(self) -> tuple[float, float]:
        p = self.irb.get_site_pose("ball")[:3, 3]
        return float(p[0]), float(p[2])

    # ------------------------------------------------------------------ #
    #  Arc helpers
    # ------------------------------------------------------------------ #

    def _init_arc(self, x_contact: float, z_contact: float) -> None:
        cx, cz = self.pivot_xz()
        self._arc_center_x, self._arc_center_z = cx, cz
        self._arc_start_angle = arc_angle_xz(x_contact, z_contact, cx, cz)
        self._arc_end_angle = math.radians(self.cfg.arc_max_angle_deg)
        radius = math.hypot(x_contact - cx, z_contact - cz)
        self._arc_fx_pos_count = 0
        self._arc_fx_neg_count = 0
        self._arc_fx_flip_count = 0
        self._arc_fx_majority_sign = None
        self._arc_fx_low_count = 0
        if self.cfg.verbose:
            print(f"[arc init] centre=({cx:.4f}, {cz:.4f})  r={radius:.4f} m  "
                  f"start={math.degrees(self._arc_start_angle):.1f} deg  "
                  f"end={self.cfg.arc_max_angle_deg:.1f} deg")

    def _current_arc_angle(self, x: float, z: float) -> float:
        return arc_angle_xz(x, z, self._arc_center_x, self._arc_center_z)

    def _arc_fx_flipped(self, angle: float, f_tangent: float) -> bool:
        """True once tangential force crosses the sign it held through the sweep.

        A majority sign is locked in after enough samples so that early contact
        transients cannot be mistaken for the real crossing.
        """
        if self._arc_start_angle is None:
            return False
        if abs(self._arc_start_angle - angle) < math.radians(self.cfg.arc_fx_sign_min_sweep_deg):
            return False
        if abs(f_tangent) < self.cfg.arc_fx_sign_deadband_n:
            self._arc_fx_flip_count = 0
            return False

        sign = 1 if f_tangent > 0.0 else -1
        if sign > 0:
            self._arc_fx_pos_count += 1
        else:
            self._arc_fx_neg_count += 1

        total = self._arc_fx_pos_count + self._arc_fx_neg_count
        if self._arc_fx_majority_sign is None and total >= self._n_fx_sign_min:
            self._arc_fx_majority_sign = 1 if self._arc_fx_pos_count >= self._arc_fx_neg_count else -1
            if self.cfg.verbose:
                print(f"[arc] tangent-force majority sign locked: "
                      f"{'+' if self._arc_fx_majority_sign > 0 else '-'} "
                      f"(pos={self._arc_fx_pos_count}, neg={self._arc_fx_neg_count})")

        if self._arc_fx_majority_sign is None or sign == self._arc_fx_majority_sign:
            self._arc_fx_flip_count = 0
            return False

        self._arc_fx_flip_count += 1
        return self._arc_fx_flip_count >= self._n_fx_flip_stable

    def _check_lost_contact(self, force: float) -> bool:
        if abs(force) < self.cfg.lost_contact_force_thresh_n:
            self._lost_contact_count += 1
        else:
            self._lost_contact_count = 0
        return self._lost_contact_count >= self._n_lost_contact

    def _transition(self, state: str) -> None:
        if state != self.state:
            if self.cfg.verbose:
                print(f"[fsm] {self.state} -> {state}  (t={self.data.time:.2f}s)")
            self.state = state
            self._state_start_time = float(self.data.time)

    def _elapsed(self) -> float:
        return float(self.data.time) - self._state_start_time

    def _timed_out(self, limit: float, label: str) -> bool:
        if self._elapsed() > limit:
            self._abort(f"{label} timed out after {limit:.1f}s")
            return True
        return False

    def _abort(self, reason: str) -> None:
        if self.cfg.verbose:
            print(f"[fsm] ABORT: {reason}")
        self.abort_reason = reason
        self._transition("RETRACT")

    def _command(self, vx: float, vy: float, vz: float) -> None:
        # v_cmd is [wx, wy, wz, vx, vy, vz]; maintain_orientation zeroes the
        # angular half, which is what makes the ball and fingertip share one
        # linear velocity.
        v = np.array([0.0, 0.0, 0.0, vx, vy, vz], dtype=float)
        self.irb.apply_cartesian_keyboard_ctrl(v, maintain_orientation=True, verbose=False)

    # ------------------------------------------------------------------ #
    #  Main tick
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        """Advance the controller one tick. Call before `mujoco.mj_step`."""
        if self.done:
            self._command(0.0, 0.0, 0.0)
            return

        cfg = self.cfg
        scale = cfg.speed_scale
        x, z = self._ball_xz()
        w_S, w_B = self._world_wrench()
        fx, fy, fz_signed = w_B[0], w_B[1], w_B[2]
        fz = abs(fz_signed)

        angle = (self._current_arc_angle(x, z)
                 if self._arc_center_x is not None else float("nan"))
        f_radial = (radial_force_xz(angle, fx, fz_signed)
                    if self._arc_center_x is not None else fz)
        f_tangent = (tangent_force_xz(angle, fx, fz_signed)
                     if self._arc_center_x is not None else 0.0)

        self._record(w_S, w_B, angle, f_radial, f_tangent)

        # Force-dependent safety uses the radial component once the arc frame
        # exists, matching the hardware's _controlled_contact_force.
        contact_force = f_radial if self.state in ("ARC", "UNARC") else fz
        if contact_force > cfg.force_hard_limit_n and self.state != "RETRACT":
            self._abort(f"hard force limit exceeded: {contact_force:.2f} N in {self.state}")
            return

        # -------- SQUASH --------
        if self.state == "SQUASH":
            if self._timed_out(cfg.squash_timeout_sec, "SQUASH"):
                return
            if fz > cfg.contact_detect_n and not self._contact_felt:
                self._contact_felt = True
                self._force_y_ref = fy
                if cfg.verbose:
                    print(f"[fsm] contact felt at t={self.data.time:.2f}s  fz={fz:.2f} N")
            self._command(0.0, 0.0, -cfg.descend_speed * scale)
            if fz >= self._force_ctrl.reference:
                self._contact_count += 1
                if self._contact_count >= self._n_contact_stable:
                    self._transition("LULL")
            else:
                self._contact_count = 0
            return

        # -------- LULL --------
        if self.state == "LULL":
            self._command(0.0, 0.0, 0.0)
            if self._elapsed() < cfg.lull_wait_sec:
                return
            if self._lull_next == "ARC":
                self._force_ctrl.reset()
                # Cap at the configured reference so ARC never starts near the
                # hard limit after an overshoot during descent.
                pull_ref = min(fz, self._force_ctrl.reference)
                self._force_ctrl.set_reference(pull_ref)
                if cfg.verbose:
                    print(f"[fsm] ARC setpoint {pull_ref:.2f} N (measured {fz:.2f} N)")
                self._init_arc(x, z)
                # Reference pose for tip detection: whatever settling happened
                # during SQUASH is not tipping, so measure rotation from here.
                self._obj_rot0 = self.irb.get_payload_pose(out="R").copy()
            self._transition(self._lull_next)
            return

        # -------- ARC --------
        if self.state == "ARC":
            if self._timed_out(cfg.arc_timeout_sec, "ARC"):
                return
            if self._check_lost_contact(f_radial):
                self._abort("lost contact during ARC")
                return
            self.max_tip_deg = max(self.max_tip_deg, self.object_tip_angle_deg())
            self._arc_step(angle, f_radial, cfg.arc_tangential_speed * scale)

            swept = abs(self._arc_start_angle - angle)
            if (swept >= math.radians(cfg.arc_fx_sign_min_sweep_deg)
                    and f_tangent < cfg.arc_fx_low_thresh_n):
                self._arc_fx_low_count += 1
                if self._arc_fx_low_count >= self._n_fx_low_stable:
                    self._finish_arc(f"tangent force below {cfg.arc_fx_low_thresh_n:.2f} N", angle, f_tangent)
                    return
            else:
                self._arc_fx_low_count = 0

            if self._arc_fx_flipped(angle, f_tangent):
                self._finish_arc("tangent force sign flip", angle, f_tangent)
                return
            if angle <= self._arc_end_angle:
                self._finish_arc(f"max arc angle {cfg.arc_max_angle_deg:.1f} deg (safety fallback)",
                                 angle, f_tangent)
            return

        # -------- UNARC --------
        if self.state == "UNARC":
            if self._timed_out(cfg.unarc_timeout_sec, "UNARC"):
                return
            self._arc_step(angle, f_radial, -cfg.arc_tangential_speed * scale)
            if angle >= self._arc_start_angle - math.radians(1.0):
                self.completed = True
                self._transition("RETRACT")
            return

        # -------- RETRACT --------
        if self.state == "RETRACT":
            if self._elapsed() < cfg.retract_duration_sec:
                self._command(0.0, 0.0, cfg.retract_speed * scale)
            else:
                self._command(0.0, 0.0, 0.0)
                self._transition("DONE")
                self.done = True
            return

    def _finish_arc(self, reason: str, angle: float, f_tangent: float) -> None:
        """Leave ARC via LULL, then reverse.

        If the object really tipped, the arc angle recorded here is where the
        tangential force vanished -- the object's balance point, and the single
        most useful number this rollout produces. If it did not tip, the same
        force signature means the finger slipped, and the angle is meaningless.
        `tipped` is what separates the two; do not fit theta* from a rollout
        where it is False.
        """
        self.max_tip_deg = max(self.max_tip_deg, self.object_tip_angle_deg())
        self.tipped = self.max_tip_deg >= self.cfg.min_tip_angle_deg
        if self.cfg.verbose:
            verdict = (f"object rotated {self.max_tip_deg:.2f} deg -- TIPPED"
                       if self.tipped else
                       f"object rotated only {self.max_tip_deg:.2f} deg "
                       f"(< {self.cfg.min_tip_angle_deg:.1f}) -- SLIPPED, not a tip")
            print(f"[fsm] ARC exit ({reason}): arc_angle={math.degrees(angle):.2f} deg  "
                  f"f_tangent={f_tangent:.3f} N")
            print(f"[fsm]   {verdict}")
        self.arc_exit_angle_rad = angle
        self.arc_exit_reason = reason
        self._lull_next = "UNARC"
        self._transition("LULL")

    def _arc_step(self, angle: float, f_radial: float, tangential_speed: float) -> None:
        cfg = self.cfg
        radial_corr = -self._force_ctrl.update(f_radial)

        if self.state == "UNARC":
            # On the way back down, gravity helps the object away from the
            # finger, so bias inward whenever the measured force falls short.
            deficit = max(0.0, self._force_ctrl.reference - f_radial)
            augment = deficit / (deficit + cfg.unarc_force_augment_softness_n) if deficit > 0.0 else 0.0
            radial_corr = clamp(radial_corr - cfg.unarc_force_augment_speed * augment,
                                cfg.max_normal_speed)

        ramp_sec = cfg.arc_tangential_ramp_sec
        ramp = min(1.0, max(0.0, self._elapsed() / ramp_sec)) if ramp_sec > 0 else 1.0
        vx, vz = arc_velocity_xz(angle, tangential_speed * ramp, radial_corr)
        self._command(vx, 0.0, vz)

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #

    def _record(self, w_S, w_B, angle, f_radial, f_tangent) -> None:
        L = self.log
        L["t_hist"].append(float(self.data.time))
        L["w_hist"].append(w_S)
        L["w_world_hist"].append(w_B)
        L["quat_hist"].append(self.irb.get_payload_pose(out="quat"))
        L["ball_pose_hist"].append(self.irb.get_site_pose("ball"))
        L["sens_pose_hist"].append(self.irb.get_site_pose("sensor"))
        L["obj_pose_hist"].append(self.irb.get_payload_pose(out="T"))
        L["con_bool_hist"].append(float(self.irb.check_contact()))
        L["state_id_hist"].append(STATE_IDS.get(self.state, 0))
        L["arc_angle_hist"].append(angle)
        L["f_radial_hist"].append(f_radial)
        L["f_tangent_hist"].append(f_tangent)
        L["force_ref_hist"].append(self._force_ctrl.reference)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for k, v in self.log.items():
            arr = np.asarray(v, dtype=float)
            out[k] = arr
        # Convenience position channels, matching shove_simulation.py.
        if len(out["ball_pose_hist"]):
            out["ball_pos_hist"] = out["ball_pose_hist"][:, :3, 3]
            out["sens_pos_hist"] = out["sens_pose_hist"][:, :3, 3]
            out["obj_pos_hist"] = out["obj_pose_hist"][:, :3, 3]
        return out
