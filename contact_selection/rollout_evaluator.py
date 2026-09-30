"""Outcome instrumentation around PressPullFSM; no alternate control law."""
from dataclasses import replace
from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_irb120.robot.controllers.robot import controller
from contact_selection.controller import PressPullFSM, STATE_IDS
from contact_selection.candidate_generator import unexpected_contacts


def label_feasibility(metrics: dict, thresholds: dict, min_tip_deg: float,
                      abort_reason: str | None = None) -> tuple[bool, list[str]]:
    """Conjunctive feasibility definition; retain all observable failure reasons."""
    reasons = []
    numeric_keys = ('arc_contact_fraction', 'max_intended_tip_deg', 'max_pivot_drift_m',
                    'max_off_axis_deg', 'force_limit_margin_n', 'min_joint_margin_rad')
    if not all(np.isfinite(metrics[key]) for key in numeric_keys):
        return False, ['simulation_instability']
    if not metrics['numerically_stable']:
        reasons.append('simulation_instability')
    if not metrics['contact_established']:
        reasons.append('failed_to_establish_contact')
    if metrics['arc_contact_fraction'] < thresholds['min_arc_contact_fraction']:
        reasons.append('contact_loss')
    if metrics['max_intended_tip_deg'] < min_tip_deg:
        reasons.append('insufficient_intended_rotation')
    if metrics['max_pivot_drift_m'] > thresholds['max_pivot_drift_m']:
        reasons.append('unintended_pivot_or_sliding')
    if metrics['max_off_axis_deg'] > thresholds['max_off_axis_deg']:
        reasons.append('unintended_rotation')
    if metrics['force_limit_margin_n'] < 0:
        reasons.append('force_limit_exceeded')
    if metrics['min_joint_margin_rad'] < -thresholds['joint_limit_tolerance_rad']:
        reasons.append('joint_limit')
    if metrics['unintended_collision']:
        reasons.append('unintended_collision')
    if not metrics['completed'] or not metrics['done']:
        reasons.append('controller_failure')
    if abort_reason:
        if 'lost contact' in abort_reason and 'contact_loss' not in reasons:
            reasons.append('contact_loss')
        elif 'hard force' in abort_reason and 'force_limit_exceeded' not in reasons:
            reasons.append('force_limit_exceeded')
        elif 'unreachable' in abort_reason:
            reasons.append('unreachable')
        if 'controller_failure' not in reasons:
            reasons.append('controller_failure')
    return not reasons, reasons


def evaluate_rollout(model, initial_data, candidate, config, thresholds: dict,
                     estimator: Callable | None = None, step_callback: Callable | None = None):
    """Copy the complete initial MjData, then execute the repository controller.

    The candidate changes only the controller's existing press_offset_xy.
    Estimator callbacks receive native logged arrays and must return a dict;
    missing estimation is explicitly unavailable, never a fabricated score.
    """
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, initial_data)
    irb = controller(model, data)
    cfg = replace(config, press_offset_xy=tuple(candidate.press_offset_xy))
    fsm = PressPullFSM(irb, model, data, cfg)
    mujoco.mj_forward(model, data)
    table_geom_id = model.geom('table').id
    pivot_initial = data.site_xpos[irb.obj_frame_site].copy()
    if cfg.arc_center_xz is not None:
        pivot_initial[[0, 2]] = cfg.arc_center_xz
    pivot_body = data.xmat[irb.payload_body_id].reshape(3, 3).T @ (pivot_initial - data.xpos[irb.payload_body_id])
    metrics = dict(numerically_stable=True, contact_established=False, arc_contact_fraction=0.0,
                   max_intended_tip_deg=0.0, max_pivot_drift_m=0.0, max_off_axis_deg=0.0,
                   max_contact_force_n=0.0, force_limit_margin_n=cfg.force_hard_limit_n,
                   min_joint_margin_rad=1e10, unintended_collision=False, completed=False,
                   done=False, max_object_translation_m=0.0, fingertip_relative_travel_m=0.0,
                   final_tip_deg=0.0, controller_max_tip_deg=0.0)
    abort = None
    collision_events = {}
    initial_warnings = np.array(data.warning.number).copy()
    sample_times, contact_flags, joint_margins, contact_forces, pivot_positions = [], [], [], [], []
    diagnostic_rotations = []
    diagnostic_phases, finger_forces, finger_torques, table_torques, total_torques, com_positions = [], [], [], [], [], []
    arc_rotation0 = arc_pivot0 = arc_body0 = last_relative = None
    arc_ticks = arc_contacts = 0
    try:
        fsm.move_to_pre_squash()
    except RuntimeError as exc:
        abort = f'unreachable: {exc}'
    if abort is None:
        irb.ft_bias(n_samples=200)
        data.time = 0.0
        fsm._state_start_time = 0.0  # same timing reset as press_pull_simulation.py
        ceiling = (cfg.squash_timeout_sec + cfg.arc_timeout_sec + cfg.unarc_timeout_sec
                   + cfg.retract_duration_sec + 4 * cfg.lull_wait_sec + 10.0)
        while not fsm.done and data.time < ceiling:
            phase = fsm.state
            fsm.step()
            # Inspect current contacts consistently with FSM's pre-step log.
            force = 0.0
            touching = False
            finger_force, finger_torque, table_torque, total_torque = (np.zeros(3) for _ in range(4))
            com = data.xipos[irb.payload_body_id].copy()
            for contact_id, contact in enumerate(data.contact):
                g0, g1 = map(int, contact.geom)
                b0, b1 = model.geom_bodyid[[g0, g1]]
                if irb.payload_body_id not in (b0, b1):
                    continue
                other = g0 if b1 == irb.payload_body_id else g1
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_id, wrench)
                # MuJoCo contact-frame force acts on geom[1]; frame axes are rows.
                sign = 1 if b1 == irb.payload_body_id else -1
                world_force = sign * contact.frame.reshape(3, 3).T @ wrench[:3]
                world_torque = (np.cross(contact.pos - com, world_force)
                                + sign * contact.frame.reshape(3, 3).T @ wrench[3:])
                total_torque += world_torque
                if other == irb.ball_geom_id:
                    touching = True
                    force += float(np.linalg.norm(wrench[:3]))
                    finger_force += world_force
                    finger_torque += world_torque
                elif other == table_geom_id:
                    table_torque += world_torque
            metrics['contact_established'] |= touching
            metrics['max_contact_force_n'] = max(metrics['max_contact_force_n'], force)
            q = data.qpos[irb.joint_idx]
            margin = float(np.min(np.minimum(q - irb.q_min, irb.q_max - q)))
            metrics['min_joint_margin_rad'] = min(metrics['min_joint_margin_rad'], margin)
            pairs = unexpected_contacts(model, data, irb.payload_body_id, irb.ball_geom_id)
            metrics['unintended_collision'] |= bool(pairs)
            for pair in sorted(pairs):
                if pair not in collision_events:
                    collision_events[pair] = {
                        'geom_ids': list(pair),
                        'geom_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
                                       for gid in pair],
                        'body_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
                                       for gid in pair],
                        'first_time_sec': float(data.time), 'first_phase': phase,
                    }
            body = data.xpos[irb.payload_body_id].copy()
            rotation = data.xmat[irb.payload_body_id].reshape(3, 3).copy()
            pivot = body + rotation @ pivot_body
            if phase == 'ARC':
                if arc_rotation0 is None:
                    arc_rotation0, arc_pivot0, arc_body0 = rotation, pivot, body
                rotvec = np.rad2deg(Rotation.from_matrix(rotation @ arc_rotation0.T).as_rotvec())
                metrics['max_intended_tip_deg'] = max(metrics['max_intended_tip_deg'], float(-rotvec[1]))
                metrics['max_off_axis_deg'] = max(metrics['max_off_axis_deg'], float(np.linalg.norm(rotvec[[0, 2]])))
                metrics['max_pivot_drift_m'] = max(metrics['max_pivot_drift_m'], float(np.linalg.norm(pivot - arc_pivot0)))
                metrics['max_object_translation_m'] = max(metrics['max_object_translation_m'], float(np.linalg.norm(body - arc_body0)))
                relative = rotation.T @ (data.site_xpos[irb.ball_site] - body)
                if last_relative is not None and touching:
                    metrics['fingertip_relative_travel_m'] += float(np.linalg.norm(relative - last_relative))
                last_relative = relative if touching else None
                arc_ticks += 1
                arc_contacts += int(touching)
            diagnostic_rotations.append(rotation)
            diagnostic_phases.append(STATE_IDS[phase])
            finger_forces.append(finger_force)
            finger_torques.append(finger_torque)
            table_torques.append(table_torque)
            total_torques.append(total_torque)
            com_positions.append(com)
            sample_times.append(float(data.time))
            contact_flags.append(touching)
            joint_margins.append(margin)
            contact_forces.append(force)
            pivot_positions.append(pivot)
            old_time = data.time
            mujoco.mj_step(model, data)
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or not np.isfinite(data.sensordata).all() or data.time <= old_time
                    or np.any(np.array(data.warning.number) > initial_warnings)):
                metrics['numerically_stable'] = False
                abort = 'simulation instability or MuJoCo warning'
                break
            if step_callback is not None:
                step_callback(model, data, candidate)
        abort = abort or fsm.abort_reason
        if not fsm.done and abort is None:
            abort = 'rollout time ceiling'
    arrays = fsm.arrays()
    arrays.update(object_rotation_world=np.asarray(diagnostic_rotations).reshape(-1, 3, 3),
                  diagnostic_state_id=np.asarray(diagnostic_phases),
                  fingertip_force_world_n=np.asarray(finger_forces).reshape(-1, 3),
                  fingertip_torque_about_com_nm=np.asarray(finger_torques).reshape(-1, 3),
                  table_torque_about_com_nm=np.asarray(table_torques).reshape(-1, 3),
                  total_contact_torque_about_com_nm=np.asarray(total_torques).reshape(-1, 3),
                  object_com_world_m=np.asarray(com_positions).reshape(-1, 3),
                  diagnostic_time=np.asarray(sample_times), ball_contact=np.asarray(contact_flags),
                  joint_margin_rad=np.asarray(joint_margins), ball_contact_force_n=np.asarray(contact_forces),
                  pivot_position=np.asarray(pivot_positions))
    metrics.update(arc_contact_fraction=arc_contacts / max(1, arc_ticks),
                   completed=bool(fsm.completed), done=bool(fsm.done),
                   controller_max_tip_deg=float(fsm.max_tip_deg), final_tip_deg=float(fsm.object_tip_angle_deg()),
                   sim_time_sec=float(data.time), arc_ticks=arc_ticks)
    # Match FSM's force-limit channel and exempt RETRACT exactly as the FSM does.
    phases = arrays['state_id_hist']
    if len(phases):
        measured = np.where(np.isin(phases, [3, 4]), arrays['f_radial_hist'], np.abs(arrays['w_world_hist'][:, 2]))
        safety = measured[phases != 5]
        metrics['force_limit_margin_n'] = float(cfg.force_hard_limit_n - np.max(safety)) if len(safety) else cfg.force_hard_limit_n
    feasible, reasons = label_feasibility(metrics, thresholds, cfg.min_tip_angle_deg, abort)
    estimate = {'status': 'unavailable', 'reason': 'No validated callable press-pull batch fitter in this repository'}
    if estimator is not None and feasible:
        estimate = estimator(arrays)
    return {'feasible': feasible, 'failure_modes': reasons, 'abort_reason': abort,
            'label_scope': 'contact_with_controller_configuration',
            'finger_pitch_deg': getattr(cfg, 'finger_pitch_deg', 0.0),
            'collision_events': list(collision_events.values()),
            'metrics': metrics, 'estimator_outputs': estimate, 'estimator_errors': None,
            'information_quality': None, 'arc_exit_reason': fsm.arc_exit_reason}, arrays
