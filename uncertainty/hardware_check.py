"""Measure the error sources on the 40 recorded hardware trials, and test two corrections.

Uses only the CSVs shipped with ``press_pull_estimator`` (``../press-pull-tipping/code/data``),
so it runs on any machine that has that repo. Per trial:

- **F/T offset and drift**: the mean wrench over the first 0.3 s of SQUASH (still
  descending, not in contact) is the offset the estimator sees at the start, which is
  the tare error plus drift since the tare. The last 0.5 s of RETRACT (finger off the
  object) minus that is the drift within the trial. The finger holds its orientation
  throughout, so the tool's gravity load is the same in all windows. The std in the
  SQUASH window is the white-noise level.
- **Pivot**: ``pivot_from_trajectory`` x against the commanded ARC_CENTER, and the
  creep offsets ``relative_pivots`` applies.
- **Finger rotation** over ARC/UNARC. If it is near zero the ball has to roll on the
  top face, and ``rolling_corrected_tilt`` applies.

It then refits every trial four ways: as published, with rolling-corrected tilt,
with the SQUASH offset and a linear-in-time drift removed, and with both. The ball radius is the URDF's
0.01325 m; it is not measured on the real finger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from uncertainty.estimator_mc import SimTrial, rolling_corrected_tilt, we

from press_pull_estimator.io import load_push, load_trial, trial_paths, _read  # noqa: E402  (path set by estimator_mc)
from press_pull_estimator.objects import OBJECTS  # noqa: E402

BALL_RADIUS_M = 0.01325
PRE_S, POST_S = 0.3, 0.5
VARIANTS = ('published', 'rolling', 'ft_offset', 'rolling+ft_offset')


def windows(ft_t: np.ndarray, pose: np.ndarray):
    """Boolean masks on the F/T clock: start of SQUASH (pre) and end of RETRACT (post)."""
    t, s = pose[:, 0], pose[:, 8]
    t_sq, t_re = t[s == we.STATE_SQUASH].min(), t[s == we.STATE_RETRACT].max()
    return (ft_t >= t_sq) & (ft_t < t_sq + PRE_S), (ft_t > t_re - POST_S) & (ft_t <= t_re)


def drift_corrected(trial: we.Trial, ft: np.ndarray, pre, post) -> we.Trial:
    """Subtract a drift that grows linearly in time from the pre to the post window."""
    t = ft[:, 0]
    w = np.hstack((ft[:, 4:7], ft[:, 1:4]))  # [tau, f], as Trial stores it
    w0, w1 = w[pre].mean(0), w[post].mean(0)
    t0, t1 = t[pre].mean(), t[post].mean()
    alpha = np.clip((trial.t - t0) / (t1 - t0), 0.0, 1.0)[:, None]
    return we.Trial(trial.t, trial.p_ball, trial.q_ball,
                    trial.wrench_S - (w0 + alpha * (w1 - w0)), trial.state, trial.name)


def rolling_corrected(trial: we.Trial, pivot: np.ndarray) -> we.Trial:
    """Same trial with the ball moved onto the object-rigid arc of the rolling-corrected tilt.

    The sensor pose stays the measured one, so only the tilt the estimator infers changes.
    """
    rv, in_contact = we.object_tilt(trial, pivot)
    i0 = int(np.argmax(in_contact))
    x0, _, height = trial.p_ball[i0] - pivot
    theta = rolling_corrected_tilt(-rv[:, 1], x0, height, BALL_RADIUS_M)
    R = Rotation.from_rotvec(np.outer(-theta, [0.0, 1.0, 0.0])).as_matrix()
    p = pivot + np.einsum('nij,j->ni', R, trial.p_ball[i0] - pivot)
    p[~in_contact] = trial.p_ball[~in_contact]
    return SimTrial(trial.t, p, trial.q_ball, trial.wrench_S, trial.state, trial.name,
                    sensor_T=trial.sensor_pose())


def check_object(obj: str) -> dict:
    gt = OBJECTS[obj]
    paths = trial_paths(obj)
    trials = [load_trial(p) for p in paths]
    pivots = we.relative_pivots(trials)
    mu_m = we.coulomb_product_from_push(*load_push(obj))
    rows, fits = [], {v: [] for v in VARIANTS}
    for path, tr, pv in zip(paths, trials, pivots):
        ft, pose = _read(path), _read(path.replace('_ft.csv', '_pose.csv'))
        pre, post = windows(ft[:, 0], pose)
        offset = ft[pre, 1:7].mean(0)  # what the estimator sees at SQUASH: tare error + drift so far
        drift = ft[post, 1:7].mean(0) - offset
        arc = np.isin(tr.state, [we.STATE_ARC, we.STATE_UNARC])
        rot = Rotation.from_quat(tr.q_ball[arc])
        finger_deg = np.degrees(np.ptp((rot * rot[0].inv()).as_rotvec()[:, 1]))
        rv, in_contact = we.object_tilt(tr, pv)
        x0, _, height = tr.p_ball[int(np.argmax(in_contact))] - pv
        fit_pivot = we.pivot_from_trajectory(tr)[0]
        rows.append({
            'trial': tr.name,
            'offset_at_squash_force_norm_n': float(np.linalg.norm(offset[:3])),
            'offset_at_squash_torque_norm_nm': float(np.linalg.norm(offset[3:])),
            'drift_force_n': drift[:3].tolist(), 'drift_force_norm_n': float(np.linalg.norm(drift[:3])),
            'drift_torque_nm': drift[3:].tolist(), 'drift_torque_norm_nm': float(np.linalg.norm(drift[3:])),
            'white_force_std_n': ft[pre, 1:4].std(0).tolist(), 'white_torque_std_nm': ft[pre, 4:7].std(0).tolist(),
            'duration_s': float(np.ptp(tr.t[arc])),
            'pivot_fit_x_minus_arc_center_mm': None if fit_pivot is None
            else float(1e3 * (fit_pivot[0] - we.PIVOT_DEFAULT[0])),
            'pivot_used_x_mm': float(1e3 * pv[0]),
            'finger_rotation_ptp_deg': float(finger_deg),
            'noslip_dev_mm': we.noslip_deviation_mm(tr, pv),
            'ball_x0_m': float(x0), 'ball_height_m': float(height),
            'rolling_slope': float(1 - BALL_RADIUS_M * height / (height ** 2 + x0 ** 2)),
            'max_tilt_deg': float(np.degrees(-rv[arc, 1].min())),
        })
        variants = {'published': tr, 'rolling': rolling_corrected(tr, pv),
                    'ft_offset': drift_corrected(tr, ft, pre, post)}
        variants['rolling+ft_offset'] = rolling_corrected(variants['ft_offset'], pv)
        for name, t in variants.items():
            r = we.estimate_press_pull(t, gt['com_x'], pv)
            fits[name].append([r.mass, r.com_z, mu_m / r.mass, r.arc.mass, r.unarc.mass,
                               r.arc.com_z, r.unarc.com_z])
    truth = np.array([gt['mass'], gt['com_z'], mu_m / gt['mass']])
    summary = {}
    for name, f in fits.items():
        f = np.array(f)
        mean = f.mean(0)
        summary[name] = {
            'mass': mean[0], 'com_z': mean[1], 'mu': mean[2],
            'mass_err': mean[0] / truth[0] - 1, 'com_z_err': mean[1] / truth[1] - 1, 'mu_err': mean[2] / truth[2] - 1,
            'mass_sd': f[:, 0].std(ddof=1), 'com_z_sd': f[:, 1].std(ddof=1),
            'arc_mass': mean[3], 'unarc_mass': mean[4], 'arc_com_z': mean[5], 'unarc_com_z': mean[6],
            'arc_unarc_mass_gap': (mean[3] - mean[4]) / truth[0], 'arc_unarc_com_z_gap': (mean[5] - mean[6]) / truth[1],
        }
    keys = ('offset_at_squash_force_norm_n', 'offset_at_squash_torque_norm_nm', 'drift_force_norm_n', 'drift_torque_norm_nm', 'pivot_fit_x_minus_arc_center_mm',
            'finger_rotation_ptp_deg', 'rolling_slope', 'max_tilt_deg', 'duration_s', 'noslip_dev_mm')
    stats = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in keys}
    stats['pivot_used_x_sd_mm'] = float(np.std([r['pivot_used_x_mm'] for r in rows]))
    stats['white_force_std_n'] = float(np.mean([r['white_force_std_n'] for r in rows]))
    stats['white_torque_std_nm'] = float(np.mean([r['white_torque_std_nm'] for r in rows]))
    stats['drift_force_mean_vec_n'] = np.mean([r['drift_force_n'] for r in rows], 0).tolist()
    return {'object': obj, 'truth': dict(zip(('mass', 'com_z', 'mu'), truth.tolist())),
            'stats': stats, 'fits': summary, 'trials': rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=None, help='write hardware_check.json here')
    args = ap.parse_args(argv)
    out = {obj: check_object(obj) for obj in OBJECTS}
    print(f"{'object':<11}{'variant':<19}{'m err':>8}{'z_c err':>9}{'mu err':>8}"
          f"{'ARC-UNARC m':>13}{'ARC-UNARC z':>13}")
    for obj, r in out.items():
        for v, f in r['fits'].items():
            print(f"{obj:<11}{v:<19}{f['mass_err']:>+8.1%}{f['com_z_err']:>+9.1%}{f['mu_err']:>+8.1%}"
                  f"{f['arc_unarc_mass_gap']:>+13.1%}{f['arc_unarc_com_z_gap']:>+13.1%}")
    print(f"\n{'object':<11}{'|offset F| N':>14}{'|offset tau| Nm':>17}")
    for obj, r in out.items():
        s = r['stats']
        print(f"{obj:<11}{s['offset_at_squash_force_norm_n'][0]:>8.3f}±{s['offset_at_squash_force_norm_n'][1]:.3f}"
              f"{s['offset_at_squash_torque_norm_nm'][0]:>11.4f}±{s['offset_at_squash_torque_norm_nm'][1]:.4f}")
    print(f"\n{'object':<11}{'|drift F| N':>13}{'|drift tau| Nm':>16}{'white F N':>11}"
          f"{'pivot fit-cmd mm':>18}{'finger rot deg':>16}{'roll slope':>11}")
    for obj, r in out.items():
        s = r['stats']
        print(f"{obj:<11}{s['drift_force_norm_n'][0]:>8.3f}±{s['drift_force_norm_n'][1]:.3f}"
              f"{s['drift_torque_norm_nm'][0]:>10.4f}±{s['drift_torque_norm_nm'][1]:.4f}"
              f"{s['white_force_std_n']:>11.4f}"
              f"{s['pivot_fit_x_minus_arc_center_mm'][0]:>12.1f}±{s['pivot_fit_x_minus_arc_center_mm'][1]:.1f}"
              f"{s['finger_rotation_ptp_deg'][0]:>16.2f}{s['rolling_slope'][0]:>11.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
