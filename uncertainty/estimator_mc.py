"""Monte Carlo propagation of measurement and geometry noise through the press-pull estimator.

One noise-free simulated press-pull rollout is perturbed many times and refit
with the canonical ``press_pull_estimator.estimate_press_pull``. Nothing is
re-simulated: the noise models what the *estimator* is told, not how the robot
moves. That is the right scope for asking which inputs the fit is sensitive to,
and it makes thousands of fits cheap (~0.07 s each at 500 Hz).

Where each source enters the estimator:

- F/T white noise and bias -> ``wrench_S``, transformed to the pivot frame by
  ``applied_wrench_in_object``. Bias does not average out over the sweep; white
  noise mostly does.
- Tilt noise/offset -> the estimator computes tilt from the ball's rotation about
  the pivot, so it is injected by rotating ``p_ball`` about the true pivot. The
  sensor pose is left alone, so this isolates a pure angle-measurement error
  (the hardware equivalent is the vision pitch stream, or FK error that only
  shows up in the angle).
- Ball/FK position noise -> moves ``p_ball`` and the sensor pose together, since
  both come from the same FK chain.
- Tool offset -> the calibrated ball->sensor distance (``BALL_TO_WRENCH_ORIGIN_X``
  on hardware): shifts the believed sensor origin along the tool axis, which
  changes the moment arm of the measured force.
- Pivot (camera-derived) -> the ``pivot`` argument. It sets both the torque
  reference point and the tilt angle, so it enters twice.
- ``com_x`` (prior trial / camera) -> the ``com_x`` argument. The fit identifies
  m*|r_com| and theta*, and ``com_x`` is what separates m from z_c.
- Friction: mu = (mu*m)_push / m_hat. The shove itself is out of scope, so only
  the F/T force bias on the slip force and the mass error are propagated.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from util.paths import REPO_ROOT

try:
    from press_pull_estimator.estimator import wrench_estimator as we
except ImportError:  # Not pip-installed yet (CLAUDE.md TODO); fall back to the sibling checkout.
    sys.path.insert(0, str(REPO_ROOT.parent / 'press-pull-tipping' / 'code'))
    from press_pull_estimator.estimator import wrench_estimator as we

PARAMS = ('mass', 'com_z', 'mu')
G = we.G


@dataclass(frozen=True)
class NoiseSpec:
    """One-sigma input uncertainties. Defaults are plausible hardware magnitudes, not measured ones."""
    ft_force_n: float = 0.05        # white, per sample, each force axis
    ft_torque_nm: float = 0.002     # white, per sample, each torque axis
    ft_bias_force_n: float = 0.05   # constant per trial (drift after biasing)
    ft_bias_torque_nm: float = 0.002
    tilt_deg: float = 0.1           # white, per sample
    tilt_bias_deg: float = 0.2      # constant offset over ARC/UNARC
    ball_pos_mm: float = 0.1        # white FK position noise, ball and sensor together
    tool_offset_mm: float = 1.0     # ball->sensor calibration, constant
    pivot_mm: float = 2.0           # pivot x and z, constant (camera)
    com_x_mm: float = 2.0           # horizontal CoM prior, constant

    def only(self, *names: str) -> NoiseSpec:
        """Copy with every source zeroed except `names` (kept at this spec's value)."""
        return NoiseSpec(**{f.name: getattr(self, f.name) if f.name in names else 0.0 for f in fields(self)})

    def scaled(self, k: float) -> NoiseSpec:
        return NoiseSpec(**{f.name: k * getattr(self, f.name) for f in fields(self)})


# Levels measured on the 40 hardware trials by ``hardware_check.py`` (2026-10-06), per
# axis. White = std while descending in SQUASH (0.006-0.011 N, 0.18-0.28 mN·m). Bias =
# the offset at the start of SQUASH relative to the tare (|F| 0.03-0.08 N, |tau|
# 1.1-2.3 mN·m, so ~0.035 N and 1 mN·m per axis); that is what the published estimator
# sees. Re-taring at SQUASH would leave only the drift within the trial (|F|
# 0.012-0.023 N, |tau| 0.3-0.7 mN·m). Pivot = trajectory fit minus ARC_CENTER (-1.0 to
# -1.7 mm, creep up to 1.3 mm). tilt_bias, tool_offset and com_x can't be measured
# from those logs and keep their assumed values.
MEASURED_NOISE = NoiseSpec(ft_force_n=0.01, ft_torque_nm=0.00025, ft_bias_force_n=0.035,
                           ft_bias_torque_nm=0.001, pivot_mm=1.5)
RETARED_NOISE = replace(MEASURED_NOISE, ft_bias_force_n=0.012, ft_bias_torque_nm=0.0004)
NOISE_PRESETS = {'assumed': NoiseSpec(), 'measured': MEASURED_NOISE, 'retared': RETARED_NOISE}


# Groups used for one-at-a-time sensitivity and sweeps; paired fields always move together.
SOURCES = {
    'ft_white': ('ft_force_n', 'ft_torque_nm'),
    'ft_bias': ('ft_bias_force_n', 'ft_bias_torque_nm'),
    'tilt_white': ('tilt_deg',),
    'tilt_bias': ('tilt_bias_deg',),
    'ball_pos': ('ball_pos_mm',),
    'tool_offset': ('tool_offset_mm',),
    'pivot': ('pivot_mm',),
    'com_x': ('com_x_mm',),
}


class SimTrial(we.Trial):
    """Trial whose sensor pose is the logged sim sensor site, not the hardware ball offset."""
    def __init__(self, *args, sensor_T: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_T = sensor_T

    def sensor_pose(self) -> np.ndarray:
        return self.sensor_T


@dataclass(frozen=True)
class Nominal:
    trial: SimTrial
    pivot: np.ndarray
    com_x: float
    truth: dict  # mass, com_z, mu: simulator ground truth


def load_rollout(run_dir: Path, decimate: int = 2) -> Nominal:
    """Load a ``run_press_pull_demo.py`` output as an estimator trial.

    The sim logs at 1 kHz. ``decimate=2`` gives the 500 Hz grid the estimator's
    Butterworth filter assumes, and sets how many independent white-noise draws
    each fit sees.
    """
    import mujoco
    a = np.load(run_dir / 'trajectory.npz')
    cfg = json.loads((run_dir / 'config.json').read_text())
    sl = slice(None, None, decimate)
    B, S, w = a['ball_pose_hist'][sl], a['sens_pose_hist'][sl], a['w_hist'][sl]
    trial = SimTrial(a['t_hist'][sl], B[:, :3, 3].copy(), Rotation.from_matrix(B[:, :3, :3]).as_quat(),
                     np.hstack((w[:, 3:], w[:, :3])), a['state_id_hist'][sl].astype(int),
                     name=run_dir.name, sensor_T=S.copy())
    pivot = np.asarray(cfg['pivot_world_m'], float)
    com = a['object_com_world_m'][0]
    model = mujoco.MjModel.from_binary_path(str(run_dir / 'model.mjb'))
    payload = model.body('payload').id
    # MuJoCo combines geom friction by maximum (CLAUDE.md), so the table pair uses the larger value.
    mu = max(model.geom_friction[model.geom('table').id, 0],
             *model.geom_friction[model.geom_bodyid == payload, 0])
    truth = {'mass': float(cfg['mass_kg']), 'com_z': float(com[2] - pivot[2]), 'mu': float(mu)}
    return Nominal(trial, pivot, float(com[0] - pivot[0]), truth)


def perturb(nom: Nominal, noise: NoiseSpec, rng: np.random.Generator) -> tuple[SimTrial, np.ndarray, float]:
    """Return (trial, pivot, com_x) as a noisy measurement pipeline would report them."""
    tr = nom.trial
    n = len(tr.t)
    wrench = tr.wrench_S.copy()  # [tau, f]
    wrench[:, :3] += rng.normal(0, noise.ft_torque_nm, (n, 3)) + rng.normal(0, noise.ft_bias_torque_nm, 3)
    wrench[:, 3:] += rng.normal(0, noise.ft_force_n, (n, 3)) + rng.normal(0, noise.ft_bias_force_n, 3)

    p_ball, sensor_T = tr.p_ball.copy(), tr.sensor_T.copy()
    arc = np.isin(tr.state, [we.STATE_ARC, we.STATE_UNARC])
    dtheta = np.deg2rad(rng.normal(0, noise.tilt_deg, n) + arc * rng.normal(0, noise.tilt_bias_deg))
    if np.any(dtheta):
        R = Rotation.from_rotvec(dtheta[:, None] * np.array([0.0, 1.0, 0.0])).as_matrix()
        p_ball = nom.pivot + np.einsum('nij,nj->ni', R, p_ball - nom.pivot)
    dp = rng.normal(0, 1e-3 * noise.ball_pos_mm, (n, 3))
    p_ball += dp
    sensor_T[:, :3, 3] += dp
    tool_axis = Rotation.from_quat(tr.q_ball).as_matrix()[:, :, 0]
    sensor_T[:, :3, 3] -= tool_axis * 1e-3 * rng.normal(0, noise.tool_offset_mm)

    noisy = SimTrial(tr.t, p_ball, tr.q_ball, wrench, tr.state, name=tr.name, sensor_T=sensor_T)
    pivot = nom.pivot + 1e-3 * rng.normal(0, noise.pivot_mm, 3) * np.array([1.0, 0.0, 1.0])
    com_x = nom.com_x + 1e-3 * rng.normal(0, noise.com_x_mm)
    return noisy, pivot, com_x


def estimate(nom: Nominal, trial: SimTrial, pivot: np.ndarray, com_x: float,
             slip_bias_n: float = 0.0) -> dict:
    """Run the canonical estimator; mu resolves the push's Coulomb product with the fitted mass."""
    r = we.estimate_press_pull(trial, com_x=com_x, pivot=pivot)
    mu_m = nom.truth['mu'] * nom.truth['mass'] + slip_bias_n / G
    return {'mass': float(r.mass), 'com_z': float(r.com_z), 'mu': float(we.resolve_friction(mu_m, r.mass)),
            'theta_star_deg': float(r.theta_star_deg), 'result': r}


def monte_carlo(nom: Nominal, noise: NoiseSpec, samples: int, seed: int = 0,
                nls: int = 0) -> dict:
    """Refit `samples` perturbed copies. `nls` > 0 also computes NLS covariance on the first `nls` draws."""
    rng = np.random.default_rng(seed)
    rows, nls_covs = [], []
    for i in range(samples):
        trial, pivot, com_x = perturb(nom, noise, rng)
        slip_bias = rng.normal(0, noise.ft_bias_force_n)
        try:
            est = estimate(nom, trial, pivot, com_x, slip_bias)
        except (ValueError, np.linalg.LinAlgError):
            rows.append({k: np.nan for k in (*PARAMS, 'theta_star_deg')})
            continue
        rows.append({k: est[k] for k in (*PARAMS, 'theta_star_deg')})
        if i < nls:
            nls_covs.append(nls_covariance(trial, com_x, pivot, est))
    x = np.array([[row[k] for k in PARAMS] for row in rows])
    out = {'noise': asdict(noise), 'samples': rows, 'stats': summarize(x, nom.truth)}
    if nls_covs:
        out['nls'] = combine_nls(nls_covs)
    return out


def summarize(x: np.ndarray, truth: dict) -> dict:
    """Mean, std, covariance, correlation and 95% intervals over finite rows."""
    ok = np.isfinite(x).all(axis=1)
    x = x[ok]
    cov = np.cov(x, rowvar=False) if len(x) > 1 else np.zeros((3, 3))
    std = np.sqrt(np.diag(cov))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(std, std)
    corr = np.where(np.outer(std, std) > 0, corr, np.nan)
    lo, hi = np.percentile(x, [2.5, 97.5], axis=0) if len(x) else (np.full(3, np.nan),) * 2
    return {'n': int(ok.sum()), 'failed': int((~ok).sum()), 'params': list(PARAMS),
            'mean': x.mean(0).tolist(), 'std': std.tolist(),
            'rel_std': (std / np.array([truth[k] for k in PARAMS])).tolist(),
            'bias_vs_truth': (x.mean(0) - np.array([truth[k] for k in PARAMS])).tolist(),
            'ci95': [lo.tolist(), hi.tolist()], 'cov': cov.tolist(), 'corr': corr.tolist()}


# ------------------------------------------------------------------------------------
#  Local NLS covariance
# ------------------------------------------------------------------------------------
def _phase_residuals(trial: SimTrial, com_x: float, pivot: np.ndarray):
    """Rebuild the estimator's per-phase residual functions (same mask, projection, model).

    Mirrors ``estimate_press_pull`` line for line so the Jacobian is of the fit the
    package actually solves; the package does not expose it.
    """
    rv, _ = we.object_tilt(trial, pivot)
    mask = np.isin(trial.state, [we.STATE_ARC, we.STATE_UNARC]) & (rv[:, 1] < -np.deg2rad(1.0))
    T_B_O = we.make_T(np.tile(pivot, (len(trial.t), 1)), Rotation.from_rotvec(rv).as_matrix())
    w_app = we.applied_wrench_in_object(trial.wrench_S[mask], trial.sensor_pose()[mask], T_B_O[mask])
    rv_c, st_c = rv[mask], trial.state[mask]
    tip_axis = rv_c.mean(0) / np.linalg.norm(rv_c.mean(0))
    tau = w_app[:, :3] @ tip_axis
    funcs = {}
    for name, state in (('arc', we.STATE_ARC), ('unarc', we.STATE_UNARC)):
        sel = st_c == state

        def resid(p, rv_ph=rv_c[sel], tau_ph=tau[sel]):
            model = we.gravity_wrench_in_object(rv_ph, np.array([com_x, 0.0, p[1]]), p[0])[:, :3] @ tip_axis
            return model - tau_ph
        funcs[name] = resid
    return funcs


def nls_covariance(trial: SimTrial, com_x: float, pivot: np.ndarray, est: dict) -> dict:
    """Gauss-Newton covariance s^2 (J^T J)^-1 per sweep, combined for the ARC/UNARC average.

    The two sweeps see disjoint samples, so with independent noise the average's
    covariance is (C_arc + C_unarc) / 4. mu = (mu m)/m_hat gets the delta method,
    ignoring the push term. This only describes white measurement noise: constant
    errors (bias, pivot, com_x) shift every residual together and are invisible to it.
    """
    r = est['result']
    funcs = _phase_residuals(trial, com_x, pivot)
    covs, conds = [], []
    for name, fit in (('arc', r.arc), ('unarc', r.unarc)):
        p = np.array([fit.mass, fit.com_z])
        f0 = funcs[name](p)
        h = 1e-6 * np.maximum(np.abs(p), 1e-3)
        J = np.column_stack([(funcs[name](p + h[j] * np.eye(2)[j]) - f0) / h[j] for j in range(2)])
        s2 = f0 @ f0 / max(len(f0) - 2, 1)
        JtJ = J.T @ J
        covs.append(s2 * np.linalg.inv(JtJ))
        Js = J * p  # relative-parameter scaling, so the condition number is unit-free
        conds.append(float(np.linalg.cond(Js.T @ Js)))
    c2 = (covs[0] + covs[1]) / 4
    dmu = -est['mu'] / est['mass']  # d mu / d m
    A = np.array([[1, 0], [0, 1], [dmu, 0]])
    return {'cov': (A @ c2 @ A.T).tolist(), 'cond_arc': conds[0], 'cond_unarc': conds[1]}


def combine_nls(items: list[dict]) -> dict:
    cov = np.median([np.array(i['cov']) for i in items], axis=0)
    std = np.sqrt(np.diag(cov))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(std, std)
    return {'n': len(items), 'cov': cov.tolist(), 'std': std.tolist(), 'corr': corr.tolist(),
            'cond_arc': float(np.median([i['cond_arc'] for i in items])),
            'cond_unarc': float(np.median([i['cond_unarc'] for i in items]))}


def truncate_sweep(nom: Nominal, max_tilt_deg: float) -> Nominal:
    """Drop ARC/UNARC samples above `max_tilt_deg`, emulating a shorter (worse-conditioned) sweep."""
    rv, _ = we.object_tilt(nom.trial, nom.pivot)
    state = nom.trial.state.copy()
    state[np.isin(state, [we.STATE_ARC, we.STATE_UNARC]) & (-np.rad2deg(rv[:, 1]) > max_tilt_deg)] = we.STATE_RETRACT
    tr = nom.trial
    trial = SimTrial(tr.t, tr.p_ball, tr.q_ball, tr.wrench_S, state, name=tr.name, sensor_T=tr.sensor_T)
    return replace(nom, trial=trial)


def synthetic_nominal(mass=0.66, com_x=0.05, com_z=0.14, mu=0.5, max_tilt_deg=15.0, n=1500,
                      ball_height=0.3135, ball_x=0.0195) -> Nominal:
    """Exact-model trial for tests: the finger wrench balances gravity about the pivot at every sample.

    ``gravity_wrench_in_object`` already carries the sign of the torque the finger
    must apply (the estimator's residual is ``model - tau_app``), so the applied
    wrench is set equal to it. Only its torque about the tip axis enters the fit,
    and the table reaction acts at the pivot and adds no torque there, so this is
    exact for the estimator's model. ``ball_height``/``ball_x`` place the ball centre
    relative to the pivot, to mimic other objects' lever arms.
    """
    pivot = np.array([0.53, 0.0, 0.05])
    third = n // 3
    theta = np.r_[np.zeros(n - 2 * third), np.linspace(0, max_tilt_deg, third), np.linspace(max_tilt_deg, 0, third)]
    state = np.r_[np.full(n - 2 * third, we.STATE_LULL), np.full(third, we.STATE_ARC), np.full(third, we.STATE_UNARC)]
    rv = np.deg2rad(theta)[:, None] * np.array([0.0, -1.0, 0.0])
    R = Rotation.from_rotvec(rv).as_matrix()
    p_ball = pivot + np.einsum('nij,j->ni', R, np.array([ball_x, 0.0, ball_height]))
    sensor_T = we.make_T(p_ball - np.array([we.BALL_TO_WRENCH_ORIGIN_X, 0, 0]), np.tile(np.eye(3), (n, 1, 1)))
    T_B_O = we.make_T(np.tile(pivot, (n, 1)), R)
    w_app_O = we.gravity_wrench_in_object(rv, np.array([com_x, 0.0, com_z]), mass)
    # Invert w_app = -Ad_{T_SO}^T w_S.
    AdT = we.adjoint(we.trans_inv(sensor_T) @ T_B_O).transpose(0, 2, 1)
    w_S = -np.linalg.solve(AdT, w_app_O[..., None])[..., 0]
    trial = SimTrial(np.arange(n) / 500.0, p_ball, np.tile([0, 0, 0, 1.0], (n, 1)), w_S, state,
                     name='synthetic', sensor_T=sensor_T)
    return Nominal(trial, pivot, com_x, {'mass': mass, 'com_z': com_z, 'mu': mu})


# ------------------------------------------------------------------------------------
#  Noise-free bias breakdown (sim rollout only)
# ------------------------------------------------------------------------------------
def rolling_corrected_tilt(phi: np.ndarray, x0: float, height: float, radius: float) -> np.ndarray:
    """Object tilt from the ball's angle about the pivot, for a ball that rolls on the top face.

    With a world-fixed finger the ball cannot be carried rigidly: it rolls, so its
    centre moves r*theta along the object's top face. In the object frame the centre
    sits at (x0 + r*theta, H), so phi = theta - atan2(x0 + r*theta, H) + atan2(x0, H),
    roughly theta * (1 - r H / (H^2 + x0^2)). For the box that is 0.958: the estimator's
    tilt reads ~4% low. The no-slip check (constant |p_ball - pivot|) cannot see it.
    """
    from scipy.optimize import brentq

    def g(th, p):
        return th - np.arctan2(x0 + radius * th, height) + np.arctan2(x0, height) - p
    return np.array([brentq(g, -0.5, 1.0, args=(p,)) if p else 0.0 for p in phi])


def bias_breakdown(run_dir: Path, decimate: int = 2) -> dict:
    """Attribute the noise-free estimate's bias to geometry, using sim ground truth.

    Steps: as-is; tilt corrected for ball rolling; plus the object's tilt at the
    reference sample (the press pre-tilts it through contact compliance, and the
    estimator calls that pose zero). Also checks that the sim F/T reading equals the
    fingertip contact force, and that torques about the CoM balance (quasi-static).
    """
    import mujoco
    nom = load_rollout(run_dir, decimate)
    a = np.load(run_dir / 'trajectory.npz')
    sl = slice(None, None, decimate)
    tr, s = nom.trial, nom.trial.state
    model = mujoco.MjModel.from_binary_path(str(run_dir / 'model.mjb'))
    radius = float(model.geom_size[model.geom('push_ball_col').id, 0])
    rv, _ = we.object_tilt(tr, nom.pivot)
    i0 = int(np.argmax(np.isin(s, [we.STATE_LULL, we.STATE_ARC, we.STATE_UNARC, we.STATE_RETRACT])))
    x0, _, height = tr.p_ball[i0] - nom.pivot
    R_obj = a['object_rotation_world'][sl]
    ref_tilt_deg = float(-np.degrees(Rotation.from_matrix(R_obj[i0] @ R_obj[0].T).as_rotvec()[1]))
    theta_true = -Rotation.from_matrix(R_obj @ R_obj[i0].T).as_rotvec()[:, 1]
    arc = np.isin(s, [we.STATE_ARC, we.STATE_UNARC])
    truth_theta = float(np.degrees(np.arctan2(nom.com_x, nom.truth['com_z'])))

    def fit(theta, label, offset_deg=0.0):
        R = Rotation.from_rotvec(np.outer(-theta, [0.0, 1.0, 0.0])).as_matrix()
        p = nom.pivot + np.einsum('nij,j->ni', R, tr.p_ball[i0] - nom.pivot)
        r = we.estimate_press_pull(SimTrial(tr.t, p, tr.q_ball, tr.wrench_S, s, sensor_T=tr.sensor_T),
                                   com_x=nom.com_x, pivot=nom.pivot)
        th = r.theta_star_deg + offset_deg
        # With com_x fixed, theta* sets z_c; the amplitude m*|r_com| then sets m.
        zc = nom.com_x / np.tan(np.radians(th))
        m = r.mass * np.hypot(nom.com_x, r.com_z) / np.hypot(nom.com_x, zc)
        return {'step': label, 'theta_star_deg': th, 'mass_rel_err': m / nom.truth['mass'] - 1,
                'com_z_rel_err': zc / nom.truth['com_z'] - 1}
    phi = -rv[:, 1]
    corrected = rolling_corrected_tilt(phi, x0, height, radius)
    f_sensor = a['w_world_hist'][sl][:, :3]
    f_contact = a['fingertip_force_world_n'][sl]
    tau = a['fingertip_torque_about_com_nm'][sl] + a['table_torque_about_com_nm'][sl]
    return {
        'truth_theta_star_deg': truth_theta, 'ball_radius_m': radius, 'x0_m': float(x0), 'height_m': float(height),
        'tilt_slope_ball_vs_object': float(np.polyfit(theta_true[arc], phi[arc], 1)[0]),
        'tilt_slope_rolling_model': float(1 - radius * height / (height ** 2 + x0 ** 2)),
        'reference_pretilt_deg': ref_tilt_deg,
        'steps': [fit(phi, 'as estimated'),
                  fit(corrected, 'rolling-corrected tilt'),
                  fit(corrected, 'rolling-corrected + reference pre-tilt', ref_tilt_deg)],
        'sensor_minus_contact_force_n': float(np.abs(f_sensor[arc] + f_contact[arc]).mean()),
        'torque_about_com_residual_nm': float(np.abs(tau[arc, 1]).mean()),
        'torque_about_com_finger_nm': float(np.abs(a['fingertip_torque_about_com_nm'][sl][arc, 1]).mean()),
    }
