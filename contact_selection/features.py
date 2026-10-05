"""Interpretable pre-execution features; no outcome or ground-truth mass input."""
import numpy as np

FEATURE_NAMES = ['local_x_m', 'local_y_m', 'local_z_m', 'normal_x', 'normal_y', 'normal_z',
                 'normalized_height', 'pivot_dx_m', 'pivot_dz_m', 'arc_radius_m',
                 'width_m', 'depth_m', 'height_m', 'approach_error_m', 'joint_margin_rad',
                 'pivot_ray_angle_rad']


def pivot_ray_angle(dx: float, dz: float) -> float:
    """Angle of the pivot->contact ray from vertical, in the fixed XZ arc plane.

    The FSM regulates force radially toward the pivot and moves tangentially, so
    the press carries no moment about the pivot. What it does do is push the object
    along this ray: tan(angle) = dx/dz is the horizontal/vertical force ratio the
    press alone puts through the object onto the table under the pivot, before any
    pull is added. Once that exceeds the table's friction the pivot slides, which is
    the failure mode of every saved low-friction box and L contact. Lighter objects
    or harder presses move the threshold toward exactly tan(angle) < mu_table.

    This is the nonlinear version of the hardware selector's double score, which
    sums a tipping arm (dz, height above the pivot) and an anti-tip arm (-w*dx,
    inboard distance) and uses positions only. That linear sum is already in the
    span of pivot_dx_m/pivot_dz_m, so it would give a linear model nothing new; its
    sign flips exactly where this angle crosses atan(1/w). The ratio is what lets a
    threshold learned on a 30 cm box transfer to a 15 cm L.

    Deliberately normal-free. Fingertip slip depends on the angle between this ray
    and the surface normal instead, but it never occurs at the simulated finger
    friction (2.0), and on the flat-topped training objects the two angles are
    identical, so the data cannot tell them apart. A normal-relative version
    penalized the curved-top flashlight, where every contact passes (see
    MASS_FORCE_RESULTS.md).
    """
    return float(np.arctan2(dx, dz))


def extract_features(candidate, geometry: dict) -> dict[str, float]:
    lo, hi = np.asarray(geometry['bounds'])
    p = np.asarray(candidate.position)
    delta = p - geometry['pivot']
    values = [*candidate.object_position, *candidate.normal, (p[2] - lo[2]) / (hi[2] - lo[2]),
              delta[0], delta[2], np.linalg.norm(delta[[0, 2]]), *(hi - lo),
              candidate.approach_error_m, candidate.joint_margin_rad,
              pivot_ray_angle(delta[0], delta[2])]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite candidate features')
    return dict(zip(FEATURE_NAMES, map(float, values), strict=True))
