"""Interpretable pre-execution features; no outcome or ground-truth mass input."""
import numpy as np

FEATURE_NAMES = ['local_x_m', 'local_y_m', 'local_z_m', 'normal_x', 'normal_y', 'normal_z',
                 'normalized_height', 'pivot_dx_m', 'pivot_dz_m', 'arc_radius_m',
                 'width_m', 'depth_m', 'height_m', 'approach_error_m', 'joint_margin_rad']


def extract_features(candidate, geometry: dict) -> dict[str, float]:
    lo, hi = np.asarray(geometry['bounds'])
    p = np.asarray(candidate.position)
    delta = p - geometry['pivot']
    values = [*candidate.object_position, *candidate.normal, (p[2] - lo[2]) / (hi[2] - lo[2]),
              delta[0], delta[2], np.linalg.norm(delta[[0, 2]]), *(hi - lo),
              candidate.approach_error_m, candidate.joint_margin_rad]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite candidate features')
    return dict(zip(FEATURE_NAMES, map(float, values), strict=True))
