"""Geometry-only press selection with local surface evidence and planar CoM.

No outcome labels or mass/friction are used. The ratio dx/dz is scale invariant;
local patch coverage avoids choosing a single noisy silhouette point. This is
a press-only replacement; legacy planar-push/forward-tip behavior is untouched.
"""
import numpy as np
from scipy.spatial import cKDTree

from contact_selection.hardware.hardware_selector import estimate_normals, estimate_pivot


def rank_positions(points, pivot, com_xy, direction=(-1., 0., 0.), *, ratio_tolerance=0.025):
    points, pivot, com_xy = map(lambda x: np.asarray(x, dtype=float), (points, pivot, com_xy))
    direction = np.array(direction, dtype=float, copy=True)
    if (points.ndim != 2 or points.shape[1] != 3 or pivot.shape != (3,) or com_xy.shape != (2,)
            or direction.shape != (3,) or abs(direction[2]) > 1e-9
            or not all(np.isfinite(x).all() for x in (points, pivot, com_xy, direction))
            or np.linalg.norm(direction) < 1e-9 or not np.isfinite(ratio_tolerance) or ratio_tolerance < 0):
        raise ValueError('Expected finite points, pivot, planar CoM and horizontal direction')
    direction /= np.linalg.norm(direction)
    axis = np.cross([0., 0., 1.], direction)
    dz = points[:, 2] - pivot[2]
    dx = -(points - pivot) @ direction
    valid = np.flatnonzero((dz > 0) & (dx >= 0))
    if not len(valid):
        return None
    ratio = dx[valid] / dz[valid]
    band = valid[ratio <= ratio.min() + ratio_tolerance]
    lateral = np.abs((points[band, :2] - com_xy) @ axis[:2])
    return int(band[np.lexsort((-dz[band], dx[band] / dz[band], lateral))[0]])


def select_press(cloud, *, table_z, com_xy, normals=None, direction=(-1., 0., 0.),
                 finger_radius=0.01325, patch_radius=0.004, normal_radius=0.015,
                 min_normal_z=0.95, noise_tolerance=0.003, validator=None):
    """Return a sensed surface point, or an explicit abstention with diagnostics.

    Assumes segmented registered Z-up clouds in metres and observed support.
    A validator(point, normal) callback can check full-tool IK/collision. Failed
    candidates are skipped before motion; outcomes must not enter this callback.
    Normals on vertically exposed patches are oriented upward, without a global
    convex-centroid assumption. Unknown/occluded patches are never filled in.
    """
    points = np.asarray(cloud, dtype=float)
    params = [table_z, finger_radius, patch_radius, normal_radius, min_normal_z, noise_tolerance]
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(params).all()
            or min(finger_radius, patch_radius, normal_radius) <= 0 or noise_tolerance < 0
            or not 0 < min_normal_z <= 1):
        raise ValueError('Invalid cloud or selector parameters')
    finite = np.isfinite(points).all(axis=1)
    supplied = None
    if normals is not None:
        supplied = np.asarray(normals, dtype=float)
        if supplied.shape != points.shape:
            raise ValueError('Normals must match cloud')
        supplied = supplied[finite].copy()
    points, unique = np.unique(points[finite], axis=0, return_index=True)
    if len(points) < 12:
        return dict(available=False, reason='insufficient_observation')
    try:
        geometry = estimate_pivot(points, direction, table_z, support_band=0.006)
    except ValueError:
        return dict(available=False, reason='support_not_observed')
    n = estimate_normals(points, radius=normal_radius) if supplied is None else supplied[unique]
    lengths = np.linalg.norm(n, axis=1)
    good = np.isfinite(n).all(axis=1) & (lengths > 1e-9)
    n[good] /= lengths[good, None]
    n[n[:, 2] < 0] *= -1  # only exposed upper patches survive below
    tree = cKDTree(points[:, :2])
    eligible = []
    for i in np.flatnonzero(good & (n[:, 2] >= min_normal_z)
                           & (points[:, 2] > table_z + finger_radius + noise_tolerance)):
        ids = np.asarray(tree.query_ball_point(points[i, :2], patch_radius * 2))
        delta = points[ids] - points[i]
        if np.any((np.linalg.norm(delta[:, :2], axis=1) < patch_radius)
                  & (delta[:, 2] > noise_tolerance + patch_radius * 0.5)):
            continue  # covered surface or low depth outlier
        # Same tangent patch must surround the point in every quadrant; no hull
        # bridging a hole, a notch or a narrow disconnected cap.
        radius = np.linalg.norm(delta[:, :2], axis=1)
        near = (np.abs(delta @ n[i]) <= noise_tolerance) & (radius >= patch_radius) & (radius <= 2 * patch_radius)
        angles = np.arctan2(delta[near, 1], delta[near, 0])
        sectors = np.unique(np.floor((angles + np.pi) / (np.pi / 2)).astype(int) % 4)
        if len(sectors) == 4:
            eligible.append(i)
    if not eligible:
        return dict(available=False, reason='no_observed_upper_contact_patch', geometry=geometry)
    ids = np.asarray(eligible)
    rejected = 0
    while len(ids):
        pick = rank_positions(points[ids], geometry['pivot'], com_xy, geometry['direction'])
        if pick is None:
            break
        i = ids[pick]
        if validator is None or validator(points[i].copy(), n[i].copy()):
            return dict(available=True, point=points[i].copy(), normal=n[i].copy(),
                        ball_center=points[i] + finger_radius * n[i], geometry=geometry,
                        candidate_count=len(eligible), validation_rejections=rejected,
                        requires_robot_validation=validator is None)
        rejected += 1
        ids = np.delete(ids, pick)
    return dict(available=False, reason='no_validated_contact' if rejected else 'no_inboard_contact',
                geometry=geometry, validation_rejections=rejected)
