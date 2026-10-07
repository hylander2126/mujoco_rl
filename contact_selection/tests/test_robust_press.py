import numpy as np
import pytest

from contact_selection.selection.robust_press import rank_positions, select_press


def box_cloud():
    axis = np.linspace(-0.05, 0.05, 51)
    x, y = np.meshgrid(axis, axis)
    top = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.2)]
    bottom = top.copy(); bottom[:, 2] = 0
    cloud = np.r_[top, bottom]
    normals = np.tile([0., 0., 1.], (len(cloud), 1))
    normals[len(top):] *= -1
    return cloud, normals


def test_scale_invariance_and_com_lateral_alignment():
    points = np.array([[0.01, 0.03, 0.2], [0.011, 0., 0.2], [0.1, 0., 0.3]])
    for scale in [0.1, 1, 10]:
        assert rank_positions(points * scale, [0, 0, 0], [0, 0]) == 1
    assert rank_positions(points, [0, 0, 0], [0, 0.03]) == 0


def test_patch_rejects_silhouette_and_isolated_high_outlier():
    cloud, normals = box_cloud()
    cloud = np.r_[cloud, [[-0.049, 0, 0.23]]]
    normals = np.r_[normals, [[0., 0., 1.]]]
    pick = select_press(cloud, normals=normals, table_z=0, com_xy=[0, 0])
    assert pick['available']
    assert pick['point'][2] == pytest.approx(0.2)
    assert -0.047 < pick['point'][0] < -0.03
    assert abs(pick['point'][1]) < 0.006


def test_top_normal_orientation_does_not_depend_on_global_centroid():
    cloud, normals = box_cloud()
    # Model a low exposed ledge alongside a tall structure. Provided inverted
    # normals emulate the legacy centroid-orientation failure on concave objects.
    normals *= -1
    pick = select_press(cloud, normals=normals, table_z=0, com_xy=[0, 0])
    assert pick['available'] and pick['normal'][2] > 0.99


def test_missing_support_abstains():
    cloud, normals = box_cloud()
    ids = cloud[:, 2] > 0.1
    pick = select_press(cloud[ids], normals=normals[ids], table_z=0, com_xy=[0, 0])
    assert not pick['available'] and pick['reason'] == 'support_not_observed'


def test_hole_is_not_filled_and_input_normals_not_mutated():
    cloud, normals = box_cloud()
    ids = ~((np.abs(cloud[:, 0] + 0.04) < 0.009) & (np.abs(cloud[:, 1]) < 0.01) & (cloud[:, 2] > 0.1))
    cloud, normals = cloud[ids], normals[ids]
    before = normals.copy()
    pick = select_press(cloud, normals=normals, table_z=0, com_xy=[0, 0])
    assert pick['available']
    assert not (abs(pick['point'][0] + 0.04) < 0.009 and abs(pick['point'][1]) < 0.01)
    np.testing.assert_array_equal(normals, before)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        rank_positions([[0, 0, float('nan')]], [0, 0, 0], [0, 0])
    cloud, _ = box_cloud()
    with pytest.raises(ValueError):
        select_press(cloud, table_z=0, com_xy=[0, 0], patch_radius=-1)


def test_validator_rejects_first_proposal_without_using_outcomes():
    cloud, normals = box_cloud()
    attempts = []
    def validator(point, normal):
        attempts.append(point)
        return len(attempts) > 1
    pick = select_press(cloud, normals=normals, table_z=0, com_xy=[0, 0], validator=validator)
    assert pick['available'] and pick['validation_rejections'] == 1
    assert not pick['requires_robot_validation']
    assert not np.array_equal(attempts[0], pick['point'])
