import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from contact_selection.sim.off_axis import arc_diagnostics


@pytest.mark.parametrize('yaw', [-4, 0, 4])
def test_signed_yaw_and_magnitude_keep_opposite_offsets_distinct(yaw):
    # Arbitrary initial pose must cancel; pure intended rotation contributes zero.
    initial = Rotation.from_euler('xyz', [10, -20, 30], degrees=True).as_matrix()
    matrices = Rotation.from_rotvec(np.deg2rad([[0, 0, 0], [0, -10, yaw]])).as_matrix() @ initial
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, :3] = matrices
    result = arc_diagnostics({'state_id_hist': np.array([3, 3]), 'obj_pose_hist': poses})
    assert result['peak_off_axis_deg'] == pytest.approx(abs(yaw), abs=1e-12)
    assert result['signed_yaw_at_peak_deg'] == pytest.approx(yaw, abs=1e-12)


def test_no_arc_is_unavailable_not_zero_rotation():
    assert arc_diagnostics({'state_id_hist': np.array([1, 2])}) is None
