"""The sim port of the hardware press rule must rank exactly as the hardware code does."""
import numpy as np

from contact_selection.hardware.hardware_selector import _press_band, estimate_pivot, select_contact_points
from contact_selection.selection.heuristic import heuristic_index, heuristic_report


def _row(dx, dz, y=0.0, obj='box', label=True, index=0):
    return {'object': obj, 'split': 'test', 'robust_feasible': label,
            'candidate': {'index': index, 'press_offset_xy': [0.0, y]},
            'features': {'pivot_dx_m': dx, 'pivot_dz_m': dz}}


def test_score_matches_hardware_press_band():
    """dz - w*dx over sim features equals hardware's combined score about a -X pivot."""
    rng = np.random.default_rng(0)
    pivot = np.array([0.5, 0.0, 0.0])
    points = pivot + np.column_stack([rng.uniform(0.005, 0.1, 40), rng.uniform(-0.05, 0.05, 40),
                                      rng.uniform(0.1, 0.3, 40)])
    direction = np.array([-1., 0., 0.])
    geometry = {'pivot': pivot, 'axis': np.cross([0., 0., 1.], direction)}
    scores = np.cross(points - pivot, direction) @ geometry['axis']
    for weight in (0.0, 1.0, 3.0):
        band = _press_band(points, np.arange(40), scores, geometry, weight, 0.0)
        rows = [_row(*(p - pivot)[[0, 2]], index=i) for i, p in enumerate(points)]
        assert np.flatnonzero(band).tolist() == [heuristic_index(rows, weight, 0.0)]


def test_band_then_centre_then_edgeward():
    rows = [_row(0.05, 0.30, y=0.04, index=0),   # in band, off-centre
            _row(0.06, 0.301, y=0.0, index=1),   # in band, centred, less edgeward
            _row(0.05, 0.30, y=0.0, index=2),    # in band, centred, most edgeward
            _row(0.01, 0.10, y=0.0, index=3)]    # far below the band
    assert heuristic_index(rows) == 2


def test_report_never_abstains_and_has_no_brier():
    rows = [_row(0.01, 0.2, label=False, obj='monitor', index=0),
            _row(0.05, 0.2, label=False, obj='monitor', index=1)]
    scene = heuristic_report(rows)['scenes'][0]
    assert scene['selected_index'] == 0 and scene['selected_success'] is False
    assert scene['brier_score'] is None


def test_hardware_pipeline_on_box_cloud_presses_near_edge_top():
    """Dense box cloud: press picks the top, at the robot-side (-X) edge, centred in Y."""
    g = np.linspace(0, 1, 21)
    u, v = np.meshgrid(g, g)
    u, v = u.ravel(), v.ravel()
    lo, size = np.array([0.53, -0.05, 0.05]), np.array([0.1, 0.1, 0.3])
    faces = [np.column_stack([u, v, np.full_like(u, z)]) for z in (0, 1)]
    faces += [np.column_stack([np.full_like(u, x), u, v]) for x in (0, 1)]
    faces += [np.column_stack([u, np.full_like(u, y), v]) for y in (0, 1)]
    cloud = lo + np.concatenate(faces) * size
    pivot = estimate_pivot(cloud, [-1., 0., 0.], table_z=0.05)
    assert pivot['kind'] == 'extended_edge' and np.isclose(pivot['pivot'][0], 0.53)
    press = select_contact_points(cloud, table_z=0.05)['press']
    assert press['available']
    assert np.isclose(press['point'][2], 0.35) and press['point'][0] < 0.55 and abs(press['point'][1]) < 0.006
