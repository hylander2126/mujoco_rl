"""Synthetic sanity checks: zero noise is deterministic, noise and poor geometry widen estimates."""
from pathlib import Path

import numpy as np
import pytest

from uncertainty import contact_robustness as cr
from uncertainty import estimator_mc as mc

FT_WHITE = mc.SOURCES['ft_white']


@pytest.fixture(scope='module')
def synthetic():
    return mc.synthetic_nominal()


def test_exact_model_recovers_truth_with_zero_variance(synthetic):
    r = mc.monte_carlo(synthetic, mc.NoiseSpec().scaled(0), 3)
    np.testing.assert_allclose(r['stats']['mean'], [synthetic.truth[p] for p in mc.PARAMS], rtol=1e-6)
    assert max(r['stats']['std']) < 1e-9


def test_more_noise_more_uncertainty(synthetic):
    spec = mc.NoiseSpec().only(*FT_WHITE)
    lo = mc.monte_carlo(synthetic, spec, 40, seed=1)['stats']['std']
    hi = mc.monte_carlo(synthetic, spec.scaled(4), 40, seed=1)['stats']['std']
    assert np.all(np.array(hi) > 2 * np.array(lo))


def test_nls_covariance_matches_monte_carlo_for_white_noise(synthetic):
    r = mc.monte_carlo(synthetic, mc.NoiseSpec().only(*FT_WHITE), 150, seed=2, nls=10)
    ratio = np.array(r['nls']['std']) / np.array(r['stats']['std'])
    assert np.all((ratio > 0.7) & (ratio < 1.4))


def test_short_sweep_is_worse_conditioned():
    spec = mc.NoiseSpec().only(*FT_WHITE)
    short = mc.monte_carlo(mc.synthetic_nominal(max_tilt_deg=4), spec, 40, seed=3, nls=3)
    full = mc.monte_carlo(mc.synthetic_nominal(max_tilt_deg=15), spec, 40, seed=3, nls=3)
    assert short['stats']['std'][1] > 2 * full['stats']['std'][1]
    assert abs(short['stats']['corr'][0][1]) > abs(full['stats']['corr'][0][1])
    assert short['nls']['cond_arc'] > full['nls']['cond_arc']


def test_pivot_error_widens_estimates(synthetic):
    r = mc.monte_carlo(synthetic, mc.NoiseSpec().only('pivot_mm'), 40, seed=4)
    assert r['stats']['std'][1] > 1e-4


@pytest.fixture(scope='module')
def box_scene():
    return cr.build_scene(Path(__file__).parents[2] / 'contact_selection/config/box_mu_0p20.json')


def test_contact_zero_noise_matches_nominal(box_scene):
    r = cr.robustness(box_scene, cr.GeometryNoise().scaled(0), 3)
    assert all(c['p_feasible'] == float(c['nominal_feasible']) for c in r['candidates'])
    assert r['robust_pick'] == r['nominal_pick'] or r['candidates'][r['robust_pick']]['p_feasible'] == 1.0


def test_near_boundary_contacts_are_less_robust(box_scene):
    r = cr.robustness(box_scene, cr.GeometryNoise(), 150, seed=5)
    feas = [c for c in r['candidates'] if c['nominal_feasible']]
    near = [c['p_feasible'] for c in feas if c['boundary_slack_mm'] < 1.0]
    far = [c['p_feasible'] for c in feas if c['boundary_slack_mm'] > 5.0]
    assert near and far and np.mean(near) < np.mean(far)
    assert r['robust_pick_p_feasible'] >= r['nominal_pick_p_feasible']


def test_rolling_correction_inverts_rolling_model():
    x0, height, r = 0.006, 0.3, 0.01325
    theta = np.deg2rad(np.linspace(0, 18, 50))
    phi = theta - np.arctan2(x0 + r * theta, height) + np.arctan2(x0, height)
    np.testing.assert_allclose(mc.rolling_corrected_tilt(phi, x0, height, r), theta, atol=1e-9)
    slope = np.polyfit(theta, phi, 1)[0]
    assert abs(slope - (1 - r * height / (height ** 2 + x0 ** 2))) < 2e-3
