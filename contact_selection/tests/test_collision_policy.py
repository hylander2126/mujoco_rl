"""Adapter exclusion changes only the intended collision pairs and survives saving."""
import mujoco
import numpy as np
import pytest

from contact_selection.scene import disable_adapter_object_collisions, load_environment


def eligibility(model):
    directed = (model.geom_contype[:, None] & model.geom_conaffinity[None, :]) != 0
    return directed | directed.T


@pytest.mark.parametrize('object_id', [0, 10, 11, 12, 13, 14])
def test_only_adapter_payload_pairs_are_disabled(object_id, tmp_path):
    model, _ = load_environment(object_id, adapter_object_collisions=True)
    expected = eligibility(model)
    adapter = np.flatnonzero(model.geom_bodyid == model.body('ft_and_adapter_link').id)
    payload = np.flatnonzero(model.geom_bodyid == model.site_bodyid[model.site('site:obj_frame').id])
    assert expected[np.ix_(adapter, payload)].any()
    expected[np.ix_(adapter, payload)] = False
    expected[np.ix_(payload, adapter)] = False
    disable_adapter_object_collisions(model)
    np.testing.assert_array_equal(eligibility(model), expected)
    ball, table = model.geom('push_ball_col').id, model.geom('table').id
    assert expected[ball, payload].all()
    assert expected[table, payload].all()
    assert expected[adapter, table].all()
    masks = model.geom_contype.copy(), model.geom_conaffinity.copy()
    disable_adapter_object_collisions(model)
    np.testing.assert_array_equal(model.geom_contype, masks[0])
    np.testing.assert_array_equal(model.geom_conaffinity, masks[1])
    for body in range(model.nbody):
        geoms = model.geom_bodyid == body
        assert model.body_contype[body] == np.bitwise_or.reduce(model.geom_contype[geoms], initial=0)
        assert model.body_conaffinity[body] == np.bitwise_or.reduce(model.geom_conaffinity[geoms], initial=0)
    path = tmp_path / 'scene.mjb'
    mujoco.mj_saveModel(model, str(path))
    loaded = mujoco.MjModel.from_binary_path(str(path))
    np.testing.assert_array_equal(eligibility(loaded), expected)
    default, _ = load_environment(object_id)
    np.testing.assert_array_equal(eligibility(default), expected)
