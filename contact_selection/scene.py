"""Contact-selection scene policy layered on the shared XML/assets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import mujoco
import numpy as np
from parameter_estimation.scene import OBJECTS, create_scene_xml


def disable_adapter_object_collisions(model) -> None:
    """Remove only adapter/payload pairs, including on a saved compiled model.

    Give each adapter geom a private collision bit and copy its existing
    permitted partners to that bit, excluding payload geoms. Other pairs keep
    their original eligibility. Updating body masks keeps broadphase in sync.
    """
    adapter = model.body('ft_and_adapter_link').id
    payload = int(model.site_bodyid[model.site('site:obj_frame').id])
    adapters = np.flatnonzero(model.geom_bodyid == adapter)
    objects = np.flatnonzero(model.geom_bodyid == payload)
    types = model.geom_contype.copy()
    affinities = model.geom_conaffinity.copy()
    allowed = ((types[:, None] & affinities[None, :]) != 0)
    allowed |= allowed.T.copy()
    # Explicit pairs bypass bitmask filtering; fail rather than silently miss one.
    for g0, g1 in zip(model.pair_geom1, model.pair_geom2):
        if (g0 in adapters and g1 in objects) or (g1 in adapters and g0 in objects):
            raise ValueError('Explicit adapter/object contact pair must be removed from the scene')
    if not allowed[np.ix_(adapters, objects)].any():
        return
    used = int(np.bitwise_or.reduce(types | affinities, initial=0))
    free_bits = [1 << i for i in range(30) if not used & (1 << i)]
    if len(free_bits) < len(adapters):
        raise ValueError('Not enough free collision bits for adapter exclusion')
    model.geom_contype[adapters] = 0
    model.geom_conaffinity[adapters] = 0
    for gid, bit in zip(adapters, free_bits):
        model.geom_contype[gid] = bit
        partners = allowed[gid].copy()
        partners[objects] = False
        model.geom_conaffinity[partners] |= bit
    for bid in range(model.nbody):
        geoms = model.geom_bodyid == bid
        model.body_contype[bid] = np.bitwise_or.reduce(model.geom_contype[geoms], initial=0)
        model.body_conaffinity[bid] = np.bitwise_or.reduce(model.geom_conaffinity[geoms], initial=0)


def load_environment(num=0, launch_viewer=False, *, adapter_object_collisions=False):
    with TemporaryDirectory(prefix='contact-selection-scene-') as directory:
        xml = create_scene_xml((num,), out=Path(directory) / 'scene.xml')
        model = mujoco.MjModel.from_xml_path(xml)
    if not adapter_object_collisions:
        disable_adapter_object_collisions(model)
    data = mujoco.MjData(model)
    if launch_viewer:
        from mujoco import viewer as mujoco_viewer
        mujoco_viewer.launch(model, data)
    return model, data
