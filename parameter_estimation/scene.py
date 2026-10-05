"""MuJoCo scenes used by the parameter-estimation experiments."""

from pathlib import Path
from tempfile import gettempdir
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_ASSETS = REPO_ROOT / "mujoco_irb120" / "robot" / "assets"
ROBOT_ASSETS = SHARED_ASSETS / "robot"
OBJECT_ASSETS = SHARED_ASSETS / "objects"
TEMPLATE_PATH = Path(__file__).with_name("scene_template.xml")
GENERATED_SCENE_PATH = Path(gettempdir()) / "mujoco_irb120_parameter_estimation.xml"

OBJECTS = {
    0: "box",
    10: "heart",
    11: "L",
    12: "monitor",
    13: "soda",
    14: "flashlight",
}


def _children_xml(path: Path) -> str:
    root = ET.parse(path).getroot()
    return "\n".join(ET.tostring(child, encoding="unicode") for child in root)


def _actuators() -> str:
    gains = [(200, 100)] * 3 + [(100, 50)] * 3
    ranges = ["-2.87979 2.87979", "-1.91986 1.91986", "-1.22173 1.91986",
              "-2.79252 2.79252", "-2.09440 2.09440", "-3.142 3.142"]
    entries = [
        f'<position name="joint_{i}" joint="joint_{i}" kp="{kp}" kv="{kv}" '
        f'ctrlrange="{limit}"/>'
        for i, ((kp, kv), limit) in enumerate(zip(gains, ranges), 1)
    ]
    return (
        "<actuator>" + "".join(entries) + "</actuator>\n<sensor>"
        '<force name="force_sensor" site="site:sensor"/>'
        '<torque name="torque_sensor" site="site:sensor"/></sensor>'
    )


def create_scene_xml(object_ids=(0,), out: Path = GENERATED_SCENE_PATH) -> str:
    names = [OBJECTS[number] for number in object_ids]
    robot = _children_xml(ROBOT_ASSETS / "robot.xml")
    objects = "\n".join(_children_xml(OBJECT_ASSETS / name / f"{name}_exp.xml") for name in names)
    asset_block = "<asset>" + _children_xml(ROBOT_ASSETS / "hardware_tool" / "meshes.xml") + "</asset>"
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = template.replace('<compiler angle="radian" meshdir="."/>',
                                f'<compiler angle="radian" meshdir="{SHARED_ASSETS.as_posix()}"/>')
    xml = template.format(actuator_block=_actuators(), asset_block=asset_block,
                          object_block=f"{robot}\n{objects}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    return str(out)


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
    model = mujoco.MjModel.from_xml_path(create_scene_xml((num,)))
    if not adapter_object_collisions:
        disable_adapter_object_collisions(model)
    data = mujoco.MjData(model)
    if launch_viewer:
        from mujoco import viewer as mujoco_viewer
        mujoco_viewer.launch(model, data)
    return model, data


def load_photoshoot():
    model = mujoco.MjModel.from_xml_path(create_scene_xml(tuple(OBJECTS)))
    data = mujoco.MjData(model)
    from mujoco import viewer as mujoco_viewer
    mujoco_viewer.launch(model, data)
    return model, data
