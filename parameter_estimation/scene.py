"""MuJoCo scenes used by the parameter-estimation experiments."""

from pathlib import Path
from tempfile import gettempdir
import xml.etree.ElementTree as ET

import mujoco


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
    asset_block = ""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = template.replace('<compiler angle="radian" meshdir="."/>',
                                f'<compiler angle="radian" meshdir="{SHARED_ASSETS.as_posix()}"/>')
    xml = template.format(actuator_block=_actuators(), asset_block=asset_block,
                          object_block=f"{robot}\n{objects}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    return str(out)


def load_environment(num=0, launch_viewer=False):
    model = mujoco.MjModel.from_xml_path(create_scene_xml((num,)))
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
