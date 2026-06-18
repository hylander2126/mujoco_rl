from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

import mujoco

from task import BinSortTaskSpec, HW1_TASK


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "robot" / "assets"
GENERATED_SCENE_PATH = Path(gettempdir()) / "mujoco_irb120_hw1_binsort.xml"


def create_hw1_scene_xml(
    task: BinSortTaskSpec = HW1_TASK,
    template_path: Path = ASSETS_DIR / "scene_template.xml",
    out: Path = GENERATED_SCENE_PATH,
) -> str:
    object_block = _hw1_world_block(task)

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            template.format(
                robot_mesh_dir=(ASSETS_DIR / "robot").as_posix(),
                object_block=object_block,
            )
        )
    return str(out)


def load_hw1_scene(task: BinSortTaskSpec = HW1_TASK):
    xml_path = create_hw1_scene_xml(task=task)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def _hw1_world_block(task: BinSortTaskSpec) -> str:
    red_x, red_y = task.bin_xy_by_color["red"]
    blue_x, blue_y = task.bin_xy_by_color["blue"]
    robot_xml = (ASSETS_DIR / "robot" / "robot.xml").as_posix()
    sort_cube_xml = (ASSETS_DIR / "objects" / "sort_cube.xml").as_posix()

    return f"""
        <include file="{robot_xml}"> </include>

        <camera name="{task.camera_name}" mode="fixed"
                pos="0.750 -1.100 1.100"
                xyaxes="0.965 0.263 0.000 -0.157 0.577 0.802"
                fovy="55"/>

        <body name="red_bin" pos="{red_x} {red_y} 0.055">
            <site name="site:red_bin" pos="0 0 0.045" size="0.015" rgba="1 0 0 1"/>
            <geom name="red_bin_floor" type="box" size="0.12 0.12 0.01" rgba="0.95 0.05 0.05 0.55" contype="1" conaffinity="1"/>
            <geom name="red_bin_wall_xp" type="box" pos="0.12 0 0.05" size="0.01 0.13 0.05" rgba="0.95 0.05 0.05 0.75" contype="1" conaffinity="1"/>
            <geom name="red_bin_wall_xn" type="box" pos="-0.12 0 0.05" size="0.01 0.13 0.05" rgba="0.95 0.05 0.05 0.75" contype="1" conaffinity="1"/>
            <geom name="red_bin_wall_yp" type="box" pos="0 0.12 0.05" size="0.13 0.01 0.05" rgba="0.95 0.05 0.05 0.75" contype="1" conaffinity="1"/>
            <geom name="red_bin_wall_yn" type="box" pos="0 -0.12 0.05" size="0.13 0.01 0.05" rgba="0.95 0.05 0.05 0.75" contype="1" conaffinity="1"/>
        </body>

        <body name="blue_bin" pos="{blue_x} {blue_y} 0.055">
            <site name="site:blue_bin" pos="0 0 0.045" size="0.015" rgba="0 0.2 1 1"/>
            <geom name="blue_bin_floor" type="box" size="0.12 0.12 0.01" rgba="0.05 0.2 0.95 0.55" contype="1" conaffinity="1"/>
            <geom name="blue_bin_wall_xp" type="box" pos="0.12 0 0.05" size="0.01 0.13 0.05" rgba="0.05 0.2 0.95 0.75" contype="1" conaffinity="1"/>
            <geom name="blue_bin_wall_xn" type="box" pos="-0.12 0 0.05" size="0.01 0.13 0.05" rgba="0.05 0.2 0.95 0.75" contype="1" conaffinity="1"/>
            <geom name="blue_bin_wall_yp" type="box" pos="0 0.12 0.05" size="0.13 0.01 0.05" rgba="0.05 0.2 0.95 0.75" contype="1" conaffinity="1"/>
            <geom name="blue_bin_wall_yn" type="box" pos="0 -0.12 0.05" size="0.13 0.01 0.05" rgba="0.05 0.2 0.95 0.75" contype="1" conaffinity="1"/>
        </body>

        <include file="{sort_cube_xml}"/>
"""
