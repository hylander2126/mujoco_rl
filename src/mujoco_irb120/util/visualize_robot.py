#!/usr/bin/env python3
"""Open generated MuJoCo environments for quick XML iteration.

Usage examples:
    python3 src/mujoco_irb120/util/visualize_robot.py
    python3 src/mujoco_irb120/util/visualize_robot.py --cube-color blue --frame site
    python3 src/mujoco_irb120/util/visualize_robot.py --scene legacy-object --object-id 14

If you run from inside `src/mujoco_irb120`, this also works:
    python3 util/visualize_robot.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


FRAME_MODE_NAMES = ("none", "body", "geom", "site", "camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the exact generated environment used by VLA collection/eval."
    )
    parser.add_argument(
        "--scene",
        choices=["hw1-bin-sort", "legacy-object"],
        default="hw1-bin-sort",
        help="Scene to visualize. Default: HW1 VLA bin-sort.",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=0,
        help="Legacy object id. Only used with --scene legacy-object.",
    )
    parser.add_argument(
        "--cube-color",
        choices=["red", "blue"],
        default="red",
        help="Cube color for the HW1 bin-sort scene.",
    )
    parser.add_argument(
        "--controller-type",
        choices=["position", "velocity"],
        default="position",
        help="Actuator/controller block to inject into the generated scene.",
    )
    parser.add_argument(
        "--frame",
        choices=FRAME_MODE_NAMES,
        default="none",
        help="MuJoCo frame overlay to show in the viewer.",
    )
    parser.add_argument(
        "--no-step",
        action="store_true",
        help="Hold physics still while the viewer is open.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import mujoco
    import mujoco.viewer
    from mujoco_irb120.util.load_obj_in_env import create_scene_xml, create_vla_binsort_scene_xml

    frame_modes = {
        "none": mujoco.mjtFrame.mjFRAME_NONE,
        "body": mujoco.mjtFrame.mjFRAME_BODY,
        "geom": mujoco.mjtFrame.mjFRAME_GEOM,
        "site": mujoco.mjtFrame.mjFRAME_SITE,
        "camera": mujoco.mjtFrame.mjFRAME_CAMERA,
    }

    if args.scene == "hw1-bin-sort":
        xml_path = create_vla_binsort_scene_xml(controller_type=args.controller_type)
    else:
        xml_path = create_scene_xml(
            object_id=args.object_id,
            controller_type=args.controller_type,
        )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    if args.scene == "hw1-bin-sort":
        _prepare_hw1_binsort_preview(mujoco, model, data, cube_color=args.cube_color)

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.opt.frame = frame_modes[args.frame]
        if args.scene == "hw1-bin-sort":
            _use_fixed_viewer_camera(mujoco, model, viewer, "vla_cam")
        print(f"Viewer running scene={args.scene} from generated XML: {xml_path}")
        print("Close the MuJoCo viewer window to exit.")

        while viewer.is_running():
            if not args.no_step:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def _use_fixed_viewer_camera(mujoco, model, viewer, camera_name: str) -> None:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        print(f"Could not find camera {camera_name!r}; viewer will use free camera.")
        return
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def _prepare_hw1_binsort_preview(mujoco, model, data, cube_color: str) -> None:
    """Mirror the key reset-time visual state used by VLAIRB120Env."""
    rgba_by_color = {
        "red": np.array([1.0, 0.05, 0.05, 1.0]),
        "blue": np.array([0.05, 0.2, 1.0, 1.0]),
    }
    mujoco.mj_forward(model, data)

    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "sort_cube_free")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sort_cube_geom")
    tray_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site:tray_center")

    if cube_geom_id >= 0:
        model.geom_rgba[cube_geom_id] = rgba_by_color[cube_color]

    if cube_joint_id >= 0 and tray_site_id >= 0:
        qpos_adr = int(model.jnt_qposadr[cube_joint_id])
        qvel_adr = int(model.jnt_dofadr[cube_joint_id])
        tray_pos = data.site_xpos[tray_site_id].copy()
        tray_mat = data.site_xmat[tray_site_id].reshape(3, 3).copy()
        tray_normal = tray_mat[:, 2]
        cube_pos = tray_pos + tray_normal * (0.025 + 0.015 + 0.004)

        data.qpos[qpos_adr : qpos_adr + 3] = cube_pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    main()
