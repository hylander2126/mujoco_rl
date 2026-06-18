#!/usr/bin/env python3
"""Open generated MuJoCo environments for quick XML iteration.

Usage examples:
    python3 util/visualize_robot.py
    python3 util/visualize_robot.py --cube-color blue --frame site
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FRAME_MODE_NAMES = ("none", "body", "geom", "site", "camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the exact HW1 VLA environment used by collection/eval."
    )
    parser.add_argument(
        "--cube-color",
        choices=["red", "blue"],
        default="red",
        help="Cube color for the HW1 bin-sort scene.",
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
    from environment.scene import create_hw1_scene_xml

    frame_modes = {
        "none": mujoco.mjtFrame.mjFRAME_NONE,
        "body": mujoco.mjtFrame.mjFRAME_BODY,
        "geom": mujoco.mjtFrame.mjFRAME_GEOM,
        "site": mujoco.mjtFrame.mjFRAME_SITE,
        "camera": mujoco.mjtFrame.mjFRAME_CAMERA,
    }

    xml_path = create_hw1_scene_xml()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    _prepare_hw1_binsort_preview(mujoco, model, data, cube_color=args.cube_color)

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.opt.frame = frame_modes[args.frame]
        _use_debug_viewer_camera(mujoco, viewer)
        print(f"Viewer running HW1 VLA scene from generated XML: {xml_path}")
        print("Close the MuJoCo viewer window to exit.")

        while viewer.is_running():
            if not args.no_step:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def _use_debug_viewer_camera(mujoco, viewer) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = np.array([0.50, 0.0, 0.30])
    viewer.cam.distance = 1.85
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -35.0


def _prepare_hw1_binsort_preview(mujoco, model, data, cube_color: str) -> None:
    """Mirror the key reset-time visual state used by VLAIRB120Env."""
    from task import HW1_TASK

    mujoco.mj_forward(model, data)

    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "sort_cube_free")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sort_cube_geom")
    tray_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tray_geom")
    tray_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site:tray_center")

    if cube_geom_id >= 0:
        model.geom_rgba[cube_geom_id] = HW1_TASK.cube_rgba[cube_color]

    if cube_joint_id >= 0 and cube_geom_id >= 0 and tray_geom_id >= 0 and tray_site_id >= 0:
        qpos_adr = int(model.jnt_qposadr[cube_joint_id])
        qvel_adr = int(model.jnt_dofadr[cube_joint_id])
        tray_pos = data.site_xpos[tray_site_id].copy()
        tray_mat = data.site_xmat[tray_site_id].reshape(3, 3).copy()
        tray_normal = tray_mat[:, 2]
        cube_half_extent = float(np.max(model.geom_size[cube_geom_id]))
        tray_half_thickness = float(model.geom_size[tray_geom_id][2])
        cube_pos = tray_pos + tray_normal * (cube_half_extent + tray_half_thickness + 0.004)

        data.qpos[qpos_adr : qpos_adr + 3] = cube_pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    main()
