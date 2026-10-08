from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if "--show-viewer" in sys.argv else "egl")

if "--show-viewer" in sys.argv and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
    raise SystemExit(
        "--show-viewer needs a graphical display; reconnect with SSH X11 forwarding "
        "(ssh -Y host) or run without --show-viewer."
    )

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parameter_estimation.scene import create_scene_xml

from admittance_test.controller import GravityCompController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IRB120 free-space gravity-compensation test.")
    parser.add_argument("--steps", type=int, default=3000, help="Headless simulation steps.")
    parser.add_argument("--save", type=Path, help="Optional .npz diagnostic output path.")
    parser.add_argument("--show-viewer", action="store_true",
                        help="Open an interactive MuJoCo window; close it to stop.")
    return parser.parse_args()


def load_robot_only() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the existing scene template without table or payload bodies."""
    source = Path(tempfile.gettempdir()) / "mujoco_irb120_admittance_source.xml"
    output = Path(tempfile.gettempdir()) / "mujoco_irb120_admittance_robot.xml"
    create_scene_xml((0,), out=source)
    tree = ET.parse(source)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    for body in list(worldbody):
        if body.tag == "body" and body.get("name") in {"table0", "payload"}:
            worldbody.remove(body)
    for geom in list(worldbody):
        if geom.tag == "geom" and geom.get("name") == "floor":
            worldbody.remove(geom)

    actuator = root.find("actuator")
    for child in list(actuator):
        actuator.remove(child)
    for index in range(1, 7):
        ET.SubElement(actuator, "motor", {
            "name": f"joint_{index}",
            "joint": f"joint_{index}",
            "gear": "1",
        })

    tree.write(output, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output))
    return model, mujoco.MjData(model)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    model, data = load_robot_only()
    controller = GravityCompController(model, data)
    trace = []

    def step_once() -> dict[str, np.ndarray | float]:
        sample = controller.step()
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Simulation became non-finite")
        return sample

    if args.show_viewer:
        from mujoco import viewer

        with viewer.launch_passive(model, data) as handle:
            while handle.is_running():
                trace.append(step_once())
                handle.sync()
    else:
        for _ in range(args.steps):
            trace.append(step_once())

    positions = np.asarray([sample["tool_position"] for sample in trace])
    torques = np.asarray([sample["joint_torque"] for sample in trace])
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, tool_position=positions, joint_torque=torques)

    print(json.dumps({
        "steps": len(trace),
        "sim_time_s": float(data.time),
        "tool_position_m": positions[-1].tolist(),
        "peak_abs_gravity_torque_nm": float(np.max(np.abs(torques))),
    }, indent=2))


if __name__ == "__main__":
    main()
