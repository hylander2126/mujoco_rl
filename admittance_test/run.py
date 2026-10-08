from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if "--show-viewer" in sys.argv else "egl")

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parameter_estimation.controllers.press_pull_fsm import PressPullConfig, PressPullFSM
from parameter_estimation.scene import load_environment

from admittance_test.controller import AdmittanceConfig, AdmittanceController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IRB120 admittance-control smoke test.")
    parser.add_argument("--steps", type=int, default=3000, help="Simulation steps.")
    parser.add_argument("--target-force", type=float, default=2.0, help="Target +X force in N.")
    parser.add_argument("--save", type=Path, help="Optional .npz trace output path.")
    parser.add_argument("--show-viewer", action="store_true",
                        help="Open an interactive MuJoCo window; close it to stop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    model, data = load_environment(0)
    model.opt.timestep = 0.001

    irb_setup = PressPullFSM(
        AdmittanceController(model, data).irb,
        model,
        data,
        PressPullConfig(approach_clearance_m=0.001, verbose=False),
    )
    irb_setup.move_to_pre_squash()
    mujoco.mj_forward(model, data)

    controller = AdmittanceController(
        model,
        data,
        AdmittanceConfig(target_force_n=args.target_force),
    )
    controller.reset()

    trace = []

    def step_once() -> dict[str, np.ndarray | float]:
        sample = controller.step()
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Simulation became non-finite")
        return sample

    if args.show_viewer:
        from mujoco import viewer

        def on_key(key: int) -> None:
            if key == ord(" "):
                controller.set_target_force(0.0)
            elif key == mujoco.mjtKey.mjKEY_UP:
                controller.set_target_force(controller.config.target_force_n + 0.25)
            elif key == mujoco.mjtKey.mjKEY_DOWN:
                controller.set_target_force(max(0.0, controller.config.target_force_n - 0.25))

        with viewer.launch_passive(model, data, key_callback=on_key) as handle:
            while handle.is_running():
                trace.append(step_once())
                handle.sync()
    else:
        for _ in range(args.steps):
            trace.append(step_once())

    force = np.asarray([sample["force_world"] for sample in trace])
    position = np.asarray([sample["tool_position"] for sample in trace])
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, time=data.time, force_world=force, tool_position=position)

    print(json.dumps({
        "steps": len(trace),
        "sim_time_s": float(data.time),
        "initial_x_m": float(position[0, 0]),
        "final_x_m": float(position[-1, 0]),
        "peak_abs_force_x_n": float(np.max(np.abs(force[:, 0]))),
    }, indent=2))


if __name__ == "__main__":
    main()
