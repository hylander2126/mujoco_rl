#!/usr/bin/env python3
"""Run the press-and-pull (squash / arc / unarc) rollout and record it.

Simulation counterpart to the hardware experiment driven by
`irb120_ws/.../irb120_control/arc_static.py`. Produces an npz whose keys match
`shove_simulation.py`'s, plus the phase and arc channels the parameter
estimator needs for segmentation.

    PYTHONPATH=$PWD python parameter_estimation/scripts/press_pull_simulation.py --object 0
    PYTHONPATH=$PWD python parameter_estimation/scripts/press_pull_simulation.py --object 0 --show-viewer
    PYTHONPATH=$PWD python parameter_estimation/scripts/press_pull_simulation.py --object 0 --adaptive
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _viewer_requested(argv) -> bool:
    return "--show-viewer" in argv and "--no-viewer" not in argv


os.environ.setdefault("MUJOCO_GL", "glfw" if _viewer_requested(sys.argv[1:]) else "egl")

import json
import math

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_irb120.robot.controllers import robot as robot_controller
from parameter_estimation.controllers import PressPullConfig, PressPullFSM
from parameter_estimation.rendering import RendererViewerOpts
from parameter_estimation.scene import OBJECTS, load_environment

np.set_printoptions(precision=4, suppress=True, linewidth=120)

ROLLOUT_DIR = REPO_ROOT / "outputs" / "parameter_estimation" / "rollouts"
OBJECT_PARAMS_PATH = REPO_ROOT / "parameter_estimation" / "object_params.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", type=int, default=0,
                   help=f"Object id. One of {sorted(OBJECTS)} ({OBJECTS}). Default: 0 (box).")
    viewer = p.add_mutually_exclusive_group()
    viewer.add_argument("--show-viewer", dest="show_viewer", action="store_true",
                        help="Open the live MuJoCo viewer.")
    viewer.add_argument("--no-viewer", dest="show_viewer", action="store_false",
                        help="Run headless and write a video instead.")
    p.set_defaults(show_viewer=False)
    p.add_argument("--force-ref", type=float, default=5.0,
                   help="Squash force reference in N. Default: 5.0 (hardware default).")
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="Multiply all motion speeds. 1.0 = hardware speed (~60 s of sim "
                        "time per rollout). Raise only for quick looks, not for data you "
                        "intend to fit. Default: 1.0.")
    p.add_argument("--mu-table", type=float, default=0.5,
                   help="Object-table sliding friction, set at runtime. Default: 0.5. "
                        "NOT the 0.2 that shove_simulation.py uses -- that experiment wants "
                        "the object to slide, this one needs it to stay put and rotate. "
                        "Below ~0.26 the box slides instead of tipping no matter how hard "
                        "you press, because the tangential force needed to tip grows faster "
                        "with press force (0.160 N/N) than table friction does (0.141 N/N).")
    p.add_argument("--mu-object", type=float, default=None,
                   help="Override the payload geom's own sliding friction. The meshed "
                        "objects ship with 0.1, which is the lowest value in the system "
                        "and usually the one that decides whether the finger can drag the "
                        "object over. Default: leave the scene value alone.")
    p.add_argument("--press-offset-x", type=float, default=0.0,
                   help="Shift the press point along X from the top-face centre. Moving it "
                        "toward the tipping edge lowers the force needed to tip. Default: 0.")
    p.add_argument("--adaptive", action="store_true",
                   help="On slip, retry with the squash force scaled up (adaptive_press.py "
                        "behaviour) until it works or the ceiling is reached.")
    p.add_argument("--max-attempts", type=int, default=5,
                   help="Cap on adaptive retries. Default: 5.")
    p.add_argument("--output", type=Path, default=None,
                   help="npz output path. Defaults to outputs/parameter_estimation/rollouts/.")
    p.add_argument("--video-path", type=Path, default=None,
                   help="Video output path. Defaults alongside the npz.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-phase logging.")
    return p.parse_args()


def run_attempt(args, force_ref: float, record_video: bool):
    """One full press-and-pull sequence from a freshly reset scene."""
    model, data = load_environment(num=args.object, launch_viewer=False)
    model.geom_friction[model.geom("table").id, 0] = args.mu_table

    irb = robot_controller.controller(model, data)
    if args.mu_object is not None:
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == irb.payload_body_id:
                model.geom_friction[gid, 0] = args.mu_object

    cfg = PressPullConfig(
        force_ref_n=force_ref,
        speed_scale=args.speed_scale,
        adaptive_retry=args.adaptive,
        press_offset_xy=(args.press_offset_x, 0.0),
        verbose=not args.quiet,
    )
    fsm = PressPullFSM(irb, model, data, cfg)

    fsm.move_to_pre_squash()
    irb.ft_bias(n_samples=200)
    # ft_bias steps the sim while settling; restart the clock so phase timeouts
    # measure control time, not settling time.
    data.time = 0.0
    fsm._state_start_time = 0.0

    # Generous ceiling: the phase timeouts inside the FSM are the real limits.
    max_sim_time = (cfg.squash_timeout_sec + cfg.arc_timeout_sec
                    + cfg.unarc_timeout_sec + cfg.retract_duration_sec
                    + 4 * cfg.lull_wait_sec + 10.0)

    rv = RendererViewerOpts(model, data, vis=args.show_viewer, show_left_UI=True)
    with rv:
        while rv.viewer_is_running() and not fsm.done and data.time < max_sim_time:
            fsm.step()
            mujoco.mj_step(model, data)
            rv.sync()
            if record_video:
                rv.capture_frame_if_due(data)

    return fsm, rv, model, data


def main() -> int:
    args = parse_args()
    if args.object not in OBJECTS:
        raise SystemExit(f"--object must be one of {sorted(OBJECTS)}, got {args.object}")

    name = OBJECTS[args.object]
    print(f"MuJoCo GL backend: {os.environ['MUJOCO_GL']}")
    print(f"Object: [{args.object}] {name}   force_ref: {args.force_ref} N   "
          f"speed_scale: {args.speed_scale}")

    params = json.load(open(OBJECT_PARAMS_PATH))["objects"]
    gt = params.get(str(args.object))
    if gt is not None:
        com_gt = np.subtract(gt["com_gt_onshape"], gt["com_gt_offset"])
        print(f"Ground truth: mass={gt['mass_gt']} kg  com={com_gt} m"
              + (f"  theta*={gt['theta_star']:.3f} deg" if "theta_star" in gt else ""))
    else:
        print(f"No ground-truth entry for object {args.object} in object_params.json "
              "-- rollout will still record, but nothing can be scored against it.")

    force_ref = args.force_ref
    attempts = []
    fsm = rv = model = data = None

    for attempt in range(1, args.max_attempts + 1):
        print(f"\n=== attempt {attempt}  force_ref={force_ref:.2f} N ===")
        fsm, rv, model, data = run_attempt(args, force_ref, record_video=not args.show_viewer)
        # A rollout is only useful if the sequence finished AND the object
        # actually went over. A finger that slips across the top face finishes
        # every phase cleanly while teaching the estimator nothing.
        success = fsm.completed and fsm.tipped
        attempts.append({
            "attempt": attempt,
            "force_ref_n": force_ref,
            "completed": fsm.completed,
            "tipped": fsm.tipped,
            "max_tip_deg": fsm.max_tip_deg,
            "success": success,
            "abort_reason": fsm.abort_reason,
            "arc_exit_angle_deg": math.degrees(fsm.arc_exit_angle_rad),
            "arc_exit_reason": fsm.arc_exit_reason,
        })
        if success:
            status = "tipped"
        elif fsm.completed:
            status = f"slipped (object rotated {fsm.max_tip_deg:.2f} deg)"
        else:
            status = f"aborted ({fsm.abort_reason})"
        print(f"--- attempt {attempt}: {status}, sim time {data.time:.2f} s ---")

        if success or not args.adaptive:
            break
        next_ref = force_ref * fsm.cfg.force_scale_factor
        if next_ref > fsm.cfg.force_ref_max_n:
            print(f"Force ceiling {fsm.cfg.force_ref_max_n:.1f} N reached; stopping.")
            break
        force_ref = next_ref

    # --- report -----------------------------------------------------------
    print("\n=== attempt summary ===")
    print(f"  {'#':>2}  {'force_ref':>9}  {'verdict':>8}  {'obj rot':>8}  {'arc exit':>9}  reason")
    for a in attempts:
        exit_ang = a["arc_exit_angle_deg"]
        ang = f"{exit_ang:8.2f}d" if not math.isnan(exit_ang) else "      n/a"
        verdict = "TIPPED" if a["success"] else ("slipped" if a["completed"] else "abort")
        print(f"  {a['attempt']:>2}  {a['force_ref_n']:8.2f}N  {verdict:>8}  "
              f"{a['max_tip_deg']:7.2f}d  {ang}  "
              f"{a['arc_exit_reason'] or a['abort_reason'] or ''}")

    if args.adaptive and len(attempts) > 1:
        # The force ladder is a measurement in its own right: the lowest normal
        # force that carried the object over bounds the friction and the
        # restoring moment, and it is known before the estimator fits anything.
        winner = next((a for a in attempts if a["success"]), None)
        if winner:
            print(f"\n  Tipped at {winner['force_ref_n']:.2f} N after "
                  f"{winner['attempt']} attempts; highest failed force was "
                  f"{max(a['force_ref_n'] for a in attempts if not a['success']):.2f} N.")
        else:
            print(f"\n  Never tipped up to {attempts[-1]['force_ref_n']:.2f} N.")

    # --- save -------------------------------------------------------------
    # Only the last attempt's per-tick rollout is written: with --adaptive that
    # is the one that tipped (or the highest force tried, if none did). Earlier
    # attempts survive only as the attempt_* summary arrays. Fitting wants the
    # successful rollout, and keeping every attempt's 30k samples would bloat
    # the npz for no gain.
    out = fsm.arrays()
    out["attempt_force_refs"] = np.array([a["force_ref_n"] for a in attempts], dtype=float)
    out["attempt_completed"] = np.array([a["completed"] for a in attempts], dtype=float)
    out["attempt_tipped"] = np.array([a["tipped"] for a in attempts], dtype=float)
    out["attempt_max_tip_deg"] = np.array([a["max_tip_deg"] for a in attempts], dtype=float)

    npz_path = args.output or (ROLLOUT_DIR / f"press_pull_{name}.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **out)
    print(f"\nSaved rollout ({len(out['t_hist'])} samples) to {npz_path}")

    if not args.show_viewer:
        if rv.frames:
            import mediapy as media
            video_path = args.video_path or (ROLLOUT_DIR / f"press_pull_{name}.mp4")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            media.write_video(video_path, rv.frames, fps=rv.framerate)
            print(f"Saved video to {video_path}")
        else:
            print("No video frames captured.")

    return 0 if (fsm.completed and fsm.tipped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
