#!/usr/bin/env python3

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

ROLLOUT_DIR = REPO_ROOT / "outputs" / "parameter_estimation" / "press_pull_rollouts"
OBJECT_PARAMS_PATH = REPO_ROOT / "parameter_estimation" / "object_params.json"

# --- Tunable rollout parameters --------------------------------------------
FORCE_REF_N = 5.0     # Squash force reference in N. Hardware default: 5.0.
SPEED_SCALE = 1.0     # Multiply all motion speeds. 1.0 = hardware speed
PRESS_OFFSET_X = 0.0  # Shift the press point along X from the top-face centre. Closer to tipping edge -> less tip force.
ADAPTIVE_RETRY = False  # On slip, retry with the squash force scaled up.
MAX_ATTEMPTS = 5      # Cap on adaptive retries.
QUIET = False         # Suppress per-phase FSM logging.
VIDEO_SPEEDUP = 2.0    # Write the mp4 at N x realtime for quicker review (frames unaffected).


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", type=int, default=0,
                   help=f"Object id. One of {sorted(OBJECTS)} ({OBJECTS}). Default: 0 (box).")
    viewer = p.add_mutually_exclusive_group()
    viewer.add_argument("--show-viewer", dest="show_viewer", action="store_true",
                        help="Open the live MuJoCo viewer.")
    viewer.add_argument("--no-viewer", dest="show_viewer", action="store_false",
                        help="Run headless and write a video instead (default).")
    p.set_defaults(show_viewer=False)
    return p.parse_args()


def run_attempt(args, force_ref: float, record_video: bool):
    """One full press-and-pull sequence from a freshly reset scene."""
    model, data = load_environment(num=args.object, launch_viewer=False)

    irb = robot_controller.controller(model, data)

    cfg = PressPullConfig(
        force_ref_n=force_ref,
        speed_scale=SPEED_SCALE,
        adaptive_retry=ADAPTIVE_RETRY,
        press_offset_xy=(PRESS_OFFSET_X, 0.0),
        verbose=not QUIET,
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
    print(f"Object: [{args.object}] {name}   force_ref: {FORCE_REF_N} N   "
          f"speed_scale: {SPEED_SCALE}")

    params = json.load(open(OBJECT_PARAMS_PATH))["objects"]
    gt = params.get(name)
    if gt is not None:
        com_gt = np.array(gt["com_gt"])
        print(f"Ground truth: mass={gt['mass_gt']} kg  com={com_gt} m"
              + (f"  theta*={gt['theta_star']:.3f} deg" if "theta_star" in gt else ""))
    else:
        print(f"No ground-truth entry for object {args.object} in object_params.json "
              "-- rollout will still record, but nothing can be scored against it.")

    force_ref = FORCE_REF_N
    attempts = []
    fsm = rv = model = data = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
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

        if success or not ADAPTIVE_RETRY:
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

    if ADAPTIVE_RETRY and len(attempts) > 1:
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

    npz_path = ROLLOUT_DIR / f"press_pull_{name}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **out)
    print(f"\nSaved rollout ({len(out['t_hist'])} samples) to {npz_path}")

    if not args.show_viewer:
        if rv.frames:
            import mediapy as media
            video_path = ROLLOUT_DIR / f"press_pull_{name}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            media.write_video(video_path, rv.frames, fps=rv.framerate * VIDEO_SPEEDUP)
            print(f"Saved video to {video_path} ({VIDEO_SPEEDUP:g}x realtime)")
        else:
            print("No video frames captured.")

    return 0 if (fsm.completed and fsm.tipped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
