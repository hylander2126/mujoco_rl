#!/usr/bin/env python3
"""Main rollout loop for policy training data collection.

This script runs full simulation episodes and records all observations/actions
at each timestep. Replace `select_action(...)` with your control policy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

from mujoco_irb120.scripts.environment.env import IRB120Env


def select_action(observation: np.ndarray, episode_idx: int, step_idx: int) -> np.ndarray:
    """Policy hook.

    Replace this with your training control algorithm.
    Returns a 6-DoF joint target action.
    """
    del observation, episode_idx, step_idx
    return np.zeros(6, dtype=np.float32)


def run_rollouts(
    num_episodes: int,
    object_id: int,
    controller_type: str,
    max_sim_time: float,
    render: bool,
    output_path: Path,
) -> None:
    """Execute episodes and save full transition logs for training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_actions = []
    all_next_obs = []
    all_done = []
    all_episode_idx = []
    all_step_idx = []
    all_sim_time = []

    start_wall = time.time()

    with IRB120Env(
        object_id=object_id,
        controller_type=controller_type,
        max_sim_time=max_sim_time,
        render_mode="human" if render else None,
    ) as env:
        for episode_idx in range(num_episodes):
            obs, info = env.reset()
            done = False
            step_idx = 0

            while not done:
                action = np.asarray(select_action(obs, episode_idx, step_idx), dtype=np.float32).reshape(6)
                next_obs, done, info = env.step(action)

                all_obs.append(obs.copy())
                all_actions.append(action.copy())
                all_next_obs.append(next_obs.copy())
                all_done.append(bool(done))
                all_episode_idx.append(episode_idx)
                all_step_idx.append(step_idx)
                all_sim_time.append(float(info["sim_time"]))

                obs = next_obs
                step_idx += 1

            print(
                f"Episode {episode_idx + 1}/{num_episodes}: "
                f"steps={step_idx}, sim_time={info['sim_time']:.3f}s"
            )

    np.savez_compressed(
        output_path,
        observations=np.asarray(all_obs, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.float32),
        next_observations=np.asarray(all_next_obs, dtype=np.float32),
        done=np.asarray(all_done, dtype=np.bool_),
        episode_idx=np.asarray(all_episode_idx, dtype=np.int32),
        step_idx=np.asarray(all_step_idx, dtype=np.int32),
        sim_time=np.asarray(all_sim_time, dtype=np.float32),
        object_id=np.asarray([object_id], dtype=np.int32),
        controller_type=np.asarray([controller_type]),
        max_sim_time=np.asarray([max_sim_time], dtype=np.float32),
    )

    elapsed = time.time() - start_wall
    print(f"Saved {len(all_actions)} transitions to: {output_path}")
    print(f"Wall-clock runtime: {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IRB120 rollouts and record trajectories.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--object-id", type=int, default=0, help="Object id used by IRB120Env.")
    parser.add_argument(
        "--controller-type",
        type=str,
        choices=["position", "velocity"],
        default="position",
        help="Controller backend for IRB120Env.",
    )
    parser.add_argument("--max-sim-time", type=float, default=30.0, help="Episode limit in sim seconds.")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer.")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "rollouts" / "irb120_rollouts.npz",
        help="Output .npz file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rollouts(
        num_episodes=args.episodes,
        object_id=args.object_id,
        controller_type=args.controller_type,
        max_sim_time=args.max_sim_time,
        render=args.render,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()