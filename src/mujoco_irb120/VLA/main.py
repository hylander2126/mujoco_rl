#!/usr/bin/env python3
"""CLI for the simulation-only VLA scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mujoco_irb120.VLA.scripts.common import load_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IRB120 simulation VLA scaffold")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of collection episodes. Defaults to collect mode when no subcommand is given.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Dataset output path for collect mode.",
    )
    parser.add_argument(
        "--max-sim-time",
        type=float,
        default=None,
        help="Episode duration limit for collect/eval.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the MuJoCo viewer during collect/eval.",
    )

    subparsers = parser.add_subparsers(dest="command")

    collect = subparsers.add_parser("collect", help="Collect simulated VLA rollouts.")
    collect.add_argument("--episodes", type=int, default=5)
    collect.add_argument("--output", type=str, default=None)
    collect.add_argument("--max-sim-time", type=float, default=None)
    collect.add_argument("--render", action="store_true")

    train = subparsers.add_parser("train", help="Train behavior cloning.")
    train.add_argument("--dataset", type=str, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)

    evaluate = subparsers.add_parser("eval", help="Evaluate a trained checkpoint.")
    evaluate.add_argument("--checkpoint", type=str, required=True)
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--render", action="store_true")
    evaluate.add_argument("--max-sim-time", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = args.command or "collect"
    cfg = load_config(args.config)
    sim_cfg = cfg["sim"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    if command == "collect":
        from mujoco_irb120.VLA.scripts.collect_sim_data import collect_sim_data

        collect_sim_data(
            output_path=resolve_repo_path(args.output or data_cfg["dataset_path"]),
            episodes=args.episodes or 5,
            object_id=sim_cfg["object_id"],
            controller_type=sim_cfg["controller_type"],
            max_sim_time=args.max_sim_time or sim_cfg["max_sim_time"],
            instruction=cfg["instruction"],
            seed=cfg["seed"],
            image_height=sim_cfg["image_height"],
            image_width=sim_cfg["image_width"],
            camera_name=sim_cfg.get("camera_name", "vla_cam"),
            render=args.render,
            domain_randomization=cfg.get("domain_randomization"),
            cube_colors=tuple(cfg.get("task", {}).get("cube_colors", ["red", "blue"])),
        )
    elif command == "train":
        from mujoco_irb120.VLA.scripts.train_bc import train_bc

        train_bc(
            dataset_path=resolve_repo_path(args.dataset or data_cfg["dataset_path"]),
            checkpoint_dir=resolve_repo_path(data_cfg["checkpoint_dir"]),
            epochs=args.epochs or train_cfg["epochs"],
            batch_size=args.batch_size or train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            train_split=train_cfg["train_split"],
            seed=cfg["seed"],
        )
    elif command == "eval":
        from mujoco_irb120.VLA.scripts.eval_policy import evaluate_policy

        evaluate_policy(
            checkpoint_path=resolve_repo_path(args.checkpoint),
            episodes=args.episodes,
            object_id=sim_cfg["object_id"],
            controller_type=sim_cfg["controller_type"],
            max_sim_time=args.max_sim_time or sim_cfg["max_sim_time"],
            instruction=cfg["instruction"],
            render=args.render,
            seed=cfg["seed"],
            image_height=sim_cfg["image_height"],
            image_width=sim_cfg["image_width"],
            camera_name=sim_cfg.get("camera_name", "vla_cam"),
            domain_randomization=cfg.get("domain_randomization"),
            cube_colors=tuple(cfg.get("task", {}).get("cube_colors", ["red", "blue"])),
        )


if __name__ == "__main__":
    main()
