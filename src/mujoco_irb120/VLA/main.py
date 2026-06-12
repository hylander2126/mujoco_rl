#!/usr/bin/env python3
"""CLI for the simulation-only VLA scaffold."""

from __future__ import annotations

import argparse

# Allows for nice and easy calling using python3 main.py instead of -m flag or installing as package.
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
    parser.add_argument(
        "--record-stride",
        type=int,
        default=None,
        help="Collect mode only: save one image/state/action sample every N sim steps.",
    )
    parser.add_argument(
        "--control-stride",
        type=int,
        default=None,
        help="Eval mode only: run the policy once every N sim steps and hold the action between updates.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=None,
        help="Eval mode only: maximum absolute joint-target change per policy update.",
    )

    subparsers = parser.add_subparsers(dest="command")

    collect = subparsers.add_parser("collect", help="Collect simulated VLA rollouts.")
    collect.add_argument("--episodes", type=int, default=5)
    collect.add_argument("--output", type=str, default=None)
    collect.add_argument("--max-sim-time", type=float, default=None)
    collect.add_argument("--render", action="store_true")
    collect.add_argument("--record-stride", type=int, default=None)

    train = subparsers.add_parser("train", help="Train behavior cloning.")
    train.add_argument("--dataset", type=str, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)

    evaluate = subparsers.add_parser("eval", help="Evaluate a trained checkpoint.")
    evaluate.add_argument("--checkpoint", type=str, required=True)
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--render", action="store_true")
    evaluate.add_argument("--max-sim-time", type=float, default=None)
    evaluate.add_argument("--control-stride", type=int, default=None)
    evaluate.add_argument("--max-joint-delta", type=float, default=None)

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
            episodes=args.episodes if args.episodes is not None else 5,
            max_sim_time=args.max_sim_time if args.max_sim_time is not None else sim_cfg["max_sim_time"],
            seed=cfg["seed"],
            image_height=sim_cfg["image_height"],
            image_width=sim_cfg["image_width"],
            record_stride=args.record_stride if args.record_stride is not None else sim_cfg.get("record_stride", 1),
            render=args.render,
            domain_randomization=cfg.get("domain_randomization"),
        )
    elif command == "train":
        from mujoco_irb120.VLA.scripts.train_bc import train_bc

        train_bc(
            dataset_path=resolve_repo_path(args.dataset or data_cfg["dataset_path"]),
            checkpoint_dir=resolve_repo_path(data_cfg["checkpoint_dir"]),
            epochs=args.epochs if args.epochs is not None else train_cfg["epochs"],
            batch_size=args.batch_size if args.batch_size is not None else train_cfg["batch_size"],
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
            max_sim_time=args.max_sim_time if args.max_sim_time is not None else sim_cfg["max_sim_time"],
            render=args.render,
            seed=cfg["seed"],
            image_height=sim_cfg["image_height"],
            image_width=sim_cfg["image_width"],
            control_stride=(
                args.control_stride
                if args.control_stride is not None
                else sim_cfg.get("control_stride", sim_cfg.get("record_stride", 1))
            ),
            max_joint_delta=args.max_joint_delta if args.max_joint_delta is not None else sim_cfg.get("max_joint_delta", 0.02),
            domain_randomization=cfg.get("domain_randomization"),
        )


if __name__ == "__main__":
    main()
