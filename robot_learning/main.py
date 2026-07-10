#!/usr/bin/env python3
"""CLI for the simulation-only VLA scaffold."""

from __future__ import annotations

import argparse

import yaml

from util.paths import REPO_ROOT, resolve_repo_path

DEFAULT_CONFIG = REPO_ROOT / "environment" / "default.yaml"


def load_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        "--render",
        action="store_true",
        help="Save headless MP4 videos during collect/eval.",
    )
    parser.add_argument(
        "--record-stride",
        type=int,
        default=None,
        help="Collect mode only: save one image/state/action sample every N sim steps.",
    )

    subparsers = parser.add_subparsers(dest="command")

    collect = subparsers.add_parser("collect", help="Collect simulated VLA rollouts.")
    collect.add_argument("--episodes", type=int, default=5)
    collect.add_argument("--output", type=str, default=None)
    collect.add_argument("--render", action="store_true", help="Save one MP4 video per episode.")
    collect.add_argument("--record-stride", type=int, default=None)
    collect.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Collection seed. Use -1 for nondeterministic bin/object randomization.",
    )
    collect.add_argument(
        "--randomize-bin-layout",
        action="store_true",
        help="Mirror/swap which physical slot each color's bin occupies per episode.",
    )
    collect.add_argument(
        "--randomize-bin-positions",
        action="store_true",
        help="Sample new reachable bin XY positions and matching Z yaw per episode.",
    )

    train = subparsers.add_parser("train", help="Train behavior cloning.")
    train.add_argument("--dataset", type=str, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument(
        "--policy-type",
        choices=["goal_conditioned", "vla", "state_only"],
        default=None,
    )
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=None)

    evaluate = subparsers.add_parser("eval", help="Evaluate a trained checkpoint.")
    evaluate.add_argument("--checkpoint", type=str, required=True)
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--render", action="store_true", help="Save one MP4 video per episode.")
    evaluate.add_argument(
        "--max-sim-time",
        type=float,
        default=None,
        help="Episode duration limit. Defaults to the task's expert timeline length; "
        "a learned policy may need more time than the scripted oracle to still succeed.",
    )
    evaluate.add_argument(
        "--control-stride",
        type=int,
        default=None,
        help="Run the policy once every N sim steps and hold the action between updates "
        "(this slows the robot, not just fidelity -- the held joint target is a no-op "
        "between updates). Defaults to 1, which tracks the demonstrations best regardless "
        "of the dataset's record_stride.",
    )
    evaluate.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluation seed. Use -1 for nondeterministic randomized layouts.",
    )
    evaluate.add_argument(
        "--bin-layout",
        choices=["normal", "swapped", "random", "randomized"],
        default="normal",
        help="'normal' keeps default bins, 'swapped' mirrors color slots, 'random' randomly swaps, "
        "and 'randomized' samples new reachable bin positions/yaws per episode.",
    )

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Check whether a VLA checkpoint's predicted action reacts to color/instruction conditioning.",
    )
    diagnose.add_argument("--checkpoint", type=str, default=None)
    diagnose.add_argument("--dataset", type=str, default=None)
    diagnose.add_argument("--num-samples", type=int, default=32)

    plot = subparsers.add_parser(
        "plot",
        help="Plot a checkpoint's train/validation loss (and color accuracy) over epochs.",
    )
    plot.add_argument("--checkpoint", type=str, default=None)
    plot.add_argument("--output", type=str, default=None, help="PNG output path. Defaults to outputs/robot_learning/figures/.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = args.command or "collect"
    cfg = load_config(args.config)
    sim_cfg = cfg["sim"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    if command == "collect":
        from scripts.collect_sim_data import collect_sim_data

        collect_sim_data(
            output_path=resolve_repo_path(args.output or data_cfg["dataset_path"]),
            episodes=args.episodes if args.episodes is not None else 5,
            seed=args.seed if args.seed is not None else cfg["seed"],
            image_height=sim_cfg["image_height"],
            image_width=sim_cfg["image_width"],
            video_height=sim_cfg.get("video_height", 720),
            video_width=sim_cfg.get("video_width", 720),
            record_stride=args.record_stride if args.record_stride is not None else sim_cfg.get("record_stride", 1),
            render=args.render,
            domain_randomization=cfg.get("domain_randomization"),
            randomize_bin_layout=args.randomize_bin_layout,
            randomize_bin_positions=args.randomize_bin_positions,
        )
    elif command == "train":
        from scripts.train_bc import train_bc

        train_bc(
            dataset_path=resolve_repo_path(args.dataset or data_cfg["dataset_path"]),
            checkpoint_dir=resolve_repo_path(data_cfg["checkpoint_dir"]),
            epochs=args.epochs if args.epochs is not None else train_cfg["epochs"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            train_split=train_cfg["train_split"],
            seed=cfg["seed"],
            batch_size=args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32),
            num_workers=args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4),
            policy_type=args.policy_type or train_cfg.get("policy_type", "goal_conditioned"),
        )
    elif command == "eval":
        from scripts.eval_policy import evaluate_policy

        evaluate_policy(
            checkpoint_path=resolve_repo_path(args.checkpoint),
            episodes=args.episodes,
            render=args.render,
            seed=args.seed if args.seed is not None else cfg["seed"],
            max_sim_time=args.max_sim_time,
            image_height=sim_cfg["image_height"],
            image_width=sim_cfg["image_width"],
            video_height=sim_cfg.get("video_height", 720),
            video_width=sim_cfg.get("video_width", 720),
            control_stride=args.control_stride,
            domain_randomization=cfg.get("domain_randomization"),
            bin_layout=args.bin_layout,
        )
    elif command == "diagnose":
        from scripts.diagnose_conditioning import diagnose_conditioning

        checkpoint_path = resolve_repo_path(
            args.checkpoint or f"{data_cfg['checkpoint_dir']}/vla_bc.pt"
        )
        dataset_path = resolve_repo_path(args.dataset or data_cfg["dataset_path"])
        result = diagnose_conditioning(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            num_samples=args.num_samples,
            seed=cfg["seed"],
        )
        if result.get("policy_type") == "goal_conditioned":
            print(f"Compared {result['num_samples']} same-color goal swaps with robot state held fixed")
            print("Action shift as a fraction of typical per-dim action std:")
            print(f"  selected-bin goal swap: {result['goal_swap_effect']:.4f}")
        else:
            print(f"Compared colors: {result['colors_compared']} ({result['num_samples']} paired samples)")
            print("Action shift caused by each swap, as a fraction of typical per-dim action std:")
            print(f"  instruction swap only : {result['instruction_swap_effect']:.4f}")
            print(f"  image swap only       : {result['image_swap_effect']:.4f}")
            print(f"  both swapped          : {result['both_swap_effect']:.4f}")
            if result.get("bin_layout_swap_effect") is not None:
                print(f"  bin layout swap only  : {result['bin_layout_swap_effect']:.4f}")
    elif command == "plot":
        from scripts.plot_training import plot_training_curves

        checkpoint_path = resolve_repo_path(
            args.checkpoint or f"{data_cfg['checkpoint_dir']}/vla_bc.pt"
        )
        output_path = resolve_repo_path(args.output) if args.output else None
        saved_path = plot_training_curves(checkpoint_path=checkpoint_path, output_path=output_path)
        print(f"Saved training curve plot to {saved_path}")


if __name__ == "__main__":
    main()
