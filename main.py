#!/usr/bin/env python3
"""CLI for the simulation-only VLA scaffold."""

from __future__ import annotations

import argparse

from scripts.common import load_config, resolve_repo_path


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
        help="Save headless MP4 videos during collect/eval.",
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
    collect.add_argument("--render", action="store_true", help="Save one MP4 video per episode.")
    collect.add_argument("--record-stride", type=int, default=None)
    collect.add_argument(
        "--randomize-bin-layout",
        action="store_true",
        help="Randomly mirror which physical slot each color's bin occupies per "
        "episode, so the training data isn't all one fixed bin layout.",
    )

    train = subparsers.add_parser("train", help="Train behavior cloning.")
    train.add_argument("--dataset", type=str, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument(
        "--policy-type",
        choices=["vla", "state_only"],
        default=None,
        help="Train image+language+state VLA policy or state-only baseline.",
    )

    evaluate = subparsers.add_parser("eval", help="Evaluate a trained checkpoint.")
    evaluate.add_argument("--checkpoint", type=str, required=True)
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--render", action="store_true", help="Save one MP4 video per episode.")
    evaluate.add_argument("--max-sim-time", type=float, default=None)
    evaluate.add_argument("--control-stride", type=int, default=None)
    evaluate.add_argument("--max-joint-delta", type=float, default=None)
    evaluate.add_argument(
        "--bin-layout",
        choices=["normal", "swapped", "random"],
        default="normal",
        help="'normal' keeps the trained bin positions, 'swapped' mirrors which "
        "physical slot each color occupies, 'random' picks per episode to test "
        "whether the policy generalizes to bin position instead of memorizing it.",
    )

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Check whether a VLA checkpoint's predicted action reacts to color/instruction conditioning.",
    )
    diagnose.add_argument("--checkpoint", type=str, default=None)
    diagnose.add_argument("--dataset", type=str, default=None)
    diagnose.add_argument("--num-samples", type=int, default=32)

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
            max_sim_time=args.max_sim_time if args.max_sim_time is not None else sim_cfg["max_sim_time"],
            seed=cfg["seed"],
            image_height=sim_cfg["image_height"],
            image_width=sim_cfg["image_width"],
            video_height=sim_cfg.get("video_height", 720),
            video_width=sim_cfg.get("video_width", 720),
            record_stride=args.record_stride if args.record_stride is not None else sim_cfg.get("record_stride", 1),
            render=args.render,
            domain_randomization=cfg.get("domain_randomization"),
            randomize_bin_layout=args.randomize_bin_layout,
        )
    elif command == "train":
        from scripts.train_bc import train_bc

        train_bc(
            dataset_path=resolve_repo_path(args.dataset or data_cfg["dataset_path"]),
            checkpoint_dir=resolve_repo_path(data_cfg["checkpoint_dir"]),
            epochs=args.epochs if args.epochs is not None else train_cfg["epochs"],
            batch_size=args.batch_size if args.batch_size is not None else train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            train_split=train_cfg["train_split"],
            seed=cfg["seed"],
            policy_type=args.policy_type or train_cfg.get("policy_type", "vla"),
        )
    elif command == "eval":
        from scripts.eval_policy import evaluate_policy

        evaluate_policy(
            checkpoint_path=resolve_repo_path(args.checkpoint),
            episodes=args.episodes,
            max_sim_time=args.max_sim_time if args.max_sim_time is not None else sim_cfg["max_sim_time"],
            render=args.render,
            seed=cfg["seed"],
            image_height=sim_cfg["image_height"],
            image_width=sim_cfg["image_width"],
            video_height=sim_cfg.get("video_height", 720),
            video_width=sim_cfg.get("video_width", 720),
            control_stride=(
                args.control_stride
                if args.control_stride is not None
                else sim_cfg.get("control_stride", sim_cfg.get("record_stride", 1))
            ),
            max_joint_delta=args.max_joint_delta if args.max_joint_delta is not None else sim_cfg.get("max_joint_delta", 0.02),
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
        print(f"Compared colors: {result['colors_compared']} ({result['num_samples']} paired samples)")
        print("Action shift caused by each swap, as a fraction of typical per-dim action std:")
        print(f"  instruction swap only : {result['instruction_swap_effect']:.4f}")
        print(f"  image swap only       : {result['image_swap_effect']:.4f}")
        print(f"  both swapped          : {result['both_swap_effect']:.4f}")
        if result.get("bin_layout_swap_effect") is not None:
            print(f"  bin layout swap only  : {result['bin_layout_swap_effect']:.4f}")


if __name__ == "__main__":
    main()
