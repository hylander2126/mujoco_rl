from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.paths import REPO_ROOT


def plot_training_curves(checkpoint_path: Path, output_path: Path | None = None) -> Path:
    """Render train/validation loss (and color accuracy, for vla checkpoints)
    over epochs from a checkpoint's saved training history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history")
    if not history or not history.get("epoch"):
        raise ValueError(
            f"{checkpoint_path} has no training history to plot. Checkpoints saved "
            "before this feature was added, or loaded/re-saved outside train_bc.py, "
            "won't have one — retrain to get a plottable checkpoint."
        )

    epochs = history["epoch"]
    policy_type = checkpoint.get("policy_type", "vla")
    has_accuracy = policy_type == "vla" and any(v is not None for v in history["train_color_acc"])

    fig, axes = plt.subplots(1, 2 if has_accuracy else 1, figsize=(12 if has_accuracy else 6, 4.5))
    loss_ax = axes[0] if has_accuracy else axes

    loss_ax.plot(epochs, history["train_loss"], label="train")
    val_epochs = [e for e, v in zip(epochs, history["validation_loss"]) if v is not None]
    val_loss = [v for v in history["validation_loss"] if v is not None]
    if val_loss:
        loss_ax.plot(val_epochs, val_loss, label="validation")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title(f"{checkpoint_path.stem}: loss")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)

    if has_accuracy:
        acc_ax = axes[1]
        acc_ax.plot(epochs, history["train_color_acc"], label="train")
        val_acc_epochs = [e for e, v in zip(epochs, history["validation_color_acc"]) if v is not None]
        val_acc = [v for v in history["validation_color_acc"] if v is not None]
        if val_acc:
            acc_ax.plot(val_acc_epochs, val_acc, label="validation")
        acc_ax.set_xlabel("epoch")
        acc_ax.set_ylabel("color accuracy")
        acc_ax.set_ylim(0.0, 1.05)
        acc_ax.set_title(f"{checkpoint_path.stem}: color accuracy")
        acc_ax.legend()
        acc_ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path is None:
        output_path = REPO_ROOT / "outputs" / "figures" / f"{checkpoint_path.stem}_training_curves.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot loss/accuracy curves from a checkpoint's training history.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/vla_bc.pt")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = (REPO_ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    output_path = args.output and ((REPO_ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output))
    saved_path = plot_training_curves(checkpoint_path, output_path)
    print(f"Saved training curve plot to {saved_path}")


if __name__ == "__main__":
    main()
