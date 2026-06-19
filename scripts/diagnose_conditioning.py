from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.policy import TinyVLAPolicy
from scripts.runtime import select_torch_device


def diagnose_conditioning(
    checkpoint_path: Path,
    dataset_path: Path,
    num_samples: int = 32,
    seed: int = 0,
) -> dict:
    """Measure whether the VLA policy's action head actually reacts to
    color/instruction conditioning, or whether it is dominated by state and
    just predicts the same averaged action regardless of cube color.

    For each sampled timestep, the true (image, state, instruction) triple is
    compared against versions with the image swapped to the opposite color,
    the instruction swapped to the opposite color, and both swapped. The
    resulting action shifts are reported as a fraction of the dataset's
    per-dimension action std, so a near-zero fraction means conditioning has
    essentially no effect on the predicted action.
    """
    device = select_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type", "vla") != "vla":
        raise ValueError("diagnose_conditioning only applies to 'vla' policy checkpoints")

    model = TinyVLAPolicy(
        state_dim=checkpoint.get("state_dim", 24),
        action_dim=checkpoint.get("action_dim", 6),
        hidden_dim=checkpoint.get("hidden_dim", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)

    data = np.load(dataset_path, allow_pickle=True)
    states = data["states"].astype(np.float32)
    images = data["images"]
    instructions = data["instructions"].astype(str)
    cube_color = data["cube_color"].astype(str)

    colors = sorted(set(cube_color.tolist()))
    if len(colors) != 2:
        raise ValueError(f"Expected exactly 2 cube colors in dataset, found {colors}")
    color_a, color_b = colors

    idx_a = np.where(cube_color == color_a)[0]
    idx_b = np.where(cube_color == color_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError("Dataset must contain samples for both colors")

    rng = np.random.RandomState(seed)
    num_samples = min(num_samples, len(idx_a), len(idx_b))
    pick_a = rng.choice(idx_a, size=num_samples, replace=False)
    pick_b = rng.choice(idx_b, size=num_samples, replace=False)

    def normalize_state(i):
        return (states[i] - state_mean) / state_std

    def to_image_tensor(i):
        return torch.from_numpy(images[i]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    def predict(image_idx, state_idx, instruction):
        image_t = to_image_tensor(image_idx)
        state_t = torch.from_numpy(normalize_state(state_idx)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(image_t, state_t, [instruction]).cpu().numpy()[0]

    deltas_instr = []
    deltas_image = []
    deltas_both = []
    for i, j in zip(pick_a, pick_b):
        baseline = predict(i, i, instructions[i])
        swap_instr = predict(i, i, instructions[j])
        swap_image = predict(j, i, instructions[i])
        swap_both = predict(j, i, instructions[j])

        deltas_instr.append(np.abs(swap_instr - baseline))
        deltas_image.append(np.abs(swap_image - baseline))
        deltas_both.append(np.abs(swap_both - baseline))

    eps = 1e-6
    mean_action_std = float(np.mean(action_std)) + eps

    def summarize(deltas):
        arr = np.stack(deltas, axis=0)
        return float(arr.mean()) / mean_action_std

    result = {
        "colors_compared": [color_a, color_b],
        "num_samples": num_samples,
        "instruction_swap_effect": summarize(deltas_instr),
        "image_swap_effect": summarize(deltas_image),
        "both_swap_effect": summarize(deltas_both),
    }

    if "swap_bins" in data.files:
        layout_result = _diagnose_bin_layout_conditioning(
            predict=predict,
            cube_color=cube_color,
            swap_bins=data["swap_bins"].astype(bool),
            colors=(color_a, color_b),
            mean_action_std=mean_action_std,
            num_samples=num_samples,
            rng=rng,
        )
        result.update(layout_result)

    return result


def _diagnose_bin_layout_conditioning(
    predict,
    cube_color: np.ndarray,
    swap_bins: np.ndarray,
    colors: tuple[str, str],
    mean_action_std: float,
    num_samples: int,
    rng: np.random.RandomState,
) -> dict:
    """For each color, compare predicted actions between samples recorded with
    the bins in their normal vs mirrored positions. A near-zero effect here
    means the policy isn't actually localizing the bin from the image at all,
    even if it does react to color.
    """
    deltas_layout = []
    per_color_n = []
    for color in colors:
        idx_normal = np.where((cube_color == color) & (~swap_bins))[0]
        idx_swapped = np.where((cube_color == color) & swap_bins)[0]
        n = min(num_samples, len(idx_normal), len(idx_swapped))
        per_color_n.append(n)
        if n == 0:
            continue
        pick_normal = rng.choice(idx_normal, size=n, replace=False)
        pick_swapped = rng.choice(idx_swapped, size=n, replace=False)
        for i, j in zip(pick_normal, pick_swapped):
            baseline = predict(i, i, "")
            swap_layout = predict(j, i, "")
            deltas_layout.append(np.abs(swap_layout - baseline))

    if not deltas_layout:
        return {
            "bin_layout_swap_effect": None,
            "bin_layout_samples_per_color": per_color_n,
        }

    arr = np.stack(deltas_layout, axis=0)
    return {
        "bin_layout_swap_effect": float(arr.mean()) / mean_action_std,
        "bin_layout_samples_per_color": per_color_n,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose whether a TinyVLAPolicy reacts to color/instruction.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/vla_bc.pt")
    parser.add_argument("--dataset", type=str, default="outputs/rollouts/sim_vla_rollouts.npz")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = (REPO_ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    dataset_path = (REPO_ROOT / args.dataset) if not Path(args.dataset).is_absolute() else Path(args.dataset)

    result = diagnose_conditioning(checkpoint_path, dataset_path, num_samples=args.num_samples, seed=args.seed)

    print(f"Compared colors: {result['colors_compared']} ({result['num_samples']} paired samples)")
    print("Action shift caused by each swap, as a fraction of typical per-dim action std:")
    print(f"  instruction swap only : {result['instruction_swap_effect']:.4f}")
    print(f"  image swap only       : {result['image_swap_effect']:.4f}")
    print(f"  both swapped          : {result['both_swap_effect']:.4f}")
    if result.get("bin_layout_swap_effect") is not None:
        print(f"  bin layout swap only  : {result['bin_layout_swap_effect']:.4f}")
    print(
        "\nRule of thumb: values well below ~0.1 mean that channel has almost no "
        "effect on the predicted action, i.e. the policy is not actually using it "
        "to decide which bin to sort into."
    )


if __name__ == "__main__":
    main()
