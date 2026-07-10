from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.policy import GoalConditionedBCPolicy, StateOnlyBCPolicy, TinyVLAPolicy
from util.debug_log import new_debug_dir
from util.rollout_dataset import load_rollout_dataset
from util.runtime import select_torch_device

COUNTERFACTUAL_MAX_JOINT_DELTA = 0.02


def _plot_action_delta_histograms(actions: np.ndarray, path) -> None:
    """Per-joint distribution of the joint-delta labels the model is trained
    on. A systematic skew in one joint here (rather than the model itself)
    would point at the dataset/oracle as the source of a biased policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_joints = actions.shape[1]
    fig, axes = plt.subplots(1, n_joints, figsize=(2.6 * n_joints, 3), sharey=False)
    axes = np.atleast_1d(axes)
    for joint, ax in enumerate(axes):
        values = actions[:, joint]
        ax.hist(values, bins=50)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"joint {joint}\nmean={values.mean():.4f}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


class EpisodeChunkedSampler(Sampler):
    """Shuffles episode order each epoch, and sample order within an episode,
    but keeps a given episode's samples adjacent in the iteration order.

    Plain per-sample `shuffle=True` scatters reads across the entire dataset on
    every batch. Once images are bigger than RAM, grouping each episode's
    samples improves storage locality while still shuffling episode order and
    sample order within each episode.
    """

    def __init__(self, indices: np.ndarray, episode_idx: np.ndarray):
        # `indices` are the global sample rows this sampler's dataset was built
        # from (e.g. BCDataset(train_indices)); the dataset maps a *position*
        # in `indices` back to a global row via `self.indices[idx]`, so this
        # sampler must yield positions (0..len(indices)-1), not global rows.
        self.num_positions = len(indices)
        self.episode_of_position = np.asarray(episode_idx)[np.asarray(indices)]

    def __iter__(self):
        rng = np.random.default_rng()
        positions = np.arange(self.num_positions)
        episodes = np.unique(self.episode_of_position)
        rng.shuffle(episodes)
        order = []
        for ep in episodes:
            ep_positions = positions[self.episode_of_position == ep].copy()
            rng.shuffle(ep_positions)
            order.append(ep_positions)
        return iter(np.concatenate(order).tolist())

    def __len__(self):
        return self.num_positions


def train_bc(
    dataset_path,
    checkpoint_dir,
    epochs,
    learning_rate,
    weight_decay,
    train_split,
    seed,
    batch_size: int = 32,
    policy_type: str = "goal_conditioned",
    color_loss_weight: float = 0.5,
    checkpoint_every: int = 20,
    num_workers: int = 4,
):
    """Train behavior cloning from collected MuJoCo demonstrations.

    `policy_type="goal_conditioned"` trains selected-bin-pose + progress BC.
    `policy_type="vla"` retains the image + language + state experiment, and
    `policy_type="state_only"` is the fixed-task proprioceptive baseline.
    """
    if policy_type not in {"goal_conditioned", "vla", "state_only"}:
        raise ValueError(
            "policy_type must be 'goal_conditioned', 'vla', or 'state_only', "
            f"got {policy_type!r}"
        )

    dataset_path = Path(dataset_path)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = new_debug_dir("train")
    print(f"[train_bc] Debug logs (dataset stats, histograms, curves) -> {debug_dir}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_rollout_dataset(dataset_path)
    states = data["states"].astype(np.float32)
    expert_joint_targets = data["actions"].astype(np.float32)
    current_joint_positions = states[:, : expert_joint_targets.shape[1]]
    actions = expert_joint_targets - current_joint_positions
    episode_idx = data["episode_idx"]

    goals = None
    goal_mean = None
    goal_std = None
    if policy_type == "goal_conditioned":
        cube_color = data["cube_color"].astype(str)
        is_red = cube_color == "red"
        goal_xy = np.where(is_red[:, None], data["red_bin_xy"], data["blue_bin_xy"]).astype(np.float32)
        goal_yaw = np.where(is_red, data["red_bin_yaw"], data["blue_bin_yaw"]).astype(np.float32)
        # Encode yaw periodically so +pi and -pi remain neighboring goals.
        sim_timestep = float(data["sim_timestep"])
        max_sim_time = float(data["max_sim_time"])
        progress = np.clip(data["step_idx"].astype(np.float32) * sim_timestep / max_sim_time, 0.0, 1.0)
        goals = np.column_stack((goal_xy, np.sin(goal_yaw), np.cos(goal_yaw), progress)).astype(np.float32)

    if policy_type == "vla":
        images = data["images"]
        instructions = data["instructions"].astype(str)
        cube_color = data["cube_color"].astype(str)
        color_classes = sorted(set(cube_color.tolist()))
        if len(color_classes) != 2:
            raise ValueError(f"Expected exactly 2 cube colors, found {color_classes}")
        color_to_idx = {color: idx for idx, color in enumerate(color_classes)}
        color_idx = np.array([color_to_idx[c] for c in cube_color], dtype=np.int64)
        print(
            f"Images shape: {images.shape}, States shape: {states.shape}, "
            f"Joint-delta actions shape: {actions.shape}, color classes: {color_classes}"
        )
    elif policy_type == "goal_conditioned":
        images = None
        instructions = None
        print(
            f"States shape: {states.shape}, Goals shape: {goals.shape}, "
            f"Joint-delta actions shape: {actions.shape}"
        )
    else:
        images = None
        instructions = None
        print(f"States shape: {states.shape}, Joint-delta actions shape: {actions.shape}")

    unique_episodes = np.unique(episode_idx)
    np.random.shuffle(unique_episodes)

    validation_frac = 1.0 - train_split
    n_validation = int(len(unique_episodes) * validation_frac)
    if len(unique_episodes) > 1 and validation_frac > 0.0:
        n_validation = max(1, n_validation)
    validation_episodes = unique_episodes[:n_validation]
    train_episodes = unique_episodes[n_validation:]

    train_indices = np.where(np.isin(episode_idx, train_episodes))[0]
    validation_indices = np.where(np.isin(episode_idx, validation_episodes))[0]
    print(f"Training indices: {len(train_indices)}, Validation indices: {len(validation_indices)}")

    # Expert-driven qpos reveals which randomized-bin trajectory is already in
    # progress, so a network can minimize offline loss while ignoring `goal`.
    # For half of goal-conditioned training rows, keep the current robot state
    # but pair it with another training episode's goal and expert joint target
    # at the same color/timestep. This makes goal information causally
    # necessary and supplies corrective labels toward alternate trajectories.
    counterfactual_target_idx = np.arange(len(states))
    counterfactual_positions = np.zeros(len(train_indices), dtype=bool)
    if policy_type == "goal_conditioned":
        step_idx = data["step_idx"].astype(np.int64)
        train_groups: dict[tuple[str, int], np.ndarray] = {}
        for color in np.unique(cube_color):
            color_rows = train_indices[cube_color[train_indices] == color]
            for step in np.unique(step_idx[color_rows]):
                rows = color_rows[step_idx[color_rows] == step]
                if len(np.unique(episode_idx[rows])) > 1:
                    train_groups[(str(color), int(step))] = rows
        pairing_rng = np.random.default_rng(seed + 1)
        for position, i in enumerate(train_indices):
            if position % 5 != 0:
                continue
            candidates = train_groups.get((str(cube_color[i]), int(step_idx[i])))
            if candidates is None:
                continue
            candidates = candidates[episode_idx[candidates] != episode_idx[i]]
            if len(candidates):
                counterfactual_target_idx[i] = int(pairing_rng.choice(candidates))
                counterfactual_positions[position] = True
        print(
            f"Counterfactual goal relabeling: {counterfactual_positions.sum()} / "
            f"{len(train_indices)} training samples"
        )

    train_label_actions = actions[train_indices].copy()
    if policy_type == "goal_conditioned":
        paired_rows = train_indices[counterfactual_positions]
        counterfactual_deltas = (
            expert_joint_targets[counterfactual_target_idx[paired_rows]]
            - current_joint_positions[paired_rows]
        )
        train_label_actions[counterfactual_positions] = np.clip(
            counterfactual_deltas,
            -COUNTERFACTUAL_MAX_JOINT_DELTA,
            COUNTERFACTUAL_MAX_JOINT_DELTA,
        )

    state_mean = states[train_indices].mean(axis=0)
    state_std = states[train_indices].std(axis=0) + 1e-6
    action_mean = train_label_actions.mean(axis=0)
    action_std = train_label_actions.std(axis=0) + 1e-6
    if policy_type == "goal_conditioned":
        goal_mean = goals[train_indices].mean(axis=0)
        goal_std = goals[train_indices].std(axis=0) + 1e-6

    np.savez_compressed(
        debug_dir / "dataset_stats.npz",
        action_mean=action_mean,
        action_std=action_std,
        action_min=train_label_actions.min(axis=0),
        action_max=train_label_actions.max(axis=0),
        state_mean=state_mean,
        state_std=state_std,
        goal_mean=goal_mean,
        goal_std=goal_std,
    )
    _plot_action_delta_histograms(train_label_actions, debug_dir / "action_delta_histograms.png")

    class BCDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices
            self.is_training = indices is train_indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]
            state = (states[i] - state_mean) / state_std
            target_i = counterfactual_target_idx[i] if self.is_training else i
            raw_action = expert_joint_targets[target_i] - current_joint_positions[i]
            if policy_type == "goal_conditioned" and target_i != i:
                raw_action = np.clip(
                    raw_action,
                    -COUNTERFACTUAL_MAX_JOINT_DELTA,
                    COUNTERFACTUAL_MAX_JOINT_DELTA,
                )
            action = (raw_action - action_mean) / action_std
            item = {
                "state": torch.tensor(state, dtype=torch.float32),
                "action": torch.tensor(action, dtype=torch.float32),
            }
            if policy_type == "vla":
                item["image"] = torch.from_numpy(images[i]).float().permute(2, 0, 1) / 255.0
                item["instruction"] = str(instructions[i])
                item["color_idx"] = torch.tensor(color_idx[i], dtype=torch.long)
            elif policy_type == "goal_conditioned":
                item["goal"] = torch.tensor((goals[target_i] - goal_mean) / goal_std, dtype=torch.float32)
            return item

    # Lazy HDF5 image loading is disk I/O per sample. With num_workers=0 it blocks the GPU
    # between batches; worker processes let reads for the next batch overlap
    # with the current batch's forward/backward pass.
    train_loader = DataLoader(
        BCDataset(train_indices),
        batch_size=batch_size,
        sampler=EpisodeChunkedSampler(train_indices, episode_idx),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    # validation_indices comes out of np.where() in ascending order, which is
    # already episode-sequential, and
    # the validation split is much smaller than train -- no need to duplicate
    # a whole persistent worker pool for it.
    validation_num_workers = min(num_workers, 2)
    validation_loader = DataLoader(
        BCDataset(validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=validation_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=validation_num_workers > 0,
    )

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    goal_dim = goals.shape[1] if goals is not None else None
    hidden_dim = 128 if policy_type == "vla" else 256
    device = select_torch_device()

    if policy_type == "goal_conditioned":
        model = GoalConditionedBCPolicy(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    elif policy_type == "vla":
        model = TinyVLAPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        model = StateOnlyBCPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)

    loss_fn = nn.MSELoss()
    color_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def predict(batch):
        state = batch["state"].to(device)
        if policy_type == "vla":
            image = batch["image"].to(device)
            instruction = list(batch["instruction"])
            return model(image, state, instruction, return_color_logits=True)
        if policy_type == "goal_conditioned":
            return model(state, batch["goal"].to(device)), None
        return model(state), None

    first_batch = next(iter(train_loader))
    with torch.no_grad():
        pred_action, _ = predict(first_batch)
    print(
        "These should match and be (batch_size, action_dim): "
        f"{pred_action.shape}, {first_batch['action'].shape}"
    )
    initial_loss = loss_fn(pred_action.cpu(), first_batch["action"])
    print(f"Initial loss (should be > 0 and probably large): {initial_loss.item()}")

    def run_epoch(loader, optimizer=None):
        is_training = optimizer is not None
        model.train() if is_training else model.eval()

        total_loss = 0.0
        total_color_correct = 0
        total_count = 0
        for batch in loader:
            action = batch["action"].to(device)
            with torch.set_grad_enabled(is_training):
                pred_action, color_logits = predict(batch)
                loss = loss_fn(pred_action, action)

                if policy_type == "vla":
                    color_target = batch["color_idx"].to(device)
                    color_loss = color_loss_fn(color_logits, color_target)
                    loss = loss + color_loss_weight * color_loss
                    total_color_correct += (color_logits.argmax(dim=-1) == color_target).sum().item()

                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            batch_size_this = action.shape[0]
            total_loss += loss.item() * batch_size_this
            total_count += batch_size_this

        if total_count == 0:
            return None, None
        color_accuracy = total_color_correct / total_count if policy_type == "vla" else None
        return total_loss / total_count, color_accuracy

    checkpoint_name = {
        "goal_conditioned": "goal_conditioned_bc.pt",
        "vla": "randomized_vla.pt",
        "state_only": "state_only_bc.pt",
    }[policy_type]
    checkpoint_path = checkpoint_dir / checkpoint_name
    best_checkpoint_path = checkpoint_dir / f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"

    history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_color_acc": [],
        "validation_color_acc": [],
    }

    def save_checkpoint(path: Path) -> None:
        ckpt = {
            "model": model.state_dict(),
            "model_state_dict": model.state_dict(),
            "policy_type": policy_type,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "goal_dim": goal_dim,
            "hidden_dim": hidden_dim,
            "state_mean": torch.tensor(state_mean, dtype=torch.float32),
            "state_std": torch.tensor(state_std, dtype=torch.float32),
            "action_mean": torch.tensor(action_mean, dtype=torch.float32),
            "action_std": torch.tensor(action_std, dtype=torch.float32),
            "goal_mean": None if goal_mean is None else torch.tensor(goal_mean, dtype=torch.float32),
            "goal_std": None if goal_std is None else torch.tensor(goal_std, dtype=torch.float32),
            "goal_encoding": "selected_bin_xy_sin_yaw_cos_yaw_progress" if goal_dim is not None else None,
            "counterfactual_goal_relabeling": policy_type == "goal_conditioned",
            "action_mode": "joint_delta",
            "record_stride": int(data["record_stride"]) if "record_stride" in data.files else None,
            "ft_bias_enabled": bool(data["ft_bias_enabled"]) if "ft_bias_enabled" in data.files else None,
            "ft_bias_samples": int(data["ft_bias_samples"]) if "ft_bias_samples" in data.files else None,
            "dataset_path": str(dataset_path),
            "history": history,
        }
        torch.save(ckpt, path)

    best_validation_loss = float("inf")

    for epoch in range(epochs):
        train_loss, train_color_acc = run_epoch(train_loader, optimizer)
        validation_loss, validation_color_acc = run_epoch(validation_loader, optimizer=None)
        if validation_loss is None:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = skipped (no validation samples)")
        elif policy_type == "vla":
            print(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f} (color acc {train_color_acc:.2f}), "
                f"Validation Loss = {validation_loss:.4f} (color acc {validation_color_acc:.2f})"
            )
        else:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {validation_loss:.4f}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_color_acc"].append(train_color_acc)
        history["validation_color_acc"].append(validation_color_acc)

        if validation_loss is not None and validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(best_checkpoint_path)
            print(f"  New best validation loss ({validation_loss:.4f}); saved to {best_checkpoint_path}")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(checkpoint_path)
            print(f"Saved periodic checkpoint at epoch {epoch + 1} to {checkpoint_path}")

    save_checkpoint(checkpoint_path)
    print(f"Saved {policy_type} BC checkpoint to {checkpoint_path}")
    if best_validation_loss < float("inf"):
        print(f"Best validation loss was {best_validation_loss:.4f}; that checkpoint is at {best_checkpoint_path}")

    from scripts.plot_training import plot_training_curves

    plot_training_curves(checkpoint_path=checkpoint_path, output_path=debug_dir / "training_curves.png")

    return checkpoint_path


def main():
    train_bc(
        dataset_path=Path("outputs/rollouts/sim_vla_rollouts.h5"),
        checkpoint_dir=Path("outputs/checkpoints"),
        epochs=10,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=1e-6,
        train_split=0.9,
        seed=7,
        policy_type="goal_conditioned",
    )


if __name__ == "__main__":
    main()
