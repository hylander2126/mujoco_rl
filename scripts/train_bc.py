from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.policy import StateOnlyBCPolicy, TinyVLAPolicy


def train_bc(
    dataset_path,
    checkpoint_dir,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    train_split,
    seed,
    policy_type: str = "vla",
):
    """Train behavior cloning from collected MuJoCo demonstrations.

    `policy_type="vla"` trains the image + language + state policy so the model
    can see cube color and the text instruction. `policy_type="state_only"` keeps
    the old proprioceptive baseline around for quick debugging.
    """
    if policy_type not in {"vla", "state_only"}:
        raise ValueError(f"policy_type must be 'vla' or 'state_only', got {policy_type!r}")

    dataset_path = Path(dataset_path)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = np.load(dataset_path, allow_pickle=True)
    states = data["states"].astype(np.float32)
    expert_joint_targets = data["actions"].astype(np.float32)
    current_joint_positions = states[:, : expert_joint_targets.shape[1]]
    actions = expert_joint_targets - current_joint_positions
    episode_idx = data["episode_idx"]

    if policy_type == "vla":
        images = data["images"]
        instructions = data["instructions"].astype(str)
        print(
            f"Images shape: {images.shape}, States shape: {states.shape}, "
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

    state_mean = states[train_indices].mean(axis=0)
    state_std = states[train_indices].std(axis=0) + 1e-6
    action_mean = actions[train_indices].mean(axis=0)
    action_std = actions[train_indices].std(axis=0) + 1e-6

    class BCDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]
            state = (states[i] - state_mean) / state_std
            action = (actions[i] - action_mean) / action_std
            item = {
                "state": torch.tensor(state, dtype=torch.float32),
                "action": torch.tensor(action, dtype=torch.float32),
            }
            if policy_type == "vla":
                item["image"] = torch.from_numpy(images[i]).float().permute(2, 0, 1) / 255.0
                item["instruction"] = str(instructions[i])
            return item

    train_loader = DataLoader(
        BCDataset(train_indices),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        BCDataset(validation_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    hidden_dim = 128 if policy_type == "vla" else 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if policy_type == "vla":
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
            return model(image, state, instruction)
        return model(state)

    first_batch = next(iter(train_loader))
    with torch.no_grad():
        pred_action = predict(first_batch)
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
        total_count = 0
        for batch in loader:
            action = batch["action"].to(device)
            with torch.set_grad_enabled(is_training):
                pred_action = predict(batch)
                loss = loss_fn(pred_action, action)

                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            batch_size_this = action.shape[0]
            total_loss += loss.item() * batch_size_this
            total_count += batch_size_this

        if total_count == 0:
            return None
        return total_loss / total_count

    for epoch in range(epochs):
        train_loss = run_epoch(train_loader, optimizer)
        validation_loss = run_epoch(validation_loader, optimizer=None)
        if validation_loss is None:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = skipped (no validation samples)")
        else:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {validation_loss:.4f}")

    ckpt = {
        "model": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "policy_type": policy_type,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "state_mean": torch.tensor(state_mean, dtype=torch.float32),
        "state_std": torch.tensor(state_std, dtype=torch.float32),
        "action_mean": torch.tensor(action_mean, dtype=torch.float32),
        "action_std": torch.tensor(action_std, dtype=torch.float32),
        "action_mode": "joint_delta",
        "record_stride": int(data["record_stride"]) if "record_stride" in data.files else None,
        "ft_bias_enabled": bool(data["ft_bias_enabled"]) if "ft_bias_enabled" in data.files else None,
        "ft_bias_samples": int(data["ft_bias_samples"]) if "ft_bias_samples" in data.files else None,
        "dataset_path": str(dataset_path),
    }
    checkpoint_name = "vla_bc.pt" if policy_type == "vla" else "bc_only_states.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(ckpt, checkpoint_path)
    print(f"Saved {policy_type} BC checkpoint to {checkpoint_path}")
    return checkpoint_path


def main():
    train_bc(
        dataset_path=Path("outputs/rollouts/sim_vla_rollouts.npz"),
        checkpoint_dir=Path("outputs/checkpoints"),
        epochs=10,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=1e-6,
        train_split=0.9,
        seed=7,
        policy_type="vla",
    )


if __name__ == "__main__":
    main()
