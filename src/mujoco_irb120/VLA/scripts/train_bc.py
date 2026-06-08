from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from mujoco_irb120.VLA.data.npz_dataset import VLADataset
from mujoco_irb120.VLA.models.policy import TinyVLAPolicy


def _collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "state": torch.stack([item["state"] for item in batch]),
        "instruction": [item["instruction"] for item in batch],
        "action": torch.stack([item["action"] for item in batch]),
    }


def train_bc(
    dataset_path: Path,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    train_split: float,
    seed: int,
) -> Path:
    dataset = VLADataset(dataset_path)
    if len(dataset) < 2:
        raise ValueError("Need at least 2 samples to train and validate.")

    generator = torch.Generator().manual_seed(seed)
    train_len = max(1, int(len(dataset) * train_split))
    val_len = len(dataset) - train_len
    if val_len == 0:
        train_len -= 1
        val_len = 1
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVLAPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "vla_bc.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)

            pred = model(image, state, batch["instruction"])
            loss = loss_fn(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(action)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                state = batch["state"].to(device)
                action = batch["action"].to(device)
                pred = model(image, state, batch["instruction"])
                val_loss += loss_fn(pred, action).item() * len(action)
        val_loss /= len(val_set)

        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": 24,
                    "action_dim": 6,
                    "best_val_mse": best_val,
                },
                checkpoint_path,
            )

    print(f"Saved best checkpoint to {checkpoint_path}")
    return checkpoint_path

