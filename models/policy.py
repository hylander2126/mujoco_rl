from __future__ import annotations

import torch
from torch import nn


class CharInstructionEncoder(nn.Module):
    """Small language encoder for short task instructions."""

    def __init__(self, embed_dim: int = 64, max_length: int = 96):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(128, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def tokenize(self, instructions: list[str], device: torch.device) -> torch.Tensor:
        tokens = torch.zeros((len(instructions), self.max_length), dtype=torch.long, device=device)
        for row, text in enumerate(instructions):
            encoded = text.encode("ascii", errors="ignore")[: self.max_length]
            if encoded:
                tokens[row, : len(encoded)] = torch.tensor(list(encoded), dtype=torch.long, device=device)
        return tokens

    def forward(self, instructions: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenize(instructions, device)
        mask = (tokens != 0).float().unsqueeze(-1)
        embedded = self.embedding(tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class StateOnlyBCPolicy(nn.Module):
    """Small state-to-action MLP for first-pass behavior cloning."""

    def __init__(self, state_dim: int = 24, action_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TinyVLAPolicy(nn.Module):
    """A compact image + language + state policy for behavior cloning."""

    def __init__(self, state_dim: int = 24, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Max-pool to a small 2x2 grid instead of averaging to 1x1: a small
            # colored cube next to two large, constant-colored bins gets washed
            # out by a global average, but survives a max over local patches.
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, hidden_dim),
            nn.ReLU(),
        )
        # Auxiliary head trained to predict cube color directly from the vision
        # embedding, so the vision tower can't free-ride on the language branch
        # explaining away all the color-dependent variance during training.
        self.color_head = nn.Linear(hidden_dim, 2)
        self.language = CharInstructionEncoder(embed_dim=64)
        self.state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        instruction: list[str],
        return_color_logits: bool = False,
    ) -> torch.Tensor:
        device = image.device
        z_img = self.vision(image)
        z_lang = self.language(instruction, device)
        z_state = self.state(state)
        action = self.action_head(torch.cat([z_img, z_lang, z_state], dim=-1))
        if return_color_logits:
            return action, self.color_head(z_img)
        return action
