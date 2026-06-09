from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mujoco_irb120.VLA.environment import DomainRandomizationConfig, VLAIRB120Env
from mujoco_irb120.VLA.models.policy import TinyVLAPolicy
from mujoco_irb120.VLA.task import BinSortTaskSpec, HW1_TASK


def evaluate_policy(
    checkpoint_path: Path,
    episodes: int,
    max_sim_time: float,
    render: bool,
    seed: int,
    image_height: int = 128,
    image_width: int = 128,
    task: BinSortTaskSpec = HW1_TASK,
    domain_randomization: DomainRandomizationConfig | dict | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TinyVLAPolicy(
        state_dim=checkpoint.get("state_dim", 24),
        action_dim=checkpoint.get("action_dim", 6),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    render_mode = "human" if render else "rgb_array"
    with VLAIRB120Env(
        max_sim_time=max_sim_time,
        render_mode=render_mode,
        image_height=image_height,
        image_width=image_width,
        task=task,
        domain_randomization=domain_randomization,
        seed=seed,
    ) as env:
        for ep in range(episodes):
            cube_color = task.colors[ep % len(task.colors)]
            prompt = task.instruction_template.format(color=cube_color)
            obs, info = env.reset(seed=seed + ep, options={"cube_color": cube_color})
            done = False
            step = 0
            while not done:
                if render:
                    env.render()
                image = env.capture_image()
                image_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                state_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(image_t, state_t, [prompt]).cpu().numpy()[0]
                obs, done, info = env.step(action)
                step += 1
            print(
                f"Eval episode {ep + 1}/{episodes}: color={cube_color}, "
                f"steps={step}, success={info['success']}, sim_time={info['sim_time']:.3f}s"
            )
