from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mujoco_irb120.VLA.environment import DomainRandomizationConfig, VLAIRB120Env
from mujoco_irb120.VLA.models.policy import StateOnlyBCPolicy, TinyVLAPolicy
from mujoco_irb120.VLA.task import BinSortTaskSpec, HW1_TASK


def evaluate_policy(
    checkpoint_path: Path,
    episodes: int,
    max_sim_time: float,
    render: bool,
    seed: int,
    image_height: int = 128,
    image_width: int = 128,
    control_stride: int = 1,
    max_joint_delta: float = 0.02,
    task: BinSortTaskSpec = HW1_TASK,
    domain_randomization: DomainRandomizationConfig | dict | None = None,
) -> None:
    if control_stride < 1:
        raise ValueError(f"control_stride must be >= 1, got {control_stride}")
    if max_joint_delta <= 0.0:
        raise ValueError(f"max_joint_delta must be > 0, got {max_joint_delta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_type = checkpoint.get("policy_type", "vla")
    action_mode = checkpoint.get("action_mode")
    trained_record_stride = checkpoint.get("record_stride")
    if trained_record_stride and trained_record_stride > 1:
        print(
            f"[eval_policy] Warning: checkpoint was trained from record_stride={trained_record_stride} data. "
            "A dense record_stride=1 dataset is usually needed for stable one-step behavior cloning."
        )
    if control_stride > 1:
        print(
            f"[eval_policy] Warning: control_stride={control_stride}. "
            "For this one-step BC policy, control_stride=1 usually tracks the demonstrations best."
        )
    if action_mode != "joint_delta":
        raise ValueError(f"Expected a joint_delta checkpoint, got action_mode={action_mode!r}")
    if policy_type == "state_only":
        model = StateOnlyBCPolicy(
            state_dim=checkpoint.get("state_dim", 24),
            action_dim=checkpoint.get("action_dim", 6),
            hidden_dim=checkpoint.get("hidden_dim", 256),
        ).to(device)
        state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
        state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
        action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
        action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    else:
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
            action = env.data.qpos[env.irb.joint_idx].copy().astype(np.float32)
            while not done:
                if render:
                    env.render()

                if step % control_stride == 0:
                    if policy_type == "state_only":
                        state = ((obs.astype(np.float32) - state_mean) / state_std).astype(np.float32)
                        state_t = torch.from_numpy(state).unsqueeze(0).to(device)
                        with torch.no_grad():
                            normalized_action = model(state_t).cpu().numpy()[0]
                        raw_action = (normalized_action * action_std) + action_mean
                    else:
                        image = env.capture_image()
                        image_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                        state_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            raw_action = model(image_t, state_t, [prompt]).cpu().numpy()[0]
                    action = _safe_delta_joint_action(
                        raw_delta=raw_action,
                        current_q=obs[:6],
                        q_min=env.irb.q_min,
                        q_max=env.irb.q_max,
                        max_joint_delta=max_joint_delta,
                    )

                obs, done, info = env.step(action)
                step += 1
            print(
                f"Eval episode {ep + 1}/{episodes}: color={cube_color}, "
                f"steps={step}, success={info['success']}, sim_time={info['sim_time']:.3f}s"
            )


def _safe_delta_joint_action(
    raw_delta: np.ndarray,
    current_q: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    max_joint_delta: float,
) -> np.ndarray:
    """Clamp predicted joint deltas before converting them to position targets."""
    raw_delta = np.asarray(raw_delta, dtype=float).reshape(6)
    current_q = np.asarray(current_q, dtype=float).reshape(6)
    delta = np.clip(raw_delta, -max_joint_delta, max_joint_delta)
    return np.clip(current_q + delta, q_min, q_max).astype(np.float32)
