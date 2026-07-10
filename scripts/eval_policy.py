from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from environment import DomainRandomizationConfig, VLAIRB120Env
from models.policy import GoalConditionedBCPolicy, StateOnlyBCPolicy, TinyVLAPolicy
from task import BinSortTaskSpec, HW1_TASK
from util.debug_log import StepLogger, new_debug_dir, save_frame
from util.paths import REPO_ROOT
from util.runtime import EpisodeVideoRecorder, select_torch_device

# Clamps predicted per-step joint deltas before converting them to position
# targets. Tuned once for this arm/task and not expected to need tuning per run.
MAX_JOINT_DELTA = 0.02


def evaluate_policy(
    checkpoint_path: Path,
    episodes: int,
    render: bool,
    seed: int,
    max_sim_time: float | None = None,
    image_height: int = 128,
    image_width: int = 128,
    video_height: int = 720,
    video_width: int = 720,
    control_stride: int | None = None,
    task: BinSortTaskSpec = HW1_TASK,
    domain_randomization: DomainRandomizationConfig | dict | None = None,
    bin_layout: str = "normal",
) -> None:
    if bin_layout not in {"normal", "swapped", "random", "randomized"}:
        raise ValueError(
            f"bin_layout must be 'normal', 'swapped', 'random', or 'randomized', got {bin_layout!r}"
        )
    if seed is None or seed < 0:
        seed = int(np.random.SeedSequence().entropy) % (2**32 - 1)
        print(f"[eval_policy] Using random seed: {seed}")

    if max_sim_time is None:
        # A learned policy may need more time than the scripted oracle to
        # still land the cube, so this stays overridable at eval time even
        # though collection (which runs the fixed-timing oracle) does not.
        max_sim_time = task.max_sim_time
    if max_sim_time <= 0.0:
        raise ValueError(f"max_sim_time must be > 0, got {max_sim_time}")

    debug_dir = new_debug_dir("eval")
    print(f"[eval_policy] Debug logs (per-step traces, frames, plots) -> {debug_dir}")

    device = select_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_type = checkpoint.get("policy_type", "vla")
    action_mode = checkpoint.get("action_mode")
    trained_record_stride = checkpoint.get("record_stride")
    trained_ft_bias_enabled = checkpoint.get("ft_bias_enabled")
    if control_stride is None:
        # record_stride only controls how densely the training *dataset* was
        # sampled; the oracle expert still moved every physics step during
        # collection. control_stride, by contrast, throttles the actual
        # robot at eval time (the held joint-position target between
        # updates is a no-op, not a step), so it should not be derived from
        # record_stride. 1 tracks the demonstrations best regardless of how
        # the training data was sampled.
        control_stride = 1
    if control_stride < 1:
        raise ValueError(f"control_stride must be >= 1, got {control_stride}")
    if trained_record_stride and trained_record_stride > 1:
        print(
            f"[eval_policy] Warning: checkpoint was trained from record_stride={trained_record_stride} data. "
            "A dense record_stride=1 dataset is usually needed for stable one-step behavior cloning."
        )
    if trained_ft_bias_enabled is not None and bool(trained_ft_bias_enabled):
        print(
            "[eval_policy] Warning: checkpoint was trained with ft_bias_enabled=True, "
            "but this implementation evaluates with F/T bias disabled. Observation normalization may be mismatched."
        )
    if control_stride > 1:
        print(
            f"[eval_policy] Warning: control_stride={control_stride}. "
            "For this one-step BC policy, control_stride=1 usually tracks the demonstrations best."
        )
    if action_mode != "joint_delta":
        raise ValueError(f"Expected a joint_delta checkpoint, got action_mode={action_mode!r}")
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    goal_mean = None
    goal_std = None
    if policy_type == "goal_conditioned":
        if checkpoint.get("goal_encoding") != "selected_bin_xy_sin_yaw_cos_yaw_progress":
            raise ValueError(f"Unsupported goal encoding: {checkpoint.get('goal_encoding')!r}")
        model = GoalConditionedBCPolicy(
            state_dim=checkpoint.get("state_dim", 24),
            goal_dim=checkpoint.get("goal_dim", 5),
            action_dim=checkpoint.get("action_dim", 6),
            hidden_dim=checkpoint.get("hidden_dim", 256),
        ).to(device)
        goal_mean = np.asarray(checkpoint["goal_mean"], dtype=np.float32)
        goal_std = np.asarray(checkpoint["goal_std"], dtype=np.float32)
    elif policy_type == "state_only":
        model = StateOnlyBCPolicy(
            state_dim=checkpoint.get("state_dim", 24),
            action_dim=checkpoint.get("action_dim", 6),
            hidden_dim=checkpoint.get("hidden_dim", 256),
        ).to(device)
    elif policy_type == "vla":
        model = TinyVLAPolicy(
            state_dim=checkpoint.get("state_dim", 24),
            action_dim=checkpoint.get("action_dim", 6),
            hidden_dim=checkpoint.get("hidden_dim", 128),
        ).to(device)
    else:
        raise ValueError(f"Unsupported policy_type in checkpoint: {policy_type!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with VLAIRB120Env(
        max_sim_time=max_sim_time,
        render_mode="rgb_array",
        image_height=image_height,
        image_width=image_width,
        task=task,
        domain_randomization=domain_randomization,
        seed=seed,
    ) as env:
        results_by_layout: dict[str, list[bool]] = {"normal": [], "swapped": [], "randomized": []}
        for ep in range(episodes):
            cube_color = task.colors[ep % len(task.colors)]
            prompt = task.instruction_template.format(color=cube_color)
            swap_bins_option = "random" if bin_layout == "random" else (bin_layout == "swapped")
            randomize_bins_option = bin_layout == "randomized"
            obs, info = env.reset(
                seed=seed + ep,
                options={
                    "cube_color": cube_color,
                    "swap_bins": swap_bins_option,
                    "randomize_bins": randomize_bins_option,
                },
            )
            goal_xy = env.task.bin_xy_by_color[cube_color].astype(np.float32)
            goal_yaw = float(env.task.bin_yaw_by_color[cube_color])
            video = None
            if render:
                video_path = (
                    REPO_ROOT
                    / "outputs"
                    / "videos"
                    / f"{checkpoint_path.stem}_eval_ep{ep + 1:03d}.mp4"
                )
                video = EpisodeVideoRecorder(video_path)
                video.capture(
                    env.capture_image(height=video_height, width=video_width),
                    info["sim_time"],
                    force=True,
                )
            done = False
            step = 0
            action = env.data.qpos[env.irb.joint_idx].copy().astype(np.float32)
            raw_action = np.zeros(6, dtype=np.float32)
            step_logger = StepLogger()
            ep_debug_dir = debug_dir / f"ep{ep + 1:03d}"
            next_debug_frame_t = 0.0
            debug_frame_stride_s = 0.5
            try:
                while not done:
                    if step % control_stride == 0:
                        state = ((obs.astype(np.float32) - state_mean) / state_std).astype(np.float32)
                        state_t = torch.from_numpy(state).unsqueeze(0).to(device)
                        if policy_type == "goal_conditioned":
                            progress = np.clip(info["sim_time"] / task.max_sim_time, 0.0, 1.0)
                            goal = np.asarray(
                                [goal_xy[0], goal_xy[1], np.sin(goal_yaw), np.cos(goal_yaw), progress],
                                dtype=np.float32,
                            )
                            normalized_goal = ((goal - goal_mean) / goal_std).astype(np.float32)
                            goal_t = torch.from_numpy(normalized_goal).unsqueeze(0).to(device)
                            with torch.no_grad():
                                normalized_action = model(state_t, goal_t).cpu().numpy()[0]
                        elif policy_type == "state_only":
                            with torch.no_grad():
                                normalized_action = model(state_t).cpu().numpy()[0]
                        else:
                            image = env.capture_image()
                            image_t = (
                                torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                            )
                            with torch.no_grad():
                                normalized_action = model(image_t, state_t, [prompt]).cpu().numpy()[0]
                        raw_action = (normalized_action * action_std) + action_mean
                        action = _safe_delta_joint_action(
                            raw_delta=raw_action,
                            current_q=obs[:6],
                            q_min=env.irb.q_min,
                            q_max=env.irb.q_max,
                            max_joint_delta=MAX_JOINT_DELTA,
                        )

                    obs, done, info = env.step(action)
                    step += 1

                    step_logger.log(
                        t=info["sim_time"],
                        qpos=env.data.qpos[env.irb.joint_idx].copy(),
                        ee_xyz=env.data.site_xpos[env.irb.ee_site].copy(),
                        cube_xyz=env.get_cube_position(),
                        goal=(
                            goal
                            if policy_type == "goal_conditioned"
                            else np.asarray([goal_xy[0], goal_xy[1], np.sin(goal_yaw), np.cos(goal_yaw)])
                        ),
                        raw_delta=raw_action,
                        action=action,
                    )
                    if info["sim_time"] + 1e-9 >= next_debug_frame_t or done:
                        save_frame(
                            env.capture_image(height=video_height, width=video_width),
                            ep_debug_dir / "frames" / f"t{info['sim_time']:.2f}.png",
                        )
                        next_debug_frame_t += debug_frame_stride_s

                    if video is not None and (done or video.is_frame_due(info["sim_time"])):
                        video.capture(
                            env.capture_image(height=video_height, width=video_width),
                            info["sim_time"],
                            force=done,
                        )
            finally:
                if video is not None:
                    video.close()
                step_logger.save(ep_debug_dir / "trace.npz", ep_debug_dir / "trace.png")
            layout_label = "randomized" if info.get("randomize_bins") else ("swapped" if info["swap_bins"] else "normal")
            results_by_layout[layout_label].append(bool(info["success"]))
            print(
                f"Eval episode {ep + 1}/{episodes}: color={cube_color}, layout={layout_label}, "
                f"steps={step}, success={info['success']}, sim_time={info['sim_time']:.3f}s"
            )

        for layout_label, successes in results_by_layout.items():
            if not successes:
                continue
            rate = sum(successes) / len(successes)
            print(f"Success rate ({layout_label} bin layout, n={len(successes)}): {rate:.2%}")


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
