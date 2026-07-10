from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from environment import DomainRandomizationConfig, VLAIRB120Env
from robot.controllers.hw1_oracle_policy import HW1BinSortExpert
from task import BinSortTaskSpec, HW1_TASK
from util.debug_log import StepLogger, new_debug_dir, save_frame
from util.paths import REPO_ROOT
from util.rollout_dataset import HDF5Writer
from util.runtime import EpisodeVideoRecorder


def collect_sim_data(
    output_path: Path,
    episodes: int,
    seed: int,
    image_height: int = 128,
    image_width: int = 128,
    video_height: int = 720,
    video_width: int = 720,
    record_stride: int = 1,
    render: bool = False,
    task: BinSortTaskSpec = HW1_TASK,
    domain_randomization: DomainRandomizationConfig | dict | None = None,
    randomize_bin_layout: bool = False,
    randomize_bin_positions: bool = False,
) -> None:
    """Collect image, language, state, action tuples from MuJoCo."""
    if record_stride < 1:
        raise ValueError(f"record_stride must be >= 1, got {record_stride}")
    if randomize_bin_layout and randomize_bin_positions:
        raise ValueError("Use either randomize_bin_layout/swap_bins or randomize_bin_positions/randomize_bins, not both.")

    # Append each completed episode to one HDF5 file. This retains the simple
    # single-file workflow without ever holding the full run in RAM.
    writer = HDF5Writer(output_path)
    print(f"[collect_sim_data] Streaming episodes to {writer.path}")

    start = time.time()
    if seed is None or seed < 0:
        seed = int(np.random.SeedSequence().entropy) % (2**32 - 1)
        print(f"[collect_sim_data] Using random seed: {seed}")

    debug_dir = new_debug_dir("collect")
    print(f"[collect_sim_data] Debug logs (per-step traces, frames, plots) -> {debug_dir}")

    with VLAIRB120Env(
        max_sim_time=task.max_sim_time,
        render_mode="rgb_array",
        image_height=image_height,
        image_width=image_width,
        task=task,
        domain_randomization=domain_randomization,
        seed=seed,
    ) as env:
        combos = (
            [(color, swap) for color in task.colors for swap in (False, True)]
            if randomize_bin_layout
            else [(color, False) for color in task.colors]
        )
        for ep in range(episodes):
            cube_color, swap_bins_option = combos[ep % len(combos)]
            prompt = task.instruction_template.format(color=cube_color)
            reset_options = {
                "cube_color": cube_color,
                "swap_bins": swap_bins_option,
                "randomize_bins": randomize_bin_positions,
            }
            obs, info = env.reset(
                seed=seed + ep,
                options=reset_options,
            )
            video = None
            if render:
                video_path = (
                    REPO_ROOT
                    / "outputs"
                    / "videos"
                    / f"{output_path.stem}_collect_ep{ep + 1:03d}.mp4"
                )
                video = EpisodeVideoRecorder(video_path)
                video.capture(
                    env.capture_image(height=video_height, width=video_width),
                    info["sim_time"],
                    force=True,
                )
            expert = HW1BinSortExpert(env, cube_color=cube_color, task=env.task)
            done = False
            step = 0
            last_progress_second = -1
            step_logger = StepLogger()
            ep_debug_dir = debug_dir / f"ep{ep + 1:03d}"
            next_debug_frame_t = 0.0
            debug_frame_stride_s = 0.5

            # Held only for this one episode, then flushed to disk below -- this
            # is what keeps peak RAM bounded regardless of --episodes.
            images: list[np.ndarray] = []
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            instructions: list[str] = []
            cube_color_labels: list[str] = []
            swap_bins_labels: list[bool] = []
            randomize_bins_labels: list[bool] = []
            red_bin_xy: list[np.ndarray] = []
            blue_bin_xy: list[np.ndarray] = []
            red_bin_yaw: list[float] = []
            blue_bin_yaw: list[float] = []
            step_idx: list[int] = []
            success_by_step: list[bool] = []
            try:
                while not done:
                    action = expert.select_action()

                    should_record = step % record_stride == 0
                    if should_record:
                        image = env.capture_image()
                    next_obs, done, info = env.step(action)

                    step_logger.log(
                        t=info["sim_time"],
                        oracle_target_q=action,
                        actual_qpos=env.data.qpos[env.irb.joint_idx].copy(),
                        ee_xyz=env.data.site_xpos[env.irb.ee_site].copy(),
                        cube_xyz=env.get_cube_position(),
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

                    if should_record:
                        images.append(image.astype(np.uint8))
                        states.append(obs.astype(np.float32))
                        actions.append(action.astype(np.float32))
                        instructions.append(prompt)
                        cube_color_labels.append(cube_color)
                        swap_bins_labels.append(env.swap_bins)
                        randomize_bins_labels.append(env.randomize_bins)
                        red_bin_xy.append(env.task.bin_xy_by_color["red"].astype(np.float32))
                        blue_bin_xy.append(env.task.bin_xy_by_color["blue"].astype(np.float32))
                        red_bin_yaw.append(float(env.task.bin_yaw_by_color["red"]))
                        blue_bin_yaw.append(float(env.task.bin_yaw_by_color["blue"]))
                        step_idx.append(step)
                        success_by_step.append(bool(info["success"]))

                    obs = next_obs
                    step += 1
                    progress_second = int(info["sim_time"])
                    if progress_second != last_progress_second:
                        last_progress_second = progress_second
                        print(
                            f"  ep={ep + 1}/{episodes} t={info['sim_time']:.2f}s "
                            f"success={info['success']} done_reason={info['done_reason']}"
                        )
            finally:
                if video is not None:
                    video.close()
                step_logger.save(ep_debug_dir / "trace.npz", ep_debug_dir / "trace.png")

            writer.write_episode(
                ep,
                images=np.asarray(images, dtype=np.uint8),
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                instructions=np.asarray(instructions),
                cube_color=np.asarray(cube_color_labels),
                swap_bins=np.asarray(swap_bins_labels, dtype=np.bool_),
                randomize_bins=np.asarray(randomize_bins_labels, dtype=np.bool_),
                red_bin_xy=np.asarray(red_bin_xy, dtype=np.float32),
                blue_bin_xy=np.asarray(blue_bin_xy, dtype=np.float32),
                red_bin_yaw=np.asarray(red_bin_yaw, dtype=np.float32),
                blue_bin_yaw=np.asarray(blue_bin_yaw, dtype=np.float32),
                episode_idx=np.full(len(actions), ep, dtype=np.int32),
                step_idx=np.asarray(step_idx, dtype=np.int32),
                success=np.asarray(success_by_step, dtype=np.bool_),
            )

            print(
                f"Collected episode {ep + 1}/{episodes}: "
                f"color={cube_color}, swap_bins={env.swap_bins}, randomize_bins={env.randomize_bins}, sim_steps={step}, "
                f"recorded_samples={len(actions)}, "
                f"success={info['success']}, done_reason={info['done_reason']}"
            )

        sim_timestep = env.model.opt.timestep if env.model is not None else np.nan

    writer.finalize(
        record_stride=np.asarray(record_stride, dtype=np.int32),
        sim_timestep=np.asarray(sim_timestep, dtype=np.float32),
        max_sim_time=np.asarray(task.max_sim_time, dtype=np.float32),
        ft_bias_enabled=np.asarray(False, dtype=np.bool_),
        ft_bias_samples=np.asarray(0, dtype=np.int32),
    )
    print(f"Saved {writer.total_samples} VLA samples across {writer.num_episodes} episodes to {writer.path}")
    print(f"Collection wall time: {time.time() - start:.2f}s")
