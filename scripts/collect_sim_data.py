from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from environment import DomainRandomizationConfig, VLAIRB120Env
from robot.controllers.hw1_oracle_policy import HW1BinSortExpert
from task import BinSortTaskSpec, HW1_TASK
from util.paths import REPO_ROOT
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    episode_idx: list[int] = []
    step_idx: list[int] = []
    success_by_step: list[bool] = []

    start = time.time()
    if seed is None or seed < 0:
        seed = int(np.random.SeedSequence().entropy) % (2**32 - 1)
        print(f"[collect_sim_data] Using random seed: {seed}")

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
            samples_before_episode = len(actions)
            last_progress_second = -1
            try:
                while not done:
                    action = expert.select_action()

                    should_record = step % record_stride == 0
                    if should_record:
                        image = env.capture_image()
                    next_obs, done, info = env.step(action)

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
                        episode_idx.append(ep)
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

            print(
                f"Collected episode {ep + 1}/{episodes}: "
                f"color={cube_color}, swap_bins={env.swap_bins}, randomize_bins={env.randomize_bins}, sim_steps={step}, "
                f"recorded_samples={len(actions) - samples_before_episode}, "
                f"success={info['success']}, done_reason={info['done_reason']}"
            )

    np.savez_compressed(
        output_path,
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
        episode_idx=np.asarray(episode_idx, dtype=np.int32),
        step_idx=np.asarray(step_idx, dtype=np.int32),
        success=np.asarray(success_by_step, dtype=np.bool_),
        record_stride=np.asarray(record_stride, dtype=np.int32),
        sim_timestep=np.asarray(env.model.opt.timestep if env.model is not None else np.nan, dtype=np.float32),
        max_sim_time=np.asarray(task.max_sim_time, dtype=np.float32),
        ft_bias_enabled=np.asarray(False, dtype=np.bool_),
        ft_bias_samples=np.asarray(0, dtype=np.int32),
    )
    print(f"Saved {len(actions)} VLA samples to {output_path}")
    print(f"Collection wall time: {time.time() - start:.2f}s")
