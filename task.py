from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class BinSortTaskSpec:
    colors: tuple[str, str]
    home_q: np.ndarray
    cube_rgba: dict[str, np.ndarray]
    bin_site_by_color: dict[str, str]
    bin_xy_by_color: dict[str, np.ndarray]
    bin_yaw_by_color: dict[str, float]
    pre_drop_xyz_by_color: dict[str, np.ndarray]
    tip_tilt_rad: float = 0.75
    pre_drop_seconds: float = 2.0
    tip_seconds: float = 1.0
    drop_hold_seconds: float = 1.5
    return_home_seconds: float = 1.0
    max_sim_time: float = 5.0
    success_bin_radius: float = 0.11
    success_cube_max_z: float = 0.18
    success_cube_max_speed: float = 0.10
    success_hold_seconds: float = 0.35
    camera_name: str = "vla_cam"
    instruction_template: str = "sort the cube into the corresponding bin"

    @classmethod
    def default(cls) -> "BinSortTaskSpec":
        home_q = np.zeros(6, dtype=np.float32)
        home_q[4] = -1.5708

        return cls(
            colors=("red", "blue"),
            home_q=home_q,
            cube_rgba={
                "red": np.array([1.0, 0.05, 0.05, 1.0]),
                "blue": np.array([0.05, 0.2, 1.0, 1.0]),
            },
            bin_site_by_color={
                "red": "site:red_bin",
                "blue": "site:blue_bin",
            },
            bin_xy_by_color={
                "red": np.array([0.56, -0.20], dtype=float),
                "blue": np.array([0.56, 0.20], dtype=float),
            },
            bin_yaw_by_color={
                "red": float(np.arctan2(-0.20, 0.56)),
                "blue": float(np.arctan2(0.20, 0.56)),
            },
            pre_drop_xyz_by_color={
                "red": np.array([0.40, -0.07, 0.55], dtype=float),
                "blue": np.array([0.40, 0.07, 0.55], dtype=float),
            },
        )

def randomize_bin_pose(task: BinSortTaskSpec, rng: np.random.Generator) -> BinSortTaskSpec:
    """Return a task variant with randomized bin XY positions and Z yaw.

    Bins are sampled in front of the robot, at reachable radii, with enough
    separation to avoid overlap. Each bin yaw equals its radial placement angle,
    i.e. the same Z-axis rotation used to place it around the robot base.
    """
    color_a, color_b = task.colors

    min_distance = 0.40
    max_distance = 0.56
    min_separation = 0.32

    # Set the first bin's position and yaw randomly, then sample the second bin until it is far enough away.
    dist_a = float(rng.uniform(min_distance, max_distance))
    angle_a = float(rng.uniform(np.pi/6, 11*np.pi/6))
    pos_a = np.array([dist_a * np.cos(angle_a), dist_a * np.sin(angle_a)], dtype=float)

    # Loop until a valid second bin position is found.
    dist = 0
    tries = 0
    while dist < min_separation:
        dist_b = float(rng.uniform(min_distance, max_distance))
        angle_b = float(rng.uniform(np.pi/6, 11*np.pi/6))
        pos_b = np.array([dist_b * np.cos(angle_b), dist_b * np.sin(angle_b)], dtype=float)
        dist = np.linalg.norm(pos_a - pos_b)
        if tries > 100:
            raise RuntimeError("Failed to sample a valid second bin position after 100 tries.")
        tries += 1

    pre_drop_radius_a = max(min_distance - 0.05, dist_a - 0.15)
    pre_drop_radius_b = max(min_distance - 0.05, dist_b - 0.15)
    pos_pre_drop_a = np.array([
        pre_drop_radius_a * np.cos(angle_a),
        pre_drop_radius_a * np.sin(angle_a),
        0.55,
    ], dtype=float)
    pos_pre_drop_b = np.array([
        pre_drop_radius_b * np.cos(angle_b),
        pre_drop_radius_b * np.sin(angle_b),
        0.55,
    ], dtype=float)

    return replace(
        task,
        bin_xy_by_color={color_a: pos_a, color_b: pos_b},
        bin_yaw_by_color={color_a: angle_a, color_b: angle_b},
        pre_drop_xyz_by_color={color_a: pos_pre_drop_a, color_b: pos_pre_drop_b},
    )


def swap_bin_colors(task: BinSortTaskSpec) -> BinSortTaskSpec:
    """Return a task variant with the two bins' physical positions swapped.

    `bin_xy_by_color` and `pre_drop_xyz_by_color` are mirror-symmetric across
    the two colors in the default layout, so swapping which color occupies
    which physical slot is just swapping the dict values between the two
    color keys. Everything else (site names, success-radius checks,
    instruction text) is unaffected, since they key off color identity
    rather than a fixed physical position. The drop tilt direction is
    derived geometrically from `bin_xy_by_color` at drop time, so it needs
    no swapping here.
    """
    color_a, color_b = task.colors

    def swapped(by_color: dict) -> dict:
        return {color_a: by_color[color_b], color_b: by_color[color_a]}

    return replace(
        task,
        bin_xy_by_color=swapped(task.bin_xy_by_color),
        bin_yaw_by_color=swapped(task.bin_yaw_by_color),
        pre_drop_xyz_by_color=swapped(task.pre_drop_xyz_by_color),
    )


HW1_TASK = BinSortTaskSpec.default()
