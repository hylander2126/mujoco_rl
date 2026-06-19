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
    pre_drop_xyz_by_color: dict[str, np.ndarray]
    tray_tip_rad_by_color: dict[str, float]
    pre_drop_seconds: float = 2.0
    tip_seconds: float = 1.0
    drop_hold_seconds: float = 0.25
    return_home_seconds: float = 1.0
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
            pre_drop_xyz_by_color={
                "red": np.array([0.40, -0.07, 0.36], dtype=float),
                "blue": np.array([0.40, 0.07, 0.36], dtype=float),
            },
            tray_tip_rad_by_color={
                "red": -0.75,
                "blue": 0.75,
            },
        )


def swap_bin_colors(task: BinSortTaskSpec) -> BinSortTaskSpec:
    """Return a task variant with the two bins' physical positions swapped.

    `bin_xy_by_color`, `pre_drop_xyz_by_color`, and `tray_tip_rad_by_color`
    are all mirror-symmetric across the two colors in the default layout, so
    swapping which color occupies which physical slot is just swapping the
    dict values between the two color keys. Everything else (site names,
    success-radius checks, instruction text) is unaffected, since they key
    off color identity rather than a fixed physical position.
    """
    color_a, color_b = task.colors

    def swapped(by_color: dict) -> dict:
        return {color_a: by_color[color_b], color_b: by_color[color_a]}

    return replace(
        task,
        bin_xy_by_color=swapped(task.bin_xy_by_color),
        pre_drop_xyz_by_color=swapped(task.pre_drop_xyz_by_color),
        tray_tip_rad_by_color=swapped(task.tray_tip_rad_by_color),
    )


HW1_TASK = BinSortTaskSpec.default()
