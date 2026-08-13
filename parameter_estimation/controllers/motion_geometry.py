"""Small geometry helpers shared by motion controllers.

Ported verbatim from the real-robot workspace at
`irb120_ws/src/irb120_ros2/irb120_control/irb120_control/util/motion_geometry.py`.
Pure math with no ROS dependency, so sim and hardware stay bit-identical here.
Keep the two copies in sync — the arc geometry is the part of the press-and-pull
controller most likely to silently diverge between platforms.
"""

from __future__ import annotations

import math


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def quat_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract Y-axis pitch from a quaternion, in radians."""
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    return math.asin(clamp(sin_pitch, 1.0))


def arc_angle_xz(x: float, z: float, center_x: float, center_z: float = 0.0) -> float:
    """Angle from +Z toward +X for an XZ-plane arc."""
    return math.atan2(x - center_x, z - center_z)


def arc_velocity_xz(
    theta: float,
    tangential_speed: float,
    radial_speed: float,
) -> tuple[float, float]:
    """Return (vx, vz) for an XZ arc.

    Positive tangential speed follows decreasing theta: toward -X for theta
    near zero. Positive radial speed moves outward from the arc center.
    """
    tangent_x = -math.cos(theta)
    tangent_z = math.sin(theta)
    radial_x = math.sin(theta)
    radial_z = math.cos(theta)
    vx = tangential_speed * tangent_x + radial_speed * radial_x
    vz = tangential_speed * tangent_z + radial_speed * radial_z
    return vx, vz


def radial_force_xz(theta: float, force_x: float, force_z: float) -> float:
    """Magnitude of force projected onto an XZ arc's outward radial direction."""
    return abs(force_x * math.sin(theta) + force_z * math.cos(theta))


def tangent_force_xz(theta: float, force_x: float, force_z: float) -> float:
    """Signed force projected onto an XZ arc's tangential direction.

    Inlined in the hardware controller as `ArcStatic._tangent_force`; hoisted
    here because the ARC exit test depends on its sign convention.
    """
    return force_x * math.cos(theta) - force_z * math.sin(theta)
