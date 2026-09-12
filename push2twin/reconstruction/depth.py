"""
Synthetic pinhole depth camera: a stand-in for a real depth sensor, ray-cast
against a trimesh mesh.

This never had anything to do with TSDF fusion itself -- it just produces a
depth image, the way a real depth camera would. Fusion is a separate concern
downstream (rotate2construct.py uses open3d's `ScalableTSDFVolume` for that;
there used to be a hand-rolled `TSDFVolume` class in this file too, dropped
once open3d was adopted as the fusion backend -- see push2twin/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class PinholeCamera:
    """A simple pinhole camera, defined entirely by its own basis vectors."""

    origin: np.ndarray   # (3,) camera center
    forward: np.ndarray  # (3,) unit vector, viewing direction
    right: np.ndarray    # (3,) unit vector
    up: np.ndarray       # (3,) unit vector
    width: int
    height: int
    fov_deg: float

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / np.tan(np.radians(self.fov_deg) / 2.0)

    def ray_directions(self) -> np.ndarray:
        """(H*W, 3) unit ray directions, one per pixel, row-major (v then u)."""
        f = self.focal_px
        cx, cy = self.width / 2.0, self.height / 2.0
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x = (u + 0.5 - cx) / f
        y = (v + 0.5 - cy) / f
        dirs = (
            self.forward[None, None, :]
            + x[..., None] * self.right[None, None, :]
            + y[..., None] * self.up[None, None, :]
        ).reshape(-1, 3)
        return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def render_depth(mesh: trimesh.Trimesh, cam: PinholeCamera) -> np.ndarray:
    """
    Synthetic z-depth image (H, W), standing in for a real depth camera.
    NaN where nothing is hit.

    z-depth (the component of the hit along `cam.forward`), not Euclidean
    range along each ray -- matches how real depth cameras report distance,
    and what a TSDF fusion step (open3d or otherwise) expects so pixel and
    voxel depths are directly comparable.

    All rays batched into one `intersects_location` call, and the nearest
    hit per ray picked out vectorized (not a Python loop) -- same approach
    as trimesh_vis_test.py's check_coverage, which hit a real slowdown doing
    this per-point.
    """
    directions = cam.ray_directions()
    origins = np.repeat(cam.origin.reshape(1, 3), len(directions), axis=0)
    locs, index_ray, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)

    depth = np.full(cam.width * cam.height, np.nan)
    if len(locs) > 0:
        rel = locs - origins[index_ray]
        z = rel @ cam.forward
        dists = np.linalg.norm(rel, axis=1)
        order = np.lexsort((dists, index_ray))  # nearest hit per ray, vectorized
        sorted_rays = index_ray[order]
        first_of_each_ray = np.concatenate(([True], sorted_rays[1:] != sorted_rays[:-1]))
        nearest = order[first_of_each_ray]
        depth[index_ray[nearest]] = z[nearest]

    return depth.reshape(cam.height, cam.width)
