"""
Generate a synthetic RGB-D turntable sequence for BundleSDF, in exactly the
dataset format its own YcbineoatReader expects (BundleTrack/scripts/
data_reader.py, read directly from the cloned repo rather than assumed from
the README):

    out_dir/
        rgb/000000.png ...      3-channel PNG
        depth/000000.png ...    uint16 PNG, millimeters (matches
                                 `cv2.imread(...)/1e3` in data_reader.py)
        masks/000000.png ...    uint8 PNG, 0 = background, else foreground
        cam_K.txt               3x3 intrinsics, whitespace-delimited
        gt_mesh.obj              ground-truth mesh, for comparison against
                                 BundleSDF's reconstruction -- not part of
                                 BundleSDF's own format, just kept alongside

One simple known object (a Genesis primitive box) rotates about world Z in
fixed increments -- a turntable sequence, camera fixed in world. This is not
an approximation of what BundleSDF expects: it has no built-in notion of
"camera moves" vs. "object moves", it only ever reasons about the object's
pose *relative to the camera* frame-to-frame from the RGBD+mask stream, so a
fixed camera watching a rotating object is already the direct, unmodified
input format -- nothing had to be adapted for this.

Deliberately isolated from push2twin/'s Genesis scaffolding (no robot, no
robot controller, no shared scene template) -- this PoC is only about
generating data BundleSDF can consume and running BundleSDF against it, per
Steven's brief. Some of the math (z-axis rotation quaternion, [w,x,y,z]
convention) is the same as push2twin's because it's the same Genesis
convention, not because this imports from there.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "synthetic_cube"
CACHE_ROOT = Path("/tmp") / "mujoco_irb120-cache"

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))

import genesis as gs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic RGB-D turntable sequence for BundleSDF.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--deg-per-frame", type=float, default=15.0, help="Rotation increment per frame.")
    parser.add_argument("--total-deg", type=float, default=360.0, help="Total rotation about world Z.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-deg", type=float, default=45.0)
    parser.add_argument("--cube-size", type=float, default=0.08, help="Full side length in meters.")
    return parser.parse_args()


def z_rotation_quat(deg: float) -> np.ndarray:
    """Quaternion (Genesis [w, x, y, z] order) for a rotation of `deg` about world Z."""
    half = np.radians(deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def main():
    args = parse_args()
    rgb_dir = args.out_dir / "rgb"
    depth_dir = args.out_dir / "depth"
    masks_dir = args.out_dir / "masks"
    for d in (rgb_dir, depth_dir, masks_dir):
        d.mkdir(parents=True, exist_ok=True)

    obj_pos = np.array([0.0, 0.0, args.cube_size / 2.0])  # resting on the ground plane, centered at world origin (x,y)

    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            segmentation_level="entity",  # camera.render(segmentation=True) then stores entity.idx per pixel
            ambient_light=(0.4, 0.4, 0.4),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    obj = scene.add_entity(
        gs.morphs.Box(pos=tuple(obj_pos), size=(args.cube_size,) * 3),
        surface=gs.surfaces.Default(color=(0.75, 0.25, 0.2, 1.0)),
        material=gs.materials.Rigid(),
    )

    # Fixed camera, looking at the object from a distance a few cube-widths
    # out -- close enough for BundleSDF's expected depth range, far enough
    # the object stays fully in frame through the whole rotation.
    cam_dist = args.cube_size * 6.0
    cam_pos = obj_pos + np.array([cam_dist, 0.0, args.cube_size * 1.5])
    camera = scene.add_camera(
        res=(args.width, args.height), pos=tuple(cam_pos), lookat=tuple(obj_pos), fov=args.fov_deg, GUI=False,
    )

    scene.build()

    K = camera.intrinsics.astype(np.float64)
    np.savetxt(args.out_dir / "cam_K.txt", K)

    n_frames = max(1, int(round(args.total_deg / args.deg_per_frame)))
    for i in range(n_frames):
        theta_deg = i * args.deg_per_frame
        obj.set_pos(obj_pos)
        obj.set_quat(z_rotation_quat(theta_deg))
        scene.step()

        rgb, depth, seg, _ = camera.render(rgb=True, depth=True, segmentation=True)

        id_str = f"{i:06d}"
        cv2.imwrite(str(rgb_dir / f"{id_str}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        depth_mm[depth <= 0] = 0  # invalid/no-hit pixels -> 0, matches BundleSDF's own "no depth" convention
        cv2.imwrite(str(depth_dir / f"{id_str}.png"), depth_mm)

        # Genesis's segmentation buffer reserves 0 for background/no-hit and
        # offsets real entities by +1 (confirmed empirically: `seg == obj.idx`
        # picked up the ground plane instead, not the box -- entity.idx
        # itself is 0-based with no background reservation).
        mask = np.where(seg == obj.idx + 1, 255, 0).astype(np.uint8)
        cv2.imwrite(str(masks_dir / f"{id_str}.png"), mask)

    # Ground truth, for visual comparison against BundleSDF's reconstruction
    # -- not part of BundleSDF's own input format.
    gt_mesh = trimesh.creation.box(extents=(args.cube_size,) * 3)
    gt_mesh.export(str(args.out_dir / "gt_mesh.obj"))

    print(f"Wrote {n_frames} frames to {args.out_dir}")
    print(f"  rgb/, depth/, masks/, cam_K.txt -- BundleSDF's expected input format")
    print(f"  gt_mesh.obj -- ground truth, for comparison only")


if __name__ == "__main__":
    main()
