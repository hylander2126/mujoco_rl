"""
Prototype: rotate an object about world Z in the Genesis scene and fuse a
TSDF reconstruction from what a fixed 'onboard camera' sees along the way.

Stepped back to this on purpose: the robot-pushes-the-object version (still
in git history, and controllers/velocity_shove.py + main_genesis_sim.py's
single-push smoke test are untouched) only reliably landed one real push per
run, so reconstruction quality was bottlenecked on the manipulation, not the
reconstruction itself. This isolates the reconstruction pipeline against a
known-good motion (direct kinematic rotation, same as the very first version
of this script) so it can be developed and trusted on its own before being
re-coupled to a real push controller. The robot is still in the scene for
visual/scale context; it is not driven.

Reconstruction is `open3d`'s `ScalableTSDFVolume` (0.19.0), not a hand-rolled
one -- there used to be one (push2twin/reconstruction/tsdf.py's `TSDFVolume`
class), dropped once open3d was adopted: off-the-shelf, by choice, once both
existed side by side and worked equally well. `push2twin/reconstruction/
depth.py`'s `PinholeCamera` + `render_depth` are still used here for depth
synthesis (never "reinventing TSDF" to begin with -- a stand-in depth
sensor, which open3d fuses but doesn't render).

`integrate_depth()` below translates `PinholeCamera`'s own basis vectors into
the `(intrinsic, extrinsic)` pair open3d's `integrate()` expects: `R =
column_stack([right, up, forward])` is the camera-to-local rotation, so the
extrinsic (local-to-camera) is `[[R.T, -R.T @ origin], [0,0,0,1]]`. Validated
standalone against a known box (multi-view synthetic sweep, no Genesis
involved) before being trusted in this loop: recovered bounds within about a
voxel width and ~109% of true volume, watertight.

Frame: the object rotates in place (position fixed, quat teleported every
tick) about its own origin, so its own local/canonical frame is easy to stay
in -- the TSDF volume lives there, and only the fixed world camera's pose
(expressed in that local frame, recomputed each tick from the object's
current orientation) changes between captures.
"""

import argparse
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
import torch
import trimesh

torch.set_printoptions(sci_mode=False)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path("/tmp") / "mujoco_irb120-cache"
ROBOT_XML = REPO_ROOT / "mujoco_irb120" / "robot" / "assets" / "robot" / "genesis_robot.xml"
OBJECT_XML = REPO_ROOT / "mujoco_irb120" / "robot" / "assets" / "objects" / "genesis_object.xml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "push2twin"
SCENE_VIDEO_PATH = OUTPUT_DIR / "rotate2construct_scene.mp4"
RECON_VIDEO_PATH = OUTPUT_DIR / "rotate2construct_recon.mp4"
MESH_PATH = OUTPUT_DIR / "rotate2construct_mesh.stl"
OBJ_POS0 = np.array([0.5, 0.16, 0.25])  # matches main_genesis_sim.py's object placement

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))

sys.path.insert(0, str(REPO_ROOT))

import genesis as gs
from mujoco_irb120.robot.controllers.genesis_robot import quat_rotate
from push2twin.reconstruction.depth import PinholeCamera, render_depth
from util.runtime import EpisodeVideoRecorder

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rotate an object about world Z, fusing an open3d TSDF reconstruction from a fixed camera."
    )
    parser.add_argument("--total-deg", type=float, default=360.0, help="Total rotation about world Z.")
    parser.add_argument("--deg-per-step", type=float, default=0.6, help="Rotation increment per capture.")
    parser.add_argument("--tsdf-resolution", type=int, default=48,
                         help="Voxels along the volume's longest side (sets voxel_length for open3d).")
    parser.add_argument("--depth-res", type=int, default=64, help="Synthetic depth image resolution (square).")
    parser.add_argument("--fov-deg", type=float, default=50.0, help="Synthetic camera field of view.")
    return parser.parse_args()


def box_full_extents(object_xml: Path) -> np.ndarray:
    """
    Read the object's box half-size out of its MJCF instead of hardcoding it.

    genesis_object.xml's box is NOT the same size as box_exp.stl (0.05 0.05
    0.2 half-extents vs. 0.05 0.05 0.15) -- a hardcoded number would silently
    drift from whichever one someone edits next.
    """
    geom = ET.parse(object_xml).getroot().find(".//geom")
    half = np.array([float(v) for v in geom.get("size").split()])
    return half * 2.0


def z_rotation_quat(deg: float) -> np.ndarray:
    """Quaternion (Genesis/MuJoCo [w, x, y, z] order) for a rotation of `deg` about world Z."""
    half = np.radians(deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def integrate_depth(volume: o3d.pipelines.integration.ScalableTSDFVolume, depth: np.ndarray, cam: PinholeCamera) -> None:
    """PinholeCamera -> open3d's (RGBDImage, intrinsic, extrinsic). See module docstring."""
    depth_f32 = np.nan_to_num(depth, nan=0.0).astype(np.float32)
    color = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)  # NoColor mode -- content unused, shape required
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth_f32)),
        depth_scale=1.0,       # our depth is already float32 meters, not the usual uint16 millimeters
        depth_trunc=10.0,      # well beyond anything in this scene; real truncation is TSDFVolume's sdf_trunc
        convert_rgb_to_intensity=False,
    )
    f = cam.focal_px
    intrinsic = o3d.camera.PinholeCameraIntrinsic(cam.width, cam.height, f, f, cam.width / 2.0, cam.height / 2.0)

    R = np.column_stack([cam.right, cam.up, cam.forward])  # camera-to-local rotation
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ cam.origin

    volume.integrate(rgbd, intrinsic, extrinsic)


def extract_mesh(volume: o3d.pipelines.integration.ScalableTSDFVolume) -> trimesh.Trimesh | None:
    o3d_mesh = volume.extract_triangle_mesh()
    verts = np.asarray(o3d_mesh.vertices)
    if len(verts) == 0:
        return None
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(o3d_mesh.triangles), process=False)


def render_recon_frame(ax, mesh: trimesh.Trimesh | None, extents: np.ndarray, theta_deg: float):
    ax.cla()
    if mesh is not None and len(mesh.vertices):
        v = mesh.vertices
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c="tab:green", alpha=0.5)
    half = extents / 2.0 * 1.3
    ax.set_xlim(-half[0], half[0])
    ax.set_ylim(-half[1], half[1])
    ax.set_zlim(-half[2], half[2])
    ax.set_box_aspect(extents)
    n_verts = 0 if mesh is None else len(mesh.vertices)
    ax.set_title(f"open3d TSDF at {theta_deg:.0f} deg: {n_verts} verts")


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)

    extents = box_full_extents(OBJECT_XML)
    local_mesh = trimesh.creation.box(extents=extents)

    # Fixed world-frame 'onboard camera', outside the object's bounding
    # sphere with margin (too close and rays originate inside the mesh --
    # see trimesh_vis_test.py), looking at the object's fixed position (it
    # never translates in this version, so no tracking is needed).
    cam_world = OBJ_POS0 + np.array([-np.linalg.norm(extents) * 0.75, 0.0, 0.0])
    forward_world = OBJ_POS0 - cam_world
    forward_world /= np.linalg.norm(forward_world)
    up_hint = np.array([0.0, 0.0, 1.0])
    right_world = np.cross(forward_world, up_hint)
    right_world /= np.linalg.norm(right_world)
    up_world = np.cross(right_world, forward_world)

    voxel_length = float(np.max(extents) * 1.4 / args.tsdf_resolution)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=voxel_length * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    ############ Scene: copied verbatim from main_genesis_sim.py ############
    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=True,
            background_color=(0.92, 0.94, 0.97),
            ambient_light=(0.1, 0.1, 0.1),
        ),
        viewer_options=gs.options.ViewerOptions(
            res=None,
            camera_pos=(1.0, -1.0, 1.5),
            camera_lookat=(0.5, 0.0, 0.5),
            camera_fov=60,
            refresh_rate=60,
            enable_gui=False,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,  # headless -- see module docstring
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(pos=(0.0, 0.0, 0.05), size=(4.0, 4.0, 0.1), fixed=True),
        surface=gs.surfaces.Default(color=[1.0, 1.0, 1.0], opacity=1.0),
        material=gs.materials.Rigid(friction=0.1, needs_coup=True),
    )
    scene.add_entity(gs.morphs.MJCF(file=str(ROBOT_XML)))  # visual/scale context only -- not driven
    obj = scene.add_entity(
        gs.morphs.MJCF(file=str(OBJECT_XML), pos=tuple(OBJ_POS0)),
        surface=gs.surfaces.Default(color=[1.0, 0.0, 0.0], opacity=1.0),
        material=gs.materials.Rigid(friction=0.1, needs_coup=True, rho=4500.0),
    )

    scene_camera = scene.add_camera(res=(1280, 720), pos=(1.0, -1.0, 1.0), lookat=(0.5, 0.0, 0.5), fov=60, GUI=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene.start_recording(
        data_func=lambda: scene_camera.render(rgb=True)[0],
        rec_options=gs.recorders.VideoFile(filename=str(SCENE_VIDEO_PATH), hz=30),
    )

    scene.build()

    ############ Reconstruction video ############
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    n_ticks = max(1, int(round(args.total_deg / args.deg_per_step)))
    recon_recorder = EpisodeVideoRecorder(RECON_VIDEO_PATH, fps=max(1, n_ticks // 10))

    ############ Rotate in place, fusing a capture every tick ############
    recon = None
    for tick in range(n_ticks):
        theta_deg = (tick + 1) * args.deg_per_step
        quat = z_rotation_quat(theta_deg)
        obj.set_pos(OBJ_POS0)
        obj.set_quat(quat)
        scene.step()

        # Camera position and orientation expressed in the object's
        # (rotating) local frame: local = R(quat)^-1 @ (world - pos), and
        # R(quat)^-1 == R(conjugate) for a unit quaternion.
        quat_conj = quat * np.array([1.0, -1.0, -1.0, -1.0])
        cam_local = PinholeCamera(
            origin=quat_rotate(quat_conj, cam_world - OBJ_POS0),
            forward=quat_rotate(quat_conj, forward_world),
            right=quat_rotate(quat_conj, right_world),
            up=quat_rotate(quat_conj, up_world),
            width=args.depth_res, height=args.depth_res, fov_deg=args.fov_deg,
        )
        depth = render_depth(local_mesh, cam_local)
        integrate_depth(volume, depth, cam_local)

        recon = extract_mesh(volume)
        render_recon_frame(ax, recon, extents, theta_deg)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        recon_recorder.capture(frame, sim_time=tick, force=True)

    if recon is None:
        print("TSDF never accumulated enough for a surface -- no mesh saved.")
    else:
        print(f"Reconstructed mesh: {len(recon.vertices)} vertices, {len(recon.faces)} faces, "
              f"watertight={recon.is_watertight}.")
        recon.export(str(MESH_PATH))
        print(f"Saved reconstructed mesh to {MESH_PATH}")

    recon_recorder.close()
    scene.stop_recording()
    print(f"Saved scene video to {SCENE_VIDEO_PATH}")


if __name__ == "__main__":
    main()
