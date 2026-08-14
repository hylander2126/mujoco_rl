import trimesh
import numpy as np
from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path("/tmp") / "mujoco_irb120-cache"
ROBOT_XML = REPO_ROOT / "mujoco_irb120" / "robot" / "assets" / "robot" / "genesis_robot.xml"
OBJECT_XML = REPO_ROOT / "mujoco_irb120" / "robot" / "assets" / "objects" / "genesis_object.xml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "push2twin"
VIDEO_PATH = OUTPUT_DIR / "genesis_sim.mp4"


trimesh.util.attach_to_log()

print(os.getcwd())

# Load mesh from file (box object)
mesh = trimesh.load_mesh("mujoco_irb120/robot/assets/objects/box/box_exp.stl")

print(f"mesh watertight: {mesh.is_watertight}, euler number: {mesh.euler_number}")

print(f"volume to convex hull ratio: {mesh.volume / mesh.convex_hull.volume}")

# Show not available on this remote system, so take a snapshot instead
# mesh.show()

print(mesh.bounds)

subsample = trimesh.sample.sample_surface(mesh, 100)

# Now setup a 'virtual camera' and run a visibility test on the subsampled points.
# Object/mesh is at a candidate azimuth, theta. Use Trimesh ray-mesh intersection.

theta = np.radians(45)
cam_origin = np.array([[-1, 0.05, 0.05]])
# cam_direct = np.array([[np.cos(theta), np.sin(theta), 0]])
cam_direct = np.array([[1, 0, 0]])

locs, index_ray, index_tri = mesh.ray.intersects_location(cam_origin, cam_direct, multiple_hits=True)
print(f"Number of visible points: {len(locs)}") # 4 because each box side is an acrylic sheet. Hollow inside.

# Want to keep track of which points are visible and seen, for multiple different camera angles.
# For each camera angle, sample new points, cull those not visible, add seen to our list.
# Essentially, we'd be rotating the object, but easier to simulate rotating camera about the obj.

# Shift object so CoM is at origin.
mesh.apply_translation(-mesh.center_mass)

viewed_pts = []

for i in range(10):
    theta = np.radians(i * 36)
    cam_origin = np.array([[np.cos(theta), np.sin(theta), 0]])
    cam_direct = np.array([[-np.cos(theta), -np.sin(theta), 0]]) # look at zero

    locs, index_ray, index_tri = mesh.ray.intersects_location(cam_origin, cam_direct, multiple_hits=False)

    viewed_pts.append(locs)

print(np.shape(viewed_pts))
viewed_pts = list(set(viewed_pts)) # unique points
print(f"Number of unique points seen from 10 camera angles: {len(viewed_pts)}")