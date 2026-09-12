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
# mesh = trimesh.load_mesh("mujoco_irb120/robot/assets/objects/box/box_exp.stl")
mesh = trimesh.load_mesh("outputs/push2twin/rotate2construct_cloud.ply")
mesh.show()

print(f"mesh watertight: {mesh.is_watertight}, euler number: {mesh.euler_number}")

print(f"volume to convex hull ratio: {mesh.volume / mesh.convex_hull.volume}")


print(mesh.bounds)
obj_radius = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) / 2


# Now setup a 'virtual camera' and run a visibility test on the subsampled points.
# Object/mesh is at a candidate azimuth, theta. Use Trimesh ray-mesh intersection.


# Want to keep track of which points are visible and seen, for multiple different camera angles.
# For each camera angle, sample new points, cull those not visible, add seen to our list.
# Essentially, we'd be rotating the object, but easier to simulate rotating camera about the obj.

# Shift object so CoM is at origin.
mesh.apply_translation(-mesh.center_mass)

# Select random set of pts to test visibility on same set each time
subsamples, _ = trimesh.sample.sample_surface(mesh, 100)
covered_pts = np.zeros(len(subsamples), dtype=bool)

# Select a random camera starting angle
start_angle = np.random.randint(0, 360)
FEASIBLE_ROTATION = 90 # Realistically, the object (camera here) can be rotated in +-90deg steps.

def check_coverage(subsamples, cam_origin):
    for j in range(len(subsamples)):
            sample = subsamples[j]
            # Cast a ray from camera to the sample point.
            ray = sample - cam_origin
            # Check what's between the camera and the sample.
            locs, index_ray, index_tri = mesh.ray.intersects_location(cam_origin, ray, multiple_hits=True)
            if len(locs) == 0:
                continue  # shouldn't happen -- the ray is aimed straight at a surface point
    
            # `intersects_location` doesn't return hits sorted by distance, so locs[0]
            # isn't necessarily the closest one -- find it explicitly. And compare with
            # a tolerance, not `==`: the hit point is recomputed by the ray/triangle
            # solve, so it won't be bit-exact with the sampled point even when it's the
            # same point.
            dists = np.linalg.norm(locs - cam_origin, axis=1)
            nearest = locs[np.argmin(dists)]
            if np.allclose(nearest, sample, atol=1e-6):
                covered_pts[j] = True
    return covered_pts


n_angles = 10
for i in range(n_angles):
    theta = np.radians(i * 36)
    # Set the camera origin at a distance from the object.
    cam_origin = np.array([[obj_radius*np.cos(theta), obj_radius*np.sin(theta), 0]])
    # cam_direct = np.array([[-np.cos(theta), -np.sin(theta), 0]]) # look at zero

    covered_pts = check_coverage(subsamples, cam_origin)
    

print(f"Number of unique points seen from {n_angles} camera angles: {covered_pts.sum()} / {len(covered_pts)}")