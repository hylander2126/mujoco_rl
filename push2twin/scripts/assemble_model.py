"""
Prototype: combine a reconstructed mesh with mass/CoM/friction into one
standalone MJCF file.

First try / MVP, and it explicitly does NOT solve the frame-reconciliation
problem flagged in push2twin/README.md: mass/CoM are taken as given, in
whatever frame the caller says they're in, and written into the output
relative to the mesh's own origin. Nothing here checks that the estimator's
CoM frame (site:obj_frame, the tipping edge, in parameter_estimation) and the
mesh's frame (rotate2construct.py's object-local capture frame) actually
agree -- they probably don't yet, since no single object has been through
both stages. Point this at a real captured object once the estimator has
produced parameters for the same object, not before.

Two modes, matching what's already in this repo:
  - No --com given: just `mass=` on the mesh geom, same as
    mujoco_irb120/robot/assets/objects/heart/heart_exp.xml and L_exp.xml --
    MuJoCo computes CoM as the mesh's own centroid and derives inertia from
    the geometry assuming uniform density. Nothing to reconcile because
    nothing but mass is being asserted.
  - --com given: an explicit <inertial> tag with a `fullinertia` computed
    from the mesh (uniform-density assumption, same as above) then shifted
    to the given CoM via the parallel axis theorem, since trimesh computes
    the tensor about the mesh's own center of mass, not an arbitrary point.
    `fullinertia` (not `diaginertia`) so nothing here has to find principal
    axes -- MuJoCo's compiler diagonalizes it internally.

The default --mass (0.676 kg) is parameter_estimation/object_params.json's
ground-truth value for the "box" object -- a stand-in, not a real estimate
for whatever mesh you actually point this at. Override it once a real
estimate exists.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "push2twin"
DEFAULT_MESH = OUTPUT_DIR / "rotate2construct_mesh.stl"
OBJECT_PARAMS_JSON = REPO_ROOT / "parameter_estimation" / "object_params.json"


def default_mass() -> float:
    params = json.loads(OBJECT_PARAMS_JSON.read_text())
    return float(params["objects"]["box"]["mass_gt"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH,
                         help="Reconstructed mesh to wrap (default: rotate2construct.py's output).")
    parser.add_argument("--mass", type=float, default=None,
                         help="Object mass in kg. Defaults to object_params.json's box ground truth -- a stand-in.")
    parser.add_argument("--com", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                         help="Estimated CoM, in the mesh's own frame. Omit to let MuJoCo use the mesh centroid.")
    parser.add_argument("--friction", type=float, default=0.1,
                         help="Sliding friction (matches box_exp.xml's convention for the other two terms).")
    parser.add_argument("--name", type=str, default="push2twin_object")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output MJCF path (default: outputs/push2twin/<name>.xml).")
    return parser.parse_args()


def parallel_axis_shift(inertia_at_com: np.ndarray, mass: float, com: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shift a 3x3 inertia tensor from `com` to `target` (Huygens-Steiner)."""
    d = com - target
    return inertia_at_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def build_inertial_block(mesh: trimesh.Trimesh, mass: float, com: np.ndarray) -> ET.Element:
    if not mesh.is_watertight:
        # trimesh does NOT quietly fall back to the convex hull here -- found
        # by testing this against a real (open-topped, non-watertight) TSDF
        # reconstruction: `mesh.moment_inertia` on an open mesh isn't just
        # "off", it can come back with negative diagonal entries (not a
        # physically valid tensor at all -- its own volume/inertia integrals
        # assume a closed surface), which MuJoCo then rejects outright
        # ("inertia must have positive eigenvalues"). Compute mass properties
        # from the convex hull explicitly instead -- always watertight by
        # construction, so always a valid tensor, at the cost of pretending
        # any concavity is solid.
        print("WARNING: mesh is not watertight -- using its convex hull for mass properties instead "
              "(pretends any concavity is solid). Fine for a first try, not for a real one.")
        mesh = mesh.convex_hull
    else:
        mesh = mesh.copy()

    mesh.density = mass / mesh.volume  # uniform-density assumption
    inertia_at_mesh_com = mesh.moment_inertia
    inertia_at_target = parallel_axis_shift(inertia_at_mesh_com, mass, mesh.center_mass, com)
    ixx, iyy, izz = inertia_at_target[0, 0], inertia_at_target[1, 1], inertia_at_target[2, 2]
    ixy, ixz, iyz = inertia_at_target[0, 1], inertia_at_target[0, 2], inertia_at_target[1, 2]

    inertial = ET.Element("inertial")
    inertial.set("pos", f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
    inertial.set("mass", f"{mass:.6f}")
    inertial.set("fullinertia", f"{ixx:.8f} {iyy:.8f} {izz:.8f} {ixy:.8f} {ixz:.8f} {iyz:.8f}")
    return inertial


def build_mjcf(mesh_path: Path, name: str, mass: float, com: np.ndarray | None, friction: float, mesh: trimesh.Trimesh) -> ET.ElementTree:
    root = ET.Element("mujoco", model=name)

    # Absolute meshdir: simplest correct thing for a first-try prototype --
    # not portable across machines/checkouts, which is a real limitation, not
    # an oversight.
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(mesh_path.parent))

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name=f"{name}_mesh", file=mesh_path.name)

    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", name=name, pos="0 0 0")
    ET.SubElement(body, "joint", type="free", damping="0.01")

    geom_attrs = {
        "name": f"{name}_geom",
        "type": "mesh",
        "mesh": f"{name}_mesh",
        "friction": f"{friction} {friction} 0.0001",  # matches box_exp.xml / genesis_object.xml's convention
        "rgba": "0.8 0.2 0.2 1",
    }
    if com is None:
        # No explicit CoM asserted -- let MuJoCo derive CoM (mesh centroid)
        # and inertia from the geometry itself, same as heart_exp.xml /
        # L_exp.xml.
        geom_attrs["mass"] = f"{mass:.6f}"
        ET.SubElement(body, "geom", **geom_attrs)
    else:
        ET.SubElement(body, "geom", **geom_attrs)
        body.append(build_inertial_block(mesh, mass, np.asarray(com, dtype=float)))

    ET.SubElement(body, "site", name="site:payload", pos="0 0 0", size="0.02 0.02 0.02", type="box", rgba="1 1 0 0")
    # No site:obj_frame -- that's the tipping-edge frame the estimator uses,
    # and this script has no way to know where that is on a reconstructed
    # mesh. See module docstring: this is exactly the frame-reconciliation
    # problem left for later, not solved here.

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def main():
    args = parse_args()
    mass = args.mass if args.mass is not None else default_mass()
    out_path = args.out or (OUTPUT_DIR / f"{args.name}.xml")

    if args.com is not None:
        print(f"--com given ({args.com}): asserting it's already in {args.mesh.name}'s own frame. "
              "Nothing here verifies that -- see module docstring.")

    mesh = trimesh.load_mesh(str(args.mesh))
    tree = build_mjcf(args.mesh, args.name, mass, args.com, args.friction, mesh)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    print(f"Wrote {out_path}")

    # Prove it's actually loadable, not just well-formed XML.
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(out_path))
    body_id = model.body(args.name).id
    print(f"Loaded OK. MuJoCo computed: mass={model.body_mass[body_id]:.4f} kg, "
          f"CoM(local)={model.body_ipos[body_id]}, "
          f"diag inertia={model.body_inertia[body_id]}")


if __name__ == "__main__":
    main()
