"""Rigid scanned-mesh fixtures using the shared robot and press-pull controller.

The scan is uniformly scaled (optional), yawed to its narrow side, and placed
on the table. Collision is a convex hull, explicitly recorded as an approximation.
No claim about a scan's real mass, compliance or friction is made here.
"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from contact_selection.sim.controller import PressPullConfig
from contact_selection.sim.scene import disable_adapter_object_collisions
from parameter_estimation.scene import create_scene_xml


def read_stl(path):
    raw = Path(path).read_bytes()
    count = int.from_bytes(raw[80:84], 'little')
    if len(raw) != 84 + count * 50:
        raise ValueError('Expected a binary STL')
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    vertices = np.frombuffer(raw, dtype=dtype, count=count, offset=84)['vertices'].reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError('Non-finite STL vertices')
    return np.unique(vertices.astype(float), axis=0)


def prepare_mesh(path, *, mass=0.4, friction=0.5, scale=1.0, yaw=0.0):
    if not np.isfinite([mass, friction, scale, yaw]).all() or min(mass, scale) <= 0 or friction < 0:
        raise ValueError('Invalid mesh physics or scale')
    vertices = read_stl(path) * scale
    # Keep the source upright; only yaw so the narrower footprint faces the pull.
    narrow_yaw = np.pi / 2 if np.ptp(vertices[:, 0]) > np.ptp(vertices[:, 1]) else 0.0
    angle = narrow_yaw + yaw
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    vertices = vertices @ rotation.T
    lo, hi = vertices.min(0), vertices.max(0)
    vertices -= np.r_[(lo[:2] + hi[:2]) / 2, lo[2]]
    hull = ConvexHull(vertices)
    # Export only hull vertices for fast compilation and deterministic collision.
    hull_vertices = vertices[hull.vertices]
    local_hull = ConvexHull(hull_vertices)
    faces = local_hull.simplices.copy()
    triangles = hull_vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flipped = np.sum(normals * local_hull.equations[:, :3], axis=1) < 0
    faces[flipped] = faces[flipped][:, [0, 2, 1]]
    with TemporaryDirectory(prefix='contact-mesh-') as tmp:
        tmp = Path(tmp)
        obj = tmp / 'payload.obj'
        obj.write_text(''.join('v %.9g %.9g %.9g\n' % tuple(v) for v in hull_vertices)
                       + ''.join('f %d %d %d\n' % tuple(f + 1) for f in faces))
        xml = Path(create_scene_xml((0,), out=tmp / 'scene.xml'))
        root = ET.parse(xml).getroot()
        body = root.find(".//body[@name='payload']")
        body.clear()
        body.attrib.update(name='payload', pos='0.60 0 0.0502')
        ET.SubElement(body, 'freejoint')
        ET.SubElement(root.find('asset'), 'mesh', name='external_payload', file=str(obj), inertia='convex')
        ET.SubElement(body, 'geom', name='payload', type='mesh', mesh='external_payload',
                      mass=str(mass), friction=f'{friction} 0.005 0.0001',
                      solref='0.002 1', rgba='0.3 0.65 0.8 1')
        ET.SubElement(body, 'site', name='site:payload', pos='0 0 0', size='0.003')
        near = hull_vertices[hull_vertices[:, 2] < 0.006]
        pivot = [near[:, 0].min(), 0, 0]
        ET.SubElement(body, 'site', name='site:obj_frame', pos=' '.join(map(str, pivot)), size='0.003')
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    disable_adapter_object_collisions(model)
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 10
    model.opt.noslip_iterations = 10
    model.geom_friction[model.geom('table').id, 0] = friction
    ball = model.geom('push_ball_col').id
    model.geom_friction[ball, 0] = 2.0
    model.geom_priority[ball] = 1
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pid = model.body('payload').id
    adr = model.jnt_qposadr[model.body_jntadr[pid]]
    # Planar CoM is an allowed selector input; align it with the robot's XZ plane.
    data.qpos[adr + 1] -= data.xipos[pid, 1]
    mujoco.mj_forward(model, data)
    cfg = replace(PressPullConfig(verbose=False), max_normal_speed=0.005,
                  arc_force_drop_fraction=0.1)
    return model, data, cfg, dict(source=str(Path(path).resolve()), scale=scale,
        yaw_rad=angle, mass_kg=mass, table_friction=friction, finger_friction=2.0,
        collision='single convex hull of scan; concavities are not validated',
        dimensions_m=np.ptp(vertices, axis=0).tolist())
