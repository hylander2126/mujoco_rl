#!/usr/bin/env python3
"""Snapshot the ROS tool assembly and generate MuJoCo meshes/transforms.

Run with the repository venv; only numpy/scipy are required. The supported
Onshape glTF export uses embedded buffers, triangle primitives and identity
nodes. Unexpected features fail explicitly instead of silently moving geometry.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'mujoco_irb120/robot/assets/robot/hardware_tool'


def transform(origin):
    t = np.eye(4)
    if origin is not None:
        t[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def mesh_triangles(path):
    d = json.loads(path.read_text())
    buffers = [base64.b64decode(b['uri'].split(',', 1)[1]) for b in d['buffers']]
    def array(index):
        a = d['accessors'][index]
        assert 'sparse' not in a
        view = d['bufferViews'][a['bufferView']]
        dtype = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[a['componentType']]
        n = {'VEC3': 3, 'SCALAR': 1}[a['type']]
        stride = view.get('byteStride', np.dtype(dtype).itemsize * n)
        return np.ndarray((a['count'], n), dtype=dtype, buffer=buffers[view['buffer']],
                          offset=view.get('byteOffset', 0) + a.get('byteOffset', 0),
                          strides=(stride, np.dtype(dtype).itemsize)).copy()
    triangles = []
    for node in d['nodes']:
        assert not set(node) & {'matrix', 'translation', 'rotation', 'scale', 'children'}
        for primitive in d['meshes'][node['mesh']]['primitives']:
            assert primitive.get('mode', 4) == 4
            vertices = array(primitive['attributes']['POSITION'])
            triangles.append(vertices[array(primitive['indices']).reshape(-1, 3)])
    return np.concatenate(triangles)


def stl(path, triangles):
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-30)
    with path.open('wb') as f:
        f.write(b'Converted from hardware Onshape glTF'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(triangles)))
        for normal, triangle in zip(normals, triangles):
            f.write(struct.pack('<12fH', *normal, *triangle.ravel(), 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    args = parser.parse_args()
    package = args.repository.resolve() / 'irb120_control'
    (OUT / 'urdf').mkdir(parents=True, exist_ok=True)
    (OUT / 'meshes').mkdir(exist_ok=True)
    hashes = {}
    for relative in ['urdf/sensor_and_adapter_assembly.urdf', 'urdf/finger_assembly.urdf', 'urdf/irb120_with_finger.xacro']:
        source = package / relative
        shutil.copy2(source, OUT / relative)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    sensor = ET.parse(package / 'urdf/sensor_and_adapter_assembly.urdf').getroot()
    finger = ET.parse(package / 'urdf/finger_assembly.urdf').getroot()
    xacro = ET.parse(package / 'urdf/irb120_with_finger.xacro').getroot()
    props = {p.get('name'): p.get('value') for p in xacro.findall('{http://www.ros.org/wiki/xacro}property')}
    poses = {'root_sensor': transform(ET.Element('origin', rpy=f'0 0 {np.pi/2}'))}
    pending = list(sensor.findall('joint'))
    while pending:
        before = len(pending)
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent in poses:
                poses[joint.find('child').get('link')] = poses[parent] @ transform(joint.find('origin'))
                pending.remove(joint)
    # Joint lives inside the production xacro:unless branch.
    mount = xacro.find(".//joint[@name='sensor_body_to_finger_root']/origin")
    poses['root_finger'] = poses['sensor_body'] @ transform(ET.Element('origin',
        xyz=props['ft_body_length_xyz'], rpy=mount.get('rpy').replace('${pi}', str(np.pi))))
    for joint in finger.findall('joint')[::-1]:
        poses[joint.find('child').get('link')] = poses[joint.find('parent').get('link')] @ transform(joint.find('origin'))
    origin = poses['root_finger'][:3, 3]
    mj = ET.Element('mujoco')
    body = ET.SubElement(mj, 'body', name='ft_and_adapter_link', gravcomp='1')
    ET.SubElement(body, 'geom', name='tool_stack_col', type='cylinder', pos=f"{float(props['tool_stack_length'])/2} 0 0",
        size=f"0.045 {float(props['tool_stack_length'])/2}", euler=f'0 {np.pi/2} 0', mass='0', rgba='0.5 0.5 0.5 0', group='3')
    # Preserve the simulator wrench axes; locate moments at the finger mounting face.
    ET.SubElement(body, 'site', name='site:sensor', pos=' '.join(map(str, origin)), euler=f'0 {np.pi/2} {np.pi/2}', size='0.01')
    pusher = ET.SubElement(body, 'body', name='pusher_link', pos=' '.join(map(str, origin)), gravcomp='1')
    assets = ET.Element('mujoco')
    for assembly, tree in [('sensor_and_adapter_assembly', sensor), ('finger_assembly', finger)]:
        for link in tree.findall('link'):
            visual = link.find('visual')
            if visual is None:
                continue
            name = link.get('name')
            relative = f'meshes/{assembly}/{name}.gltf'
            source = package / relative
            destination = OUT / relative
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
            t = poses[name] @ transform(visual.find('origin'))
            triangles = mesh_triangles(source) @ t[:3, :3].T + t[:3, 3]
            if assembly == 'finger_assembly':
                triangles -= origin
            stl(OUT / 'meshes' / f'{name}.stl', triangles)
            ET.SubElement(assets, 'mesh', name=f'hardware_{name}', file=f'robot/hardware_tool/meshes/{name}.stl')
            ET.SubElement(pusher if assembly == 'finger_assembly' else body, 'geom',
                name='push_rod' if name == 'pusher_body' else f'hardware_{name}', type='mesh',
                mesh=f'hardware_{name}', mass='0', contype='0', conaffinity='0',
                rgba=visual.find('material/color').get('rgba'))
    # Measured aggregate mass and CoG supersede the tiny CAD ball mass/CoG.
    cog = poses['root_finger'][:3, :3] @ np.array([0, 0, float(props['finger_cog'])])
    rot = poses['root_finger'][:3, :3]
    cad = finger.find("link[@name='pusher_body']/inertial/inertia")
    inertia = rot @ np.diag([float(cad.get(k)) for k in ['ixx', 'iyy', 'izz']]) @ rot.T
    ET.SubElement(pusher, 'inertial', pos=' '.join(map(str, cog)), mass=props['finger_mass'],
        fullinertia=' '.join(map(str, [inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]])))
    ball = rot @ np.fromstring(props['finger_ball_center_xyz'], sep=' ')
    radius = float(props['finger_ball_radius'])
    ET.SubElement(pusher, 'geom', name='push_ball_col', type='sphere', pos=' '.join(map(str, ball)),
        size=str(radius), friction='1.5 0.02 0.001', condim='4', mass='0', rgba='0 0 0 0', group='3')
    for name, position in [('site:ball_center', ball), ('site:fingertip', ball + rot @ [0, 0, radius])]:
        ET.SubElement(pusher, 'site', name=name, pos=' '.join(map(str, position)), size='0.01')
    robot_path = OUT.parent / 'robot.xml'
    robot_text = robot_path.read_text()
    start = robot_text.index('<body name="ft_and_adapter_link"')
    depth = 0
    for tag in re.finditer(r'</?body\b[^>]*>', robot_text[start:]):
        depth += -1 if tag.group().startswith('</') else 1
        if depth == 0:
            end = start + tag.end()
            break
    ET.indent(body, space='    ', level=8)
    robot_path.write_text(robot_text[:start] + ET.tostring(body, encoding='unicode').rstrip() + robot_text[end:])
    for name, tree in [('tool.xml', mj), ('meshes.xml', assets)]:
        ET.indent(tree)
        ET.ElementTree(tree).write(OUT / name, encoding='unicode')
    manifest = dict(source_commit=subprocess.check_output(['git', '-C', str(args.repository), 'rev-parse', 'HEAD'], text=True).strip(),
        sha256=hashes, finger_root_in_tool0_m=origin.tolist(), ball_in_tool0_m=(origin+ball).tolist(),
        measured_finger_mass_kg=float(props['finger_mass']), measured_finger_cog_m=float(props['finger_cog']))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
