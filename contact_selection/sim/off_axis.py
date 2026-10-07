"""Matched-contact diagnostics: separate applied torque from supported rotation."""
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation


def arc_diagnostics(arrays):
    """Signed rotation uses the same world-frame ARC-onset reference as labels."""
    phase = arrays.get('diagnostic_state_id', arrays['state_id_hist'])
    indices = np.flatnonzero(phase == 3)
    if not len(indices):
        return None
    # Older controller pose logs precede its kinematics refresh by up to one tick.
    matrices = (arrays['object_rotation_world'][indices] if 'object_rotation_world' in arrays
                else arrays['obj_pose_hist'][indices, :3, :3])
    signed = np.rad2deg(Rotation.from_matrix(matrices @ matrices[0].T).as_rotvec())
    magnitude = np.linalg.norm(signed[:, [0, 2]], axis=1)
    peak = int(np.argmax(magnitude))
    result = {'peak_off_axis_deg': float(magnitude[peak]),
              'signed_roll_at_peak_deg': float(signed[peak, 0]),
              'signed_yaw_at_peak_deg': float(signed[peak, 2])}
    for key in ('fingertip_torque_about_com_nm', 'table_torque_about_com_nm',
                'total_contact_torque_about_com_nm'):
        if key in arrays:
            values = arrays[key][indices]
            result[key + '_mean'] = values.mean(axis=0).tolist()
            result[key + '_off_axis_rms'] = float(np.sqrt(np.mean(np.sum(values[:, [0, 2]] ** 2, axis=1))))
    return result


def plot_diagnostic(directory: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from contact_selection.dataset import write_json

    rows = []
    for path in sorted(directory.glob('center_*_offset_*.json')):
        row = json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as arrays:
            row['analysis'] = arc_diagnostics(arrays)
        # The stored evaluator scalar is authoritative for the magnitude panel.
        row['analysis']['peak_off_axis_deg'] = row['metrics']['max_off_axis_deg']
        rows.append(row)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for center_y, color in [(0.08, 'tab:orange'), (0.0, 'tab:blue')]:
        group = sorted([r for r in rows if r['center_y'] == center_y], key=lambda r: r['y_offset'])
        if not group:
            continue
        y = np.array([r['y_offset'] for r in group]) * 1000
        a = [r['analysis'] for r in group]
        label = f'Box world Y = {center_y:.2f} m'
        axes[0, 0].plot(y, [v['peak_off_axis_deg'] for v in a], 'o-', color=color, label=label)
        axes[0, 1].plot(y, [v['signed_yaw_at_peak_deg'] for v in a], 'o-', color=color, label=label)
        axes[1, 0].plot(y, [v['fingertip_torque_about_com_nm_off_axis_rms'] for v in a], 'o-', color=color, label=label)
        axes[1, 1].plot(y, [v['total_contact_torque_about_com_nm_off_axis_rms'] for v in a], 'o-', color=color, label=label)
    titles = ['Measured off-axis rotation', 'Signed yaw at peak off-axis rotation',
              'Fingertip off-axis torque about COM', 'Net contact off-axis torque about COM']
    labels = ['Peak during ARC (degrees)', 'World-Z rotation vector (degrees)',
              'RMS during ARC (N m)', 'RMS during ARC (N m)']
    for ax, title, label in zip(axes.flat, titles, labels):
        ax.set(title=title, xlabel='Contact Y − initial COM Y (mm)', ylabel=label)
        ax.axvline(0, color='gray', lw=0.7)
        ax.axhline(0, color='gray', lw=0.7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle('Same box, contact X = 0.580 m, table friction 0.5, adapter collisions disabled\n'
                 'Physical placement relative to the robot changes symmetry; plotting coordinates alone do not')
    fig.savefig(directory / 'off_axis_comparison.png', dpi=160)
    plt.close(fig)
    write_json(directory / 'analysis.json', rows)
    return rows
