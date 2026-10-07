"""Draw each object's top view on a centimetre grid, for writing human contacts.

Frame (cm, fixed per object): origin at the default robot-side tipping edge (the
press-pull pivot), x away from the robot, y to the robot's left. The dashed outline
is the support footprint, from which a tipping edge can be chosen. Entries already
in config/human_contacts.json are drawn (contact, chosen edge, pull arrow) so the
file can be checked by eye.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
OBJECTS = ['box', 'L', 'heart', 'flashlight', 'soda', 'monitor']
DEFAULT_SUITE = PACKAGE.parent / 'outputs/contact_selection/suites/2026-10-05_mass_force'


def footprint_cm(geometry: dict, pivot: np.ndarray) -> np.ndarray:
    """Closed support polygon (convex hull of collision vertices at table height)."""
    from scipy.spatial import ConvexHull
    vertices = np.concatenate([np.asarray(h) for h in geometry['hulls']])
    base = vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.005, :2]
    ring = base[ConvexHull(base).vertices]
    return (np.vstack([ring, ring[:1]]) - pivot[:2]) * 100


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from contact_selection.hardware.sim_cloud import load_scene, payload_body, simulate_cloud
    from contact_selection.selection.human import load, resolve
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--suite', type=Path, default=DEFAULT_SUITE, help='Suite with saved *_mu_0p50 scenes')
    parser.add_argument('--output', type=Path, default=PACKAGE / 'figures' / 'annotation_guide.png')
    args = parser.parse_args()
    entries = load()
    ink, grid, surface, human, com_c = '#0b0b0b', '#e4e3df', '#fcfcfb', '#eb6834', '#2a78d6'
    plt.rcParams.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'font.size': 9})
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, obj in zip(axes.flat, OBJECTS):
        scene_path = sorted(args.suite.glob(f'*_mu_0p50/{obj}_trial_01/scene.json'))[0]
        scene, model, data = load_scene(scene_path)
        geometry = scene['geometry']
        pivot = np.asarray(geometry['pivot'])
        cloud, normals, _ = simulate_cloud(model, data, payload_body(model), geometry['bounds'], noise=0.0)
        top = cloud[normals[:, 2] > 0.5]
        xy = (top[:, :2] - pivot[:2]) * 100
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=top[:, 2] * 100, s=1, cmap='Greys', rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.7, label='surface height (cm)')
        ring = footprint_cm(geometry, pivot)
        ax.plot(ring[:, 0], ring[:, 1], color=ink, lw=1, ls='--', label='support footprint')
        com = tuple((np.asarray(geometry['com_world_m'][:2]) - pivot[:2]) * 100)
        ax.scatter(*com, marker='+', s=160, c=com_c, linewidths=2, zorder=3)
        ax.plot([0, 0], [ring[:, 1].min(), ring[:, 1].max()], color=ink, lw=2.5)
        status = 'no pick yet'
        try:
            pick = resolve(entries.get(obj, {}), com)
        except ValueError as exc:
            pick, status = None, f'ERROR: {exc}'
        if pick is not None:
            cx, cy = pick.contact_cm
            if pick.tipping_edge_cm is not None:
                (x1, y1), (x2, y2) = pick.tipping_edge_cm
                ax.plot([x1, x2], [y1, y2], color=human, lw=4, solid_capstyle='round', zorder=4)
                dist = np.min(np.linalg.norm(ring[:, None, :] - np.array(pick.tipping_edge_cm)[None], axis=2), axis=0)
                if dist.max() > 1.0:
                    status = f'WARNING: edge end {dist.max():.1f} cm off the footprint'
            if pick.pull_through_com:
                ax.plot([com[0], cx], [com[1], cy], color=human, lw=1, ls=':', zorder=3)
            span = max(np.ptp(ring[:, 0]), np.ptp(ring[:, 1]))
            dx, dy = np.array(pick.pull_direction) * 0.25 * span
            from matplotlib.patches import FancyArrowPatch
            ax.add_patch(FancyArrowPatch((cx, cy), (cx + dx, cy + dy), arrowstyle='-|>', color=human, lw=2.5,
                                         mutation_scale=18, zorder=5))
            ax.scatter(cx, cy, marker='*', s=320, c=human, edgecolors=ink, zorder=6)
            if not status.startswith(('ERROR', 'WARNING')):
                edge = 'custom edge' if pick.tipping_edge_cm else 'default edge'
                pull = 'through CoM' if pick.pull_through_com else f'pull ({pick.pull_direction[0]:+.2f}, {pick.pull_direction[1]:+.2f})'
                status = f'({cx:g}, {cy:g}) · {edge} · {pull}'
        lo = ring.min(axis=0) - 1
        hi = ring.max(axis=0) + 1
        ax.set_xticks(np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 1), minor=True)
        ax.set_yticks(np.arange(np.floor(lo[1]), np.ceil(hi[1]) + 1), minor=True)
        ax.grid(which='both', color=grid, lw=0.6)
        ax.grid(which='major', color='#c3c2b7', lw=0.8)
        ax.set_aspect('equal')
        ax.set_xlabel('x_cm  (away from robot →)')
        ax.set_ylabel("y_cm  (robot's left ↑)")
        ax.set_title(f'{obj}  ·  {status}', loc='left', color=ink, fontsize=9.5)
    fig.suptitle('Human contact picks · 1 cm grid · black bar = default tipping edge · dashed = support footprint · '
                 'blue + = 2D CoM · orange = your contact ★, edge, pull →', x=0.01, ha='left', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
