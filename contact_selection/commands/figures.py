"""Comparison figures for one suite: where each selector presses, and whether it works.

Writes contact_selection/figures/{contact_map,selector_outcomes,cloud_parity}.png from the
suite's saved labels, selectors and `cloud-parity --execute` report.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/contact-selection-mpl')

import numpy as np

# Reference palette (dataviz skill): categorical slots 1-3 for selectors, status for outcomes.
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
GOOD, BAD, NEUTRAL = '#0ca30c', '#d03b3b', '#a3a29c'
SELECTORS = {  # key: (label, colour, marker)
    'hardware': ('Hardware pipeline (cloud, executed)', INK, '*'),
    'heuristic': ('Hardware rule on grid', '#2a78d6', 's'),
    'logistic15': ('Logistic, 15 features', '#eb6834', 'D'),
    'logistic16': ('Logistic, 16 features', '#1baf7a', '^'),
}
OBJECTS = ['box', 'L', 'heart', 'flashlight', 'soda', 'monitor']


def _style(plt):
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'text.color': INK, 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})


def load(suite: Path, labels_root: Path, extra: list[Path], friction_selector: Path, model16: Path) -> dict:
    from contact_selection.commands.compare import SCOPES
    from contact_selection.selection.heuristic import heuristic_index
    from contact_selection.selection.selector import evaluate, load_robust_contacts
    scopes = {'friction': [labels_root / n for n in SCOPES['friction']]}
    for folder in extra:
        scopes[f'+{folder.name}'] = list(scopes.values())[-1] + sorted(
            p.parent for p in folder.glob('*/rollouts.jsonl'))
    contacts = {scope: load_robust_contacts(dirs)[0] for scope, dirs in scopes.items()}
    models = {'logistic15': json.loads(friction_selector.read_text()),
              'logistic16': json.loads(model16.read_text())}
    parity = json.loads(sorted(suite.glob('cloud_parity_*.json'))[-1].read_text())
    out = {'scopes': list(scopes), 'objects': {}}
    for scope, rows in contacts.items():
        picks = {key: {s['object']: s['selected_index'] for s in evaluate(m, rows)['scenes']}
                 for key, m in models.items()}
        for obj in OBJECTS:
            obj_rows = [r for r in rows if r['object'] == obj]
            entry = out['objects'].setdefault(obj, {'positions': [r['candidate']['position'] for r in obj_rows],
                                                    'labels': {}, 'picks': {}, 'success': {}})
            labels = [r['robust_feasible'] for r in obj_rows]
            entry['labels'][scope] = labels
            pick = {'heuristic': obj_rows[heuristic_index(obj_rows)]['candidate']['index'], **{
                key: picks[key][obj] for key in models}}
            entry['picks'][scope] = pick
            entry['success'][scope] = {k: (None if i is None else labels[i]) for k, i in pick.items()}
            executed = (parity.get('executed') or {}).get('robust', {}).get(obj, {}).get(scope)
            entry['success'][scope]['hardware'] = executed['robust_feasible'] if executed else None
    for obj, p in parity['objects'].items():
        e = out['objects'][obj]
        e['hardware_point'] = p['hardware_press'].get('point')
        e['pivot'], e['scene'] = p['pivot'], p['scene']
        e['normal_error'] = p['top_normal_error_deg']
        rolls = [r for r in (parity.get('executed') or {}).get('rollouts', {}).values()
                 if r['scene'].startswith(obj + '_trial')]
        if rolls and all(r.get('rejected') for r in rolls):  # never executed: no verdict
            for scope in e['success']:
                e['success'][scope]['hardware'] = None
    return out


def contact_map(data: dict, scope: str, path: Path, cloud_seed: int = 0):
    """Top view per object: grid candidates by robust label, each selector's pick, the cloud."""
    import matplotlib.pyplot as plt
    from contact_selection.hardware.sim_cloud import load_scene, payload_body, simulate_cloud
    _style(plt)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    for ax, obj in zip(axes.flat, OBJECTS):
        e = data['objects'][obj]
        scene, model, mjdata = load_scene(Path(e['scene']))
        cloud, _, _ = simulate_cloud(model, mjdata, payload_body(model), scene['geometry']['bounds'], seed=cloud_seed)
        top = cloud[cloud[:, 2] > np.percentile(cloud[:, 2], 30)]
        ax.scatter(top[:, 0] * 100, top[:, 1] * 100, s=0.3, c=GRID, rasterized=True, zorder=0)
        pos = np.array(e['positions']) * 100
        ok = np.array(e['labels'][scope])
        ax.scatter(pos[ok, 0], pos[ok, 1], s=36, c=GOOD, edgecolors=SURFACE, linewidths=1, zorder=2)
        ax.scatter(pos[~ok, 0], pos[~ok, 1], s=36, c=BAD, marker='X', edgecolors=SURFACE, linewidths=0.6, zorder=2)
        true = np.array(e['pivot']['true']) * 100
        ax.axvline(true[0], color=MUTED, lw=1.5, ls='--', zorder=1)
        for k, (label, colour, marker) in list(SELECTORS.items())[1:]:
            i = e['picks'][scope][k]
            if i is not None:
                ax.scatter(*pos[i, :2], s=150, facecolors='none', edgecolors=colour, marker=marker, linewidths=2,
                           zorder=3)
        if e['hardware_point'] is not None:
            hp = np.array(e['hardware_point']) * 100
            ax.scatter(hp[0], hp[1], s=220, c=INK, marker='*', edgecolors=SURFACE, linewidths=1, zorder=4)
        n = int(ok.sum())
        hw = e['success'][scope]['hardware']
        hw_text = {True: 'passes', False: 'fails', None: 'not executable'}[hw]
        ax.set_title(f'{obj}\ngrid robust {n}/{len(ok)} · hardware pick {hw_text}', fontsize=9, loc='left',
                     color=INK)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('x (cm, robot at left)')
        ax.set_ylabel('y (cm)')
        ax.grid(color=GRID, lw=0.5)
    handles = [plt.Line2D([], [], ls='', marker='o', color=GOOD, label='grid candidate: robust'),
               plt.Line2D([], [], ls='', marker='X', color=BAD, label='grid candidate: fails'),
               plt.Line2D([], [], ls='--', color=MUTED, label='pivot edge (sim)')]
    handles += [plt.Line2D([], [], ls='', marker=m, markersize=11 if m == '*' else 8,
                           markerfacecolor=c if m == '*' else 'none', markeredgecolor=c, markeredgewidth=2, label=l)
                for l, c, m in SELECTORS.values()]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle('Where each selector presses (top view) · strictest labels', x=0.01, ha='left',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def selector_outcomes(data: dict, path: Path):
    """Pass / fail / abstain per selector x object, one panel per label scope."""
    import matplotlib.pyplot as plt
    _style(plt)
    scopes = [data['scopes'][0], data['scopes'][-1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    keys = list(SELECTORS)
    for ax, scope in zip(axes, scopes):
        for row, key in enumerate(keys):
            for col, obj in enumerate(OBJECTS):
                result = data['objects'][obj]['success'][scope][key]
                abstain = key != 'hardware' and data['objects'][obj]['picks'][scope].get(key) is None
                colour, text = ((NEUTRAL, 'abstain') if abstain else
                                (NEUTRAL, 'n/a') if result is None else
                                (GOOD, 'pass') if result else (BAD, 'fail'))
                ax.add_patch(plt.Rectangle((col + 0.04, row + 0.06), 0.92, 0.88, color=colour, lw=0))
                ax.text(col + 0.5, row + 0.5, text, ha='center', va='center', color='white', fontsize=9,
                        fontweight='bold')
        ax.set_xlim(0, len(OBJECTS))
        ax.set_ylim(len(keys), 0)
        counts = [f"{obj}\n{sum(data['objects'][obj]['labels'][scope])}/{len(data['objects'][obj]['labels'][scope])}"
                  for obj in OBJECTS]
        ax.set_xticks(np.arange(len(OBJECTS)) + 0.5, counts)
        ax.set_yticks(np.arange(len(keys)) + 0.5, [SELECTORS[k][0] for k in keys])
        ax.tick_params(length=0)
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
        title = 'Friction labels (μ per object as configured)' if scope == 'friction' else \
            'Strictest labels (every object down to μ 0.15, + mass, force)'
        ax.set_title(title, loc='left', fontsize=10)
    fig.text(0.01, 0.01, 'Column header: robust grid candidates / total. "n/a": hardware point rejected by sim '
             'candidate filters (soda pivot-span artifact).', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cloud_parity(data: dict, path: Path):
    """Perception error of the hardware pipeline on simulated clouds."""
    import matplotlib.pyplot as plt
    _style(plt)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
    y = np.arange(len(OBJECTS))
    dx = [data['objects'][o]['pivot'].get('dx_error_m', np.nan) * 1000 for o in OBJECTS]
    a.hlines(y, 0, dx, color='#2a78d6', lw=2)
    a.scatter(dx, y, s=40, c='#2a78d6', zorder=3)
    for yi, v in zip(y, dx):
        a.text(v + (0.4 if v >= 0 else -0.4), yi - 0.3, f'{v:+.1f}', va='center', ha='center', fontsize=8,
               color=INK)
    a.axvline(0, color=MUTED, lw=1)
    a.set_xlim(-4, 12)
    a.set_xlabel('estimated − true pivot x (mm); + = inboard')
    a.set_title('Pivot estimate from cloud', loc='left', fontsize=10)
    med = [data['objects'][o]['normal_error']['median'] for o in OBJECTS]
    p90 = [data['objects'][o]['normal_error']['p90'] for o in OBJECTS]
    b.hlines(y, med, p90, color=GRID, lw=4)
    b.scatter(med, y, s=40, c='#eb6834', zorder=3, label='median')
    b.scatter(p90, y, s=40, facecolors='none', edgecolors='#eb6834', lw=1.5, zorder=3, label='90th pct')
    b.set_xscale('log')
    b.set_xlabel('top-surface normal error (deg, log); 180 = flipped')
    b.set_title('Normal estimate from cloud', loc='left', fontsize=10)
    b.legend(frameon=False, fontsize=8, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.32))
    a.set_yticks(y, OBJECTS)
    a.set_ylim(len(OBJECTS) - 0.5, -0.5)
    for ax in (a, b):
        ax.grid(axis='x', color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('suite', type=Path)
    parser.add_argument('--labels', type=Path, help='Folder with the friction datasets (default: SUITE)')
    parser.add_argument('--extra', type=Path, nargs='+', default=[], help='Extra scenario folders, as in compare')
    parser.add_argument('--friction-selector', type=Path, required=True,
                        help='15-feature model.json trained on the friction suite')
    parser.add_argument('--model16', type=Path, help='16-feature model.json (default: SUITE/geometry_selector_ray)')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'figures')
    args = parser.parse_args()
    data = load(args.suite, args.labels or args.suite, args.extra, args.friction_selector,
                args.model16 or args.suite / 'geometry_selector_ray' / 'model.json')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    contact_map(data, data['scopes'][-1], out / 'contact_map.png')
    selector_outcomes(data, out / 'selector_outcomes.png')
    cloud_parity(data, out / 'cloud_parity.png')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
