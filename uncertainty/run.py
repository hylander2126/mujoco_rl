"""Run the full uncertainty analysis and write CSV/JSON tables and plots.

Sections (each a JSON file plus a CSV where tabular):
    bias_breakdown    where the noise-free estimate's bias comes from
    mc_all            all sources at their default sigma
    sensitivity       one source at a time, plus all and none
    sweeps            sigma multiplier x source
    nls_vs_mc         local Gauss-Newton covariance vs Monte Carlo (white F/T noise)
    sweep_geometry    shorter tilt sweeps: worse conditioning, stronger correlation
    confidence        all-source spread vs sweep reached, sim box + the 4 hardware geometries
    contacts          per-candidate robustness to reconstructed-geometry error
    sanity_checks     pass/fail summary of the expected qualitative behaviour
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from util.paths import OUTPUT_ROOT, dated
from uncertainty import contact_robustness as cr
from uncertainty import estimator_mc as mc

UNCERTAINTY_OUTPUTS = OUTPUT_ROOT / 'uncertainty'
NOMINAL_ROLLOUT = UNCERTAINTY_OUTPUTS / 'nominal_box'
SWEEP_LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)
SWEEP_SOURCES = ('ft_white', 'ft_bias', 'tilt_bias', 'pivot', 'com_x')
SWEEP_RANGES_DEG = (4.0, 8.0, None)  # None -> full sweep
CONTACT_CONFIGS = ('box_mu_0p20.json', 'l_mu_0p25.json')
CONFIDENCE_DEG = (3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
# Ball centre above / ahead of the pivot and the max tilt reached, from hardware_check.py
# (40 trials). Mass, com_x, com_z are the press_pull_estimator ground truth.
HARDWARE_GEOMETRY = {'box': (0.317, 0.000, 16.7), 'heart': (0.216, 0.001, 18.1),
                     'flashlight': (0.217, 0.025, 13.3), 'monitor': (0.510, 0.049, 13.0)}
LABELS = {'mass': 'm (kg)', 'com_z': 'z_c (m)', 'mu': 'mu'}

# Reference categorical palette (dataviz skill), fixed order; text stays in ink colors.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def ensure_rollout(path: Path) -> Path:
    """Simulate the validated box press-pull once if no nominal rollout exists (~75 s)."""
    if (path / 'trajectory.npz').exists():
        return path
    from parameter_estimation.press_pull_demo import BoxDemoConfig, run_demo
    print(f'No nominal rollout at {path}; simulating the box demo preset once...', flush=True)
    result = run_demo(BoxDemoConfig(), path, video=False, verbose=False)
    if not result['feasible']:
        raise RuntimeError(f'Nominal rollout did not tip cleanly: {result["failure_modes"]}')
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def stat_rows(condition: str, stats: dict, truth: dict, nominal: dict, **extra) -> list[dict]:
    return [{'condition': condition, **extra, 'param': p, 'truth': truth[p], 'noise_free_estimate': nominal[p],
             'mean': stats['mean'][i], 'std': stats['std'][i], 'rel_std': stats['rel_std'][i],
             'bias_vs_truth': stats['bias_vs_truth'][i], 'bias_vs_noise_free': stats['mean'][i] - nominal[p],
             'ci95_lo': stats['ci95'][0][i], 'ci95_hi': stats['ci95'][1][i], 'n': stats['n'], 'failed': stats['failed']}
            for i, p in enumerate(mc.PARAMS)]


def run(args) -> dict:
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    nom = mc.load_rollout(ensure_rollout(args.rollout), args.decimate)
    base = mc.NOISE_PRESETS[args.noise]
    zero = base.scaled(0)
    nominal = {k: v for k, v in mc.estimate(nom, nom.trial, nom.pivot, nom.com_x).items() if k != 'result'}
    print(f'truth {nom.truth}\nnoise-free estimate {nominal}', flush=True)
    meta = {'rollout': str(args.rollout), 'decimate': args.decimate, 'truth': nom.truth,
            'noise_free_estimate': nominal, 'noise_preset': args.noise, 'default_noise': base.__dict__, 'sources': mc.SOURCES}
    write_json(out / 'meta.json', meta)
    write_json(out / 'bias_breakdown.json', mc.bias_breakdown(args.rollout, args.decimate))

    # 2-3. Monte Carlo (all sources) and one-at-a-time sensitivity.
    conditions = {'none': zero, **{s: base.only(*f) for s, f in mc.SOURCES.items()}, 'all': base}
    sens, rows = {}, []
    for i, (name, spec) in enumerate(conditions.items()):
        print(f'sensitivity: {name}', flush=True)
        r = mc.monte_carlo(nom, spec, args.samples if name != 'none' else 5, seed=args.seed + i)
        sens[name] = r['stats']
        rows += stat_rows(name, r['stats'], nom.truth, nominal)
        if name == 'all':
            write_csv(out / 'mc_all_samples.csv', r['samples'])
            write_json(out / 'mc_all.json', r['stats'])
    write_json(out / 'sensitivity.json', sens)
    write_csv(out / 'sensitivity.csv', rows)

    # 4. Noise-magnitude sweeps.
    sweeps, rows = {}, []
    for s in SWEEP_SOURCES:
        for k in SWEEP_LEVELS:
            print(f'sweep: {s} x{k}', flush=True)
            r = mc.monte_carlo(nom, base.only(*mc.SOURCES[s]).scaled(k), args.sweep_samples if k else 3,
                               seed=args.seed + 1000 + int(10 * k))
            sweeps.setdefault(s, []).append({'multiplier': k, **r['stats']})
            sigma = {f: k * getattr(base, f) for f in mc.SOURCES[s]}
            rows += stat_rows(f'{s}_x{k:g}', r['stats'], nom.truth, nominal, source=s, multiplier=k,
                              input_sigma=json.dumps(sigma))
    write_json(out / 'sweeps.json', sweeps)
    write_csv(out / 'sweeps.csv', rows)

    # 5. NLS covariance vs Monte Carlo, for white F/T (where NLS assumptions hold) and all sources.
    nls = {}
    for name, spec in (('ft_white', base.only(*mc.SOURCES['ft_white'])), ('all', base)):
        print(f'nls: {name}', flush=True)
        r = mc.monte_carlo(nom, spec, args.samples, seed=args.seed + 2000, nls=args.nls_samples)
        nls[name] = {'mc': r['stats'], 'nls': r['nls'],
                     'std_ratio_nls_over_mc': (np.array(r['nls']['std']) / np.array(r['stats']['std'])).tolist()}
    write_json(out / 'nls_vs_mc.json', nls)

    # 8c. Poor estimation geometry: shorter sweeps.
    geo = []
    for th in SWEEP_RANGES_DEG:
        n2 = mc.truncate_sweep(nom, th) if th is not None else nom
        r = mc.monte_carlo(n2, base.only(*mc.SOURCES['ft_white']), args.sweep_samples, seed=args.seed + 3000,
                           nls=min(args.nls_samples, 10))
        geo.append({'max_tilt_deg': th or float(np.max(-np.rad2deg(mc.we.object_tilt(nom.trial, nom.pivot)[0][:, 1]))),
                    'std': r['stats']['std'], 'corr_m_zc': r['stats']['corr'][0][1],
                    'cond_arc': r['nls']['cond_arc'], 'cond_unarc': r['nls']['cond_unarc']})
        print(f'sweep range {geo[-1]["max_tilt_deg"]:.1f} deg: std {geo[-1]["std"]}', flush=True)
    write_json(out / 'sweep_geometry.json', geo)
    write_csv(out / 'sweep_geometry.csv', [{'max_tilt_deg': g['max_tilt_deg'], 'std_mass': g['std'][0],
                                            'std_com_z': g['std'][1], 'std_mu': g['std'][2],
                                            'corr_m_zc': g['corr_m_zc'], 'cond_arc': g['cond_arc'],
                                            'cond_unarc': g['cond_unarc']} for g in geo])

    # Confidence vs how far the sweep got, all sources. Exact synthetic trials for the
    # hardware geometries isolate the lever-arm/height effect from the sim box's biases.
    from press_pull_estimator.objects import OBJECTS as HW
    objects = {'box (sim rollout)': (nom, None)}
    for name, (height, x0, max_deg) in HARDWARE_GEOMETRY.items():
        objects[f'{name} (synthetic)'] = (mc.synthetic_nominal(HW[name]['mass'], HW[name]['com_x'], HW[name]['com_z'],
                                                               max_tilt_deg=max_deg, n=3000, ball_height=height,
                                                               ball_x=x0), max_deg)
    confidence = []
    for name, (n0, max_deg) in objects.items():
        full = max_deg or float(np.max(-np.rad2deg(mc.we.object_tilt(n0.trial, n0.pivot)[0][:, 1])))
        for th in [t for t in CONFIDENCE_DEG if t < full - 0.5] + [full]:
            n2 = mc.truncate_sweep(n0, th) if th < full else n0
            r = mc.monte_carlo(n2, base, args.confidence_samples, seed=args.seed + 4000, nls=5)
            confidence.append({'object': name, 'max_tilt_deg': th, 'theta_star_deg': float(np.degrees(
                np.arctan2(n0.com_x, n0.truth['com_z']))), 'rel_std': r['stats']['rel_std'],
                'rel_bias': (np.array(r['stats']['bias_vs_truth']) / [n0.truth[p] for p in mc.PARAMS]).tolist(), 'corr_m_zc': r['stats']['corr'][0][1],
                'nls_rel_std': (np.array(r['nls']['std']) / [n0.truth[p] for p in mc.PARAMS]).tolist()})
            print(f'confidence {name} {th:.1f} deg: rel std {np.round(confidence[-1]["rel_std"], 3)}', flush=True)
    write_json(out / 'confidence.json', confidence)
    write_csv(out / 'confidence.csv', [{'object': c['object'], 'max_tilt_deg': c['max_tilt_deg'],
                                        'theta_star_deg': c['theta_star_deg'],
                                        **{f'rel_std_{p}': c['rel_std'][i] for i, p in enumerate(mc.PARAMS)},
                                        **{f'rel_bias_{p}': c['rel_bias'][i] for i, p in enumerate(mc.PARAMS)},
                                        **{f'nls_rel_std_{p}': c['nls_rel_std'][i] for i, p in enumerate(mc.PARAMS)},
                                        'corr_m_zc': c['corr_m_zc']} for c in confidence])

    # 7. Contact-selection robustness.
    selector = json.loads(args.selector_model.read_text()) if args.selector_model else None
    contacts, rows = [], []
    for cfg in args.contact_configs:
        scene = cr.build_scene(Path(cfg) if Path(cfg).exists() else
                               Path(__file__).resolve().parents[1] / 'contact_selection/config' / cfg)
        print(f'contacts: {scene.name}', flush=True)
        r = cr.robustness(scene, cr.GeometryNoise(), args.contact_samples, args.seed, selector, args.threshold)
        r['zero_noise'] = cr.robustness(scene, cr.GeometryNoise().scaled(0), 3, args.seed, selector, args.threshold)['candidates']
        r['noise_sweep'] = []
        for k in SWEEP_LEVELS[1:]:
            rk = cr.robustness(scene, cr.GeometryNoise().scaled(k), args.contact_samples // 2, args.seed, selector, args.threshold)
            r['noise_sweep'].append({'multiplier': k, 'nominal_pick_p_feasible': rk['nominal_pick_p_feasible'],
                                     'robust_pick_p_feasible': rk['robust_pick_p_feasible'],
                                     'mean_p_feasible_nominally_feasible': float(np.mean(
                                         [c['p_feasible'] for c in rk['candidates'] if c['nominal_feasible']] or [np.nan]))})
        contacts.append(r)
        rows += [{'scene': scene.name, **c, 'is_nominal_pick': c['index'] == r['nominal_pick'],
                  'is_robust_pick': c['index'] == r['robust_pick']} for c in r['candidates']]
    write_json(out / 'contacts.json', contacts)
    write_csv(out / 'contacts.csv', rows)

    checks = sanity_checks(sens, sweeps, geo, contacts)
    write_json(out / 'sanity_checks.json', checks)
    results = {'meta': meta, 'sensitivity': sens, 'sweeps': sweeps, 'nls': nls, 'geometry': geo,
               'confidence': confidence, 'contacts': contacts, 'checks': checks}
    plot_all(out, results)
    return results


def sanity_checks(sens, sweeps, geo, contacts) -> dict:
    from scipy.stats import spearmanr
    checks = {}
    checks['zero_noise_zero_variance'] = {'max_std': max(sens['none']['std']),
                                          'pass': max(sens['none']['std']) < 1e-9}
    mono = {}
    for s, levels in sweeps.items():
        std = np.array([lv['std'] for lv in levels])  # levels x params
        # Allow 10% MC jitter between neighbouring levels.
        mono[s] = bool(np.all(std[1:] >= 0.9 * std[:-1] - 1e-12))
    checks['more_noise_more_uncertainty'] = {'by_source': mono, 'pass': all(mono.values())}
    short, full = geo[0], geo[-1]
    checks['poor_geometry_more_uncertain'] = {
        'short_sweep_deg': short['max_tilt_deg'], 'full_sweep_deg': full['max_tilt_deg'],
        'std_ratio_short_over_full': (np.array(short['std']) / np.array(full['std'])).tolist(),
        'abs_corr_short': abs(short['corr_m_zc']), 'abs_corr_full': abs(full['corr_m_zc']),
        'pass': bool(short['std'][1] > full['std'][1] and abs(short['corr_m_zc']) > abs(full['corr_m_zc']))}
    per_scene = {}
    for c in contacts:
        feas = [x for x in c['candidates'] if x['nominal_feasible']]
        rho = spearmanr([x['boundary_slack_mm'] for x in feas], [x['p_feasible'] for x in feas]).statistic \
            if len(feas) > 2 else np.nan
        zero_ok = all(z['p_feasible'] == float(z['nominal_feasible']) for z in c['zero_noise'])
        per_scene[c['scene']] = {'spearman_slack_vs_p_feasible': float(rho), 'zero_noise_deterministic': zero_ok}
    checks['near_boundary_less_robust'] = {
        'by_scene': per_scene,
        'pass': all(v['spearman_slack_vs_p_feasible'] > 0 and v['zero_noise_deterministic'] for v in per_scene.values())}
    return checks


# ------------------------------------------------------------------------------------
#  Plots
# ------------------------------------------------------------------------------------
def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_all(out: Path, res: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK, 'axes.titlesize': 10})
    truth, nominal = res['meta']['truth'], res['meta']['noise_free_estimate']

    # 1. MC distributions (all sources).
    import csv as _csv
    with (out / 'mc_all_samples.csv').open() as f:
        x = np.array([[float(r[p]) for p in mc.PARAMS] for r in _csv.DictReader(f)])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for i, p in enumerate(mc.PARAMS):
        ax = axes[i]
        ax.hist(x[:, i], bins=30, color=SERIES[0], edgecolor='white', lw=0.5)
        ax.axvline(truth[p], color=INK, lw=1.5, label='truth')
        ax.axvline(nominal[p], color=MUTED, lw=1.5, ls='--', label='noise-free fit')
        lo, hi = np.percentile(x[:, i], [2.5, 97.5])
        ax.axvspan(lo, hi, color=SERIES[0], alpha=0.08, label='95% interval')
        ax.set_xlabel(LABELS[p]); _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    axes[3].scatter(x[:, 0], x[:, 1], s=8, color=SERIES[0], alpha=0.5, edgecolors='none')
    axes[3].plot(truth['mass'], truth['com_z'], 'o', color=INK, ms=6)
    axes[3].set_xlabel(LABELS['mass']); axes[3].set_ylabel(LABELS['com_z']); _style(axes[3])
    axes[3].set_title(f"corr(m, z_c) = {res['sensitivity']['all']['corr'][0][1]:.2f}", fontsize=9)
    fig.suptitle(f"Monte Carlo, all sources ({res['meta']['noise_preset']} noise)", x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(out / 'mc_distributions.png', dpi=150); plt.close(fig)

    # 2. One-at-a-time sensitivity (relative std).
    names = [n for n in res['sensitivity'] if n != 'none']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for i, p in enumerate(mc.PARAMS):
        v = [100 * res['sensitivity'][n]['rel_std'][i] for n in names]
        axes[i].barh(names, v, color=[INK if n == 'all' else SERIES[0] for n in names], height=0.6)
        for j, val in enumerate(v):
            axes[i].text(val, j, f' {val:.2g}%', va='center', fontsize=7, color=MUTED)
        axes[i].set_xlabel(f'std / truth of {LABELS[p]} (%)'); _style(axes[i])
    axes[0].invert_yaxis()
    fig.suptitle('One-at-a-time sensitivity (each source alone at its default sigma)', x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(out / 'sensitivity.png', dpi=150); plt.close(fig)

    # 3. Noise sweeps.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for i, p in enumerate(mc.PARAMS):
        for j, (s, levels) in enumerate(res['sweeps'].items()):
            k = [lv['multiplier'] for lv in levels]
            y = [100 * lv['rel_std'][i] for lv in levels]
            axes[i].plot(k, y, '-o', color=SERIES[j], lw=2, ms=4, label=s)
            axes[i].annotate(s, (k[-1], y[-1]), xytext=(3, 0), textcoords='offset points', fontsize=7,
                             color=MUTED, va='center')
        axes[i].set_xlabel('input sigma / default sigma'); axes[i].set_ylabel(f'std / truth of {LABELS[p]} (%)')
        _style(axes[i])
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Input uncertainty vs parameter uncertainty', x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(out / 'sweeps.png', dpi=150); plt.close(fig)

    # 4. Covariance / correlation and conditioning.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    corr = np.array(res['sensitivity']['all']['corr'])
    im = axes[0].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    for (a, b), v in np.ndenumerate(corr):
        axes[0].text(b, a, f'{v:.2f}', ha='center', va='center', fontsize=8, color=INK if abs(v) < 0.6 else 'white')
    axes[0].set_xticks(range(3), mc.PARAMS); axes[0].set_yticks(range(3), mc.PARAMS)
    axes[0].set_title('MC correlation, all sources'); fig.colorbar(im, ax=axes[0], shrink=0.8)
    w = res['nls']['ft_white']
    pos = np.arange(3)
    axes[1].bar(pos - 0.18, np.array(w['mc']['std']) / np.array([truth[p] for p in mc.PARAMS]) * 100, 0.34,
                color=SERIES[0], label='Monte Carlo')
    axes[1].bar(pos + 0.18, np.array(w['nls']['std']) / np.array([truth[p] for p in mc.PARAMS]) * 100, 0.34,
                color=SERIES[1], label='NLS (J^T J)^-1')
    a = res['nls']['all']
    axes[1].scatter(pos, np.array(a['mc']['std']) / np.array([truth[p] for p in mc.PARAMS]) * 100, marker='_',
                    s=300, color=INK, label='MC, all sources')
    axes[1].set_yscale('log'); axes[1].set_xticks(pos, mc.PARAMS); axes[1].set_ylabel('std / truth (%)')
    axes[1].set_title('White F/T noise: NLS vs MC'); axes[1].legend(fontsize=7, frameon=False); _style(axes[1])
    g = res['geometry']
    th = [x['max_tilt_deg'] for x in g]
    axes[2].plot(th, [100 * x['std'][1] / truth['com_z'] for x in g], '-o', color=SERIES[0], lw=2, label='std z_c (%)')
    axes[2].plot(th, [100 * x['std'][0] / truth['mass'] for x in g], '-o', color=SERIES[2], lw=2, label='std m (%)')
    for x in g:  # |corr| and conditioning as labels, not a second y-axis
        axes[2].annotate(f"|corr|={abs(x['corr_m_zc']):.2f}\ncond={x['cond_arc']:.0f}",
                         (x['max_tilt_deg'], 100 * x['std'][1] / truth['com_z']), xytext=(4, 4),
                         textcoords='offset points', fontsize=7, color=MUTED)
    axes[2].set_ylim(0, 1.3 * axes[2].get_ylim()[1])
    axes[2].set_xlabel('max tilt used in fit (deg)'); axes[2].set_title('Shorter sweep, white F/T noise')
    axes[2].legend(fontsize=7, frameon=False); _style(axes[2])
    fig.tight_layout(); fig.savefig(out / 'covariance.png', dpi=150); plt.close(fig)

    # 5. Contact robustness, one panel per scene (top view).
    cs = res['contacts']
    fig, axes = plt.subplots(1, len(cs), figsize=(5.2 * len(cs), 4.2), squeeze=False)
    for ax, c in zip(axes[0], cs):
        xs = np.array([[k['x_m'], k['y_m']] for k in c['candidates']])
        pf = np.array([k['p_feasible'] for k in c['candidates']])
        fq = np.array([k['selection_freq'] for k in c['candidates']])
        sc = ax.scatter(xs[:, 0], xs[:, 1], c=pf, cmap='Blues', vmin=0, vmax=1, s=60 + 600 * fq,
                        edgecolors=MUTED, linewidths=0.6)
        for k, (px, py) in zip(c['candidates'], xs):
            ax.annotate(str(k['index']), (px, py), xytext=(5, 4), textcoords='offset points', fontsize=7, color=MUTED)
        for idx, label, mk in ((c['nominal_pick'], 'nominal pick', 's'), (c['robust_pick'], 'robust pick', 'D')):
            if idx is not None:
                ax.scatter(*xs[idx], marker=mk, s=220, facecolors='none', edgecolors=INK, linewidths=1.5, label=label)
        ax.set_aspect('equal'); ax.set_xlabel('world x (m)'); ax.set_ylabel('world y (m)'); _style(ax)
        npf = c['nominal_pick_p_feasible']
        ax.set_title(f"{c['scene']}  (mu={c['mu_table']:g})\nnominal pick P(feasible)="
                     f"{npf if npf is None else round(npf, 2)}, robust pick {c['robust_pick_p_feasible']:.2f}", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2, markerscale=0.6)
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, pad=0.02,
                 label='P(feasible) under geometry noise')
    fig.suptitle('Contact robustness: color = P(feasible), size = selection frequency', x=0.01, ha='left')
    fig.savefig(out / 'contact_robustness.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # 6. Confidence vs sweep extent, all sources (log scale; dashed = NLS alone for the sim box).
    names = list(dict.fromkeys(c['object'] for c in res['confidence']))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for i, p in enumerate(mc.PARAMS):
        ax = axes[i]
        for j, name in enumerate(names):
            rows = [c for c in res['confidence'] if c['object'] == name]
            ax.plot([c['max_tilt_deg'] for c in rows], [100 * c['rel_std'][i] for c in rows], '-o', ms=4, lw=2,
                    color=SERIES[j], label=name)
            if j == 0:
                ax.plot([c['max_tilt_deg'] for c in rows], [100 * c['nls_rel_std'][i] for c in rows], '--',
                        lw=1.5, color=SERIES[j], label='box: NLS covariance alone')
        ax.set_yscale('log'); ax.set_xlabel('max tilt reached (deg)'); ax.set_title(LABELS[p]); _style(ax)
    axes[0].set_ylabel('std / truth (%)')
    axes[0].legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(0, -0.2), ncol=3)
    fig.suptitle(f"Spread vs sweep reached, all sources ({res['meta']['noise_preset']} noise)", x=0.01, ha='left')
    fig.savefig(out / 'confidence.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rollout', type=Path, default=NOMINAL_ROLLOUT,
                        help='run_press_pull_demo.py output; simulated once if missing')
    parser.add_argument('--output', type=Path, default=None, help='New directory (default: outputs/uncertainty/DATE_analysis)')
    parser.add_argument('--samples', type=int, default=200, help='MC draws per sensitivity condition')
    parser.add_argument('--sweep-samples', type=int, default=100, help='MC draws per sweep level')
    parser.add_argument('--nls-samples', type=int, default=20, help='Draws on which NLS covariance is computed')
    parser.add_argument('--contact-samples', type=int, default=500)
    parser.add_argument('--contact-configs', nargs='+', default=list(CONTACT_CONFIGS))
    parser.add_argument('--selector-model', type=Path, help='Trained selector model.json; default uses the proxy score')
    parser.add_argument('--threshold', type=float, default=0.5, help='Selector threshold (only with --selector-model)')
    parser.add_argument('--decimate', type=int, default=2, help='1 kHz sim log -> 500 Hz estimator grid')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', choices=list(mc.NOISE_PRESETS), default='assumed',
                        help="'measured' uses the F/T and pivot levels from hardware_check.py")
    parser.add_argument('--confidence-samples', type=int, default=60, help='MC draws per object x sweep extent')
    args = parser.parse_args(argv)
    args.output = args.output or UNCERTAINTY_OUTPUTS / dated('analysis')
    res = run(args)
    print(json.dumps({k: v['pass'] for k, v in res['checks'].items()}, indent=1))
    print(f'Saved to {args.output}')
    return 0
