"""Compare selector ablations on one suite: feature set x robustness label scope.

Every model is scored against every label scope. A model trained on friction-only
labels can look better simply because its target is easier, so the comparison
that matters is how each model's picks hold up under the strictest labels.
Held-out means the fixed validation/test objects; leave-one-object-out (LOO)
refits once per object on all others, ignoring the split, for more held-out cases,
and scores the held-out object against the strictest scope.
"""
import argparse
from collections import defaultdict
from datetime import date
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

from contact_selection.sim.dataset import write_json
from contact_selection.selection.features import FEATURE_NAMES
from contact_selection.selection.heuristic import HARDWARE_WEIGHT, heuristic_report
from contact_selection.selection.selector import evaluate, fit_logistic, load_robust_contacts

FRICTION = ['box_mu_0p50', 'box_mu_0p20', 'box_mu_0p15', 'heart_l_mu_0p50',
            'l_mu_0p25', 'flashlight_mu_0p50', 'monitor_soda_mu_0p50']
MASS = [f'{base}_{variant}' for variant in ('mass_x0p5', 'mass_x2p0')
        for base in ('box', 'heart_l', 'flashlight', 'monitor_soda')]
FORCE = [f'{base}_force_x2p6' for base in ('box', 'heart_l', 'flashlight', 'monitor_soda')]
SCOPES = {'friction': FRICTION, 'friction+mass': FRICTION + MASS,
          'friction+mass+force': FRICTION + MASS + FORCE}
FEATURES = {'legacy15': [name for name in FEATURE_NAMES if name != 'pivot_ray_angle_rad'],
            'ray16': FEATURE_NAMES}
# press_pivot_weight values for the hardware heuristic; HARDWARE_WEIGHT is the deployed default.
HEURISTIC_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def summarize(report: dict, objects: set[str] | None = None) -> dict:
    scenes = [s for s in report['scenes'] if objects is None or s['object'] in objects]
    selected = [s for s in scenes if s['selected_index'] is not None]
    return {'objects': len(scenes),
            'selected_success': sum(s['selected_success'] for s in selected),
            'selected_failure': sum(not s['selected_success'] for s in selected),
            'abstained': len(scenes) - len(selected),
            'false_abstain': sum(s['selected_index'] is None and s['oracle_success'] for s in scenes),
            'mean_brier': (float(np.mean([s['brier_score'] for s in scenes]))
                           if scenes and scenes[0]['brier_score'] is not None else None)}


def heuristic_loo(strict: list[dict], weights: list[float]) -> dict:
    """Tune press_pivot_weight on all other objects, then apply it to the held one.

    Tuning maximizes successes minus failures under the strictest labels; ties go
    to the weight closest to the hardware default.
    """
    out = {}
    objects = sorted({row['object'] for row in strict})
    outcome = {w: {s['object']: s['selected_success'] for s in heuristic_report(strict, w)['scenes']}
               for w in weights}
    for held in objects:
        def gain(w):
            return sum(1 if outcome[w][o] else -1 for o in objects if o != held and outcome[w][o] is not None)
        best = max(weights, key=lambda w: (gain(w), -abs(np.log2(w / HARDWARE_WEIGHT)) if w > 0 else -99))
        out[held] = {'weight': best, 'selected_success': outcome[best][held]}
    return out


def relabel(contacts: list[dict], target: list[dict]) -> list[dict]:
    """Score one scope's model against another scope's labels (same candidates)."""
    labels = {(row['object'], row['candidate']['index']): row['robust_feasible'] for row in target}
    return [{**row, 'robust_feasible': labels[(row['object'], row['candidate']['index'])]}
            for row in contacts]


def leave_one_object_out(contacts: list[dict], names: list[str], strict: list[dict],
                         threshold: float) -> dict:
    out = {}
    for held in sorted({row['object'] for row in contacts}):
        train = [{**row, 'split': 'train' if row['object'] != held else 'held'} for row in contacts]
        try:
            model = fit_logistic(train, feature_names=names)
        except ValueError as exc:
            out[held] = {'skipped': str(exc)}
            continue
        held_rows = [row for row in relabel(contacts, strict) if row['object'] == held]
        scene = evaluate(model, held_rows, threshold)['scenes'][0]
        out[held] = {key: scene[key] for key in ('candidates', 'robust_feasible', 'selected_index',
                                                 'selected_success', 'abstain_reason', 'brier_score')}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path, help='Suite from `contact_selection rerun`')
    parser.add_argument('--baseline', type=Path, nargs='*', default=[],
                        help='Saved model.json files to score on this suite (e.g. older selectors)')
    parser.add_argument('--output', type=Path,
                        help='Fresh JSON report path (default: SUITE/selector_comparison_YYYY-MM-DD.json)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--extra', type=Path, nargs='+', default=[],
                        help='Folders of extra datasets (e.g. low friction x mass); each adds a '
                             'cumulative, stricter scope named after the folder')
    args = parser.parse_args()
    args.output = args.output or args.suite / f'selector_comparison_{date.today():%Y-%m-%d}.json'
    if args.output.exists():
        raise FileExistsError(args.output)
    scopes = {scope: [args.suite / name for name in dirs] for scope, dirs in SCOPES.items()}
    for folder in args.extra:
        previous = list(scopes.values())[-1]
        scopes[f'+{folder.name}'] = previous + sorted(p.parent for p in folder.glob('*/rollouts.jsonl'))
    contacts = {scope: load_robust_contacts(dirs)[0] for scope, dirs in scopes.items()}
    strict = contacts[list(scopes)[-1]]
    held_out = {row['object'] for row in strict if row['split'] != 'train'}
    labels = {scope: {obj: f"{sum(r['robust_feasible'] for r in rows if r['object'] == obj)}/"
                           f"{sum(r['object'] == obj for r in rows)}"
                      for obj in sorted({r['object'] for r in rows})}
              for scope, rows in contacts.items()}
    models = {}
    for scope, rows in contacts.items():
        for feature_set, names in FEATURES.items():
            models[f'{feature_set}|{scope}'] = (fit_logistic(rows, feature_names=names), rows)
    for weight in HEURISTIC_WEIGHTS:
        models[f'heuristic|w={weight:g}'] = (None, weight)
    for path in args.baseline:
        models[f'saved:{path.parent.name}'] = (json.loads(path.read_text()), None)
    results = {}
    for key, (model, own) in models.items():
        heuristic = model is None
        entry = ({'press_pivot_weight': own} if heuristic else
                 {'feature_names': model['feature_names'], 'weights': dict(zip(model['feature_names'], model['weights']))})
        for scope, rows in contacts.items():
            report = heuristic_report(rows, own) if heuristic else evaluate(model, rows, args.threshold)
            entry[f'eval:{scope}'] = {'held_out': summarize(report, held_out), 'all': summarize(report),
                                      'scenes': {s['object']: {k: s[k] for k in (
                                          'robust_feasible', 'candidates', 'selected_index', 'selected_success',
                                          'selected_score', 'abstain_reason', 'brier_score')}
                                          for s in report['scenes']}}
        if own is not None and not heuristic:
            entry['loo_strict'] = leave_one_object_out(own, model['feature_names'], strict, args.threshold)
        results[key] = entry
    results['heuristic|tuned_loo'] = {'loo_strict': heuristic_loo(strict, HEURISTIC_WEIGHTS)}
    write_json(args.output, {'suite': str(args.suite), 'threshold': args.threshold,
                             'scopes': {scope: [str(d) for d in dirs] for scope, dirs in scopes.items()},
                             'held_out_objects': sorted(held_out), 'labels': labels, 'models': results})
    print('robust labels per scope:', json.dumps(labels, indent=1))
    header = f"{'model':42} {'eval scope':22} held-out: ok/fail/abst(false) brier"
    print(header)
    for key, entry in results.items():
        for scope in scopes if f'eval:{list(scopes)[0]}' in entry else []:
            h = entry[f'eval:{scope}']['held_out']
            brier = 'n/a' if h['mean_brier'] is None else f"{h['mean_brier']:.3f}"
            print(f"{key:42} {scope:22} {h['selected_success']}/{h['selected_failure']}/"
                  f"{h['abstained']}({h['false_abstain']})  {brier}")
        if key == 'heuristic|tuned_loo':
            loo = entry['loo_strict']
            ok = sum(v['selected_success'] is True for v in loo.values())
            bad = sum(v['selected_success'] is False for v in loo.values())
            print(f"{key:42} LOO (strictest labels)    {ok}/{bad}/0  weights "
                  + ', '.join(f"{o}={v['weight']:g}" for o, v in loo.items()))
        elif 'loo_strict' in entry:
            loo = entry['loo_strict']
            ok = sum(1 for v in loo.values() if v.get('selected_success') is True)
            bad = sum(1 for v in loo.values() if v.get('selected_success') is False)
            ab = sum(1 for v in loo.values() if 'selected_index' in v and v['selected_index'] is None)
            print(f"{'':42} LOO (strictest labels)    {ok}/{bad}/{ab}  "
                  f"{np.mean([v['brier_score'] for v in loo.values() if 'brier_score' in v]):.3f}")


if __name__ == '__main__':
    main()
