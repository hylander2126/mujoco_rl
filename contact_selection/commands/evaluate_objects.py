"""Leave-one-object-out comparison of selectors across the original and YCB objects.

For every object, the learned selector is refit on all *other* objects and its pick
on the held object is scored two ways: the robust label (passes in every included
physics scenario) and the fraction of scenarios it passes. The hardware heuristic,
the top-centre contact and the per-object oracle are scored on the same candidates.

Scenarios with no witnessed success are excluded (`eligible_only`), matching the
task assumption that the object is tippable; that is an evaluation scope, never
something the selector sees at deployment.
"""
import argparse
from collections import defaultdict
from datetime import date
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

from contact_selection.sim.dataset import read_records, write_json
from contact_selection.sim.rollout_evaluator import toppled
from contact_selection.selection.features import FEATURE_NAMES
from contact_selection.selection.heuristic import heuristic_index
from contact_selection.selection.selector import (feature_distances, fit_logistic, load_robust_contacts,
                                                  score_contacts, select_contact)

SUITE = Path('outputs/contact_selection/suites/2026-10-05_mass_force/features_ray')


def scenario_outcomes(directories: list[Path], scenarios: dict) -> dict:
    """(object, candidate index) -> pass/fail per included scenario."""
    included = {(s['directory'], s['candidate_set_id']) for v in scenarios.values() for s in v if s['included']}
    out = defaultdict(list)
    for directory in directories:
        for row in read_records(directory / 'rollouts.jsonl'):
            if (str(directory), row['candidate_set_id']) in included:
                out[(row['object_name'], row['candidate']['index'])].append(
                    bool(row['feasible'] and not toppled(row.get('metrics', {}))))
    return out


def score_pick(rows, index, outcomes):
    if index is None:
        return dict(index=None, robust=None, scenario_rate=None)
    row = rows[index]
    passes = outcomes[(row['object'], row['candidate']['index'])]
    return dict(index=row['candidate']['index'], robust=bool(row['robust_feasible']),
                scenario_rate=float(np.mean(passes)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmarks', type=Path, nargs='+', help='benchmark_objects outputs (their training/ folders are used)')
    p.add_argument('--suite', type=Path, default=SUITE, help='Original-object suite (features_ray layout)')
    p.add_argument('--exclude', nargs='*', default=[], help='Object names to drop (bad fixtures)')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    directories = sorted(d.parent for d in args.suite.glob('*/rollouts.jsonl'))
    for bench in args.benchmarks:
        directories += sorted(d.parent for d in (bench / 'training').glob('*/rollouts.jsonl'))
    contacts, scenarios = load_robust_contacts(directories, eligible_only=True)
    contacts = [{**row, 'split': 'train'} for row in contacts if row['object'] not in args.exclude]
    outcomes = scenario_outcomes(directories, scenarios)
    by_object = defaultdict(list)
    for row in contacts:
        by_object[row['object']].append(row)
    results = {}
    for held, rows in sorted(by_object.items()):
        train = [row for row in contacts if row['object'] != held]
        model = fit_logistic(train, feature_names=FEATURE_NAMES)
        scores = score_contacts(model, rows)
        distances = feature_distances(model, rows)
        learned = select_contact(rows, scores, args.threshold, distances, model['max_feature_distance'])
        center = int(np.argmin([np.linalg.norm(r['candidate']['press_offset_xy']) for r in rows]))
        rates = [np.mean(outcomes[(held, r['candidate']['index'])]) for r in rows]
        results[held] = dict(
            candidates=len(rows), robust_candidates=int(sum(r['robust_feasible'] for r in rows)),
            scenarios=sum(s['included'] for s in scenarios[held]),
            oracle=dict(robust=any(r['robust_feasible'] for r in rows), scenario_rate=float(max(rates))),
            mean_candidate_rate=float(np.mean(rates)),
            learned=score_pick(rows, learned, outcomes),
            learned_ignoring_abstain=score_pick(rows, int(np.argmax(scores)), outcomes),
            heuristic=score_pick(rows, heuristic_index(rows), outcomes),
            center=score_pick(rows, center, outcomes))
    totals = {key: dict(robust=sum(bool(r[key]['robust']) for r in results.values()),
                        abstain=sum(r[key]['index'] is None for r in results.values()),
                        mean_scenario_rate=float(np.mean([r[key]['scenario_rate'] or 0 for r in results.values()])))
              for key in ('learned', 'learned_ignoring_abstain', 'heuristic', 'center')}
    totals['oracle'] = dict(robust=sum(r['oracle']['robust'] for r in results.values()),
                            mean_scenario_rate=float(np.mean([r['oracle']['scenario_rate'] for r in results.values()])))
    report = dict(objects=len(results), totals=totals, per_object=results,
                  directories=[str(d) for d in directories], excluded=args.exclude)
    output = args.output or args.benchmarks[-1] / f'loo_comparison_{date.today():%Y-%m-%d}.json'
    write_json(output, report)
    print(f"{'object':22s} cand robust  oracle   learned        heuristic      center")
    for name, r in results.items():
        cell = lambda k: (f"{'abstain':>13s}" if r[k]['index'] is None else
                          f"{'✓' if r[k]['robust'] else '✗'} {r[k]['scenario_rate']:4.0%} #{r[k]['index']:<3d}  ")
        print(f"{name:22s} {r['candidates']:4d} {r['robust_candidates']:5d}   {r['oracle']['scenario_rate']:4.0%}  "
              f"{cell('learned')} {cell('heuristic')} {cell('center')}")
    print(json.dumps(totals, indent=1))


if __name__ == '__main__':
    main()
