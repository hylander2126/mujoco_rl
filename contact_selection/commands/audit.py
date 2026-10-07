"""Audit completed sweeps without treating unwitnessed feasibility as selection error."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from contact_selection.selection.heuristic import heuristic_index
from contact_selection.sim.dataset import write_json
from contact_selection.sim.rollout_evaluator import toppled


def audit(roots):
    scenes = []
    paths = sorted({p for root in roots for p in Path(root).rglob('rollouts.jsonl')
                    if 'features_ray' not in p.parts})
    for path in paths:
        groups = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups[row['candidate_set_id']].append(row)
        for scene_id, rows in groups.items():
            manifest = json.loads((path.parent / scene_id / 'scene.json').read_text())
            indices = [r['candidate']['index'] for r in rows]
            if len(indices) != len(manifest['candidates']) or len(set(indices)) != len(indices):
                raise ValueError(f'Incomplete/duplicate rollouts: {path}/{scene_id}')
            valid = [r['feasible'] and not toppled(r.get('metrics', {})) for r in rows]
            pick = heuristic_index(rows)
            reasons = Counter()
            for row in rows:
                failures = set(row['failure_modes'])
                if toppled(row.get('metrics', {})):
                    failures.add('toppled')
                reasons.update(failures)
            scenes.append(dict(directory=str(path.parent), scene=scene_id, object=rows[0]['object_name'],
                candidates=len(rows), successful_contacts=sum(valid), sampled_eligible=any(valid),
                selected_candidate=rows[pick]['candidate']['index'], selected_success=bool(valid[pick]),
                failures=dict(reasons), historical_topple_label_corrections=sum(
                    r['feasible'] and toppled(r.get('metrics', {})) for r in rows)))
    eligible = [s for s in scenes if s['sampled_eligible']]
    return dict(summary=dict(scenarios=len(scenes), candidates=sum(s['candidates'] for s in scenes),
        eligible_scenarios=len(eligible), unresolved_scenarios=len(scenes)-len(eligible),
        selected_successes=sum(s['selected_success'] for s in eligible),
        selection_failures=sum(not s['selected_success'] for s in eligible),
        historical_topple_label_corrections=sum(s['historical_topple_label_corrections'] for s in scenes)),
        interpretation='Eligibility is a sampled success witness in this scenario, not proof of physical untippability when absent.',
        scenes=scenes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = audit(args.roots)
    write_json(args.output, report)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
