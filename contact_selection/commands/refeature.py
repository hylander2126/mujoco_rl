"""Recompute pre-action features for saved datasets without re-simulating.

Features are a pure function of each candidate and its scene's saved geometry,
so a feature change does not need new rollouts (about 20 MB of trajectory per
contact). This writes a label-only copy -- rollouts.jsonl and scene manifests,
no trajectories -- that `train` and `compare` accept in place of the original.
Replay still needs the original dataset.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from contact_selection.sim.candidate_generator import Candidate
from contact_selection.sim.dataset import append_record, read_records, write_json
from contact_selection.selection.features import extract_features


def refeature(source: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    geometry = {}
    for path in sorted(source.glob('*/scene.json')):
        manifest = json.loads(path.read_text())
        geometry[manifest['candidate_set_id']] = manifest['geometry']
        (output / path.parent.name).mkdir()
        write_json(output / path.parent.name / 'scene.json', manifest)
    rows = read_records(source / 'rollouts.jsonl')
    (output / 'rollouts.jsonl').touch()
    for row in rows:
        features = extract_features(Candidate(**row['candidate']), geometry[row['candidate_set_id']])
        append_record(output / 'rollouts.jsonl', {**row, 'features': features,
                                                  'refeatured_from': str(source)})
    for name in ('config.json', 'provenance.json'):
        if (source / name).exists():
            (output / name).write_text((source / name).read_text())
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', type=Path, nargs='+', help='Completed contact-selection datasets')
    parser.add_argument('--output', type=Path,
                        help='Fresh folder; one label-only copy per dataset, by dataset name '
                             '(default: features_YYYY-MM-DD beside the first dataset)')
    args = parser.parse_args()
    args.output = args.output or args.datasets[0].parent / f'features_{date.today():%Y-%m-%d}'
    args.output.mkdir(parents=True, exist_ok=False)
    for source in args.datasets:
        print(f'{source.name}: {refeature(source, args.output / source.name)} records', flush=True)


if __name__ == '__main__':
    main()
