"""Train an exploratory geometry-only contact selector on saved scenarios."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from contact_selection.selector import train_and_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+', help='Completed contact-selection datasets')
    parser.add_argument('--output', type=Path, required=True, help='Fresh model/report directory')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    report = train_and_save(args.directories, args.output, args.threshold)
    print(json.dumps({key: value for key, value in report.items() if key != 'scenes'}, indent=2))
    for scene in report['scenes']:
        print(f"{scene['split']:10} {scene['object']:12} "
              f"robust={scene['robust_feasible']}/{scene['candidates']} "
              f"selected={scene['selected_index']} success={scene['selected_success']} "
              f"center={scene['center_success']}")


if __name__ == '__main__':
    main()
