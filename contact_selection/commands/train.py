"""Train an exploratory geometry-only contact selector on saved scenarios."""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from contact_selection.features import FEATURE_NAMES
from contact_selection.selector import train_and_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+', help='Completed contact-selection datasets')
    parser.add_argument('--output', type=Path,
                        help='Fresh model/report directory (default: geometry_selector_YYYY-MM-DD beside the first dataset)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--exclude-features', nargs='+', default=[], choices=FEATURE_NAMES,
                        metavar='NAME', help='Ablate features (e.g. pivot_ray_angle_rad for the 15-feature baseline)')
    args = parser.parse_args()
    args.output = args.output or args.directories[0].parent / f"geometry_selector_{date.today():%Y-%m-%d}"
    names = [name for name in FEATURE_NAMES if name not in args.exclude_features]
    report = train_and_save(args.directories, args.output, args.threshold, names)
    print(json.dumps({key: value for key, value in report.items() if key != 'scenes'}, indent=2))
    for scene in report['scenes']:
        print(f"{scene['split']:10} {scene['object']:12} "
              f"robust={scene['robust_feasible']}/{scene['candidates']} "
              f"selected={scene['selected_index']} success={scene['selected_success']} "
              f"center={scene['center_success']}")


if __name__ == '__main__':
    main()
