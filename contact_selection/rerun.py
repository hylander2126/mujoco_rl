#!/usr/bin/env python3
"""Regenerate active datasets, sensitivity probes, selector evaluation, and videos into a fresh suite directory."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/contact-selection-mpl')

CONFIGS = ['box_mu_0p50', 'box_mu_0p20', 'box_mu_0p15', 'heart_l_mu_0p50',
           'l_mu_0p25', 'flashlight_mu_0p50', 'monitor_soda_mu_0p50']


def main():
    from contact_selection.generate import generate
    from contact_selection.visualize import plot, summarize
    from contact_selection.dataset import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh suite directory')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name in CONFIGS:
        print(f'RUN {name}', flush=True)
        cfg = json.loads((ROOT / 'contact_selection/config' / f'{name}.json').read_text())
        generate(cfg, args.output / name, workers=args.workers)
        plot(args.output / name)
        reports[name] = summarize(args.output / name)
        write_json(args.output / 'suite_summary.json', reports)
        print(f'COMPLETE {name}: {reports[name]["rollouts"]} rollouts', flush=True)
    from contact_selection.selector import train_and_save
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    train_and_save([args.output / name for name in CONFIGS], args.output / 'geometry_selector')
    subprocess.run([sys.executable, '-m', 'contact_selection', 'probes',
                    str(args.output), '--workers', str(args.workers)], check=True)
    tasks = []
    for name in CONFIGS:
        for scene in reports[name]['scenes']:
            tasks.append((name, scene['scene'], 0))
    tasks.extend([('box_mu_0p20', 'box_trial_01', 2), ('box_mu_0p15', 'box_trial_01', 1)])
    def record(task):
        name, scene, candidate = task
        subprocess.run([sys.executable, '-m', 'contact_selection', 'replay',
                        str(args.output / name), '--scene', scene, '--candidate', str(candidate)], check=True)
    with ThreadPoolExecutor(max_workers=min(3, args.workers)) as pool:
        list(pool.map(record, tasks))
    for name, dt in [('press_pull_box_demo_clean', '0.001'), ('press_pull_box_demo_halfstep', '0.0005')]:
        subprocess.run([sys.executable, '-m', 'contact_selection', 'demo',
                        '--output', str(args.output / name), '--timestep', dt, '--quiet'], check=True)
    print(f'Finished datasets, probes, selector, plots, and videos: {args.output}', flush=True)


if __name__ == '__main__':
    main()
