#!/usr/bin/env python3
"""Record one saved contact as an MP4, excluding adapter/object collisions by default."""
import argparse
import os
from pathlib import Path
import sys

from util.paths import CONTACT_SELECTION_OUTPUTS

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'glfw' if '--show-viewer' in sys.argv else 'egl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Saved run directory, scene.json, or config JSON with a saved run of the same name')
    parser.add_argument('--candidate', type=int, required=True, help='Contact index printed in the plot')
    parser.add_argument('--scene', help='For multi-object runs, e.g. heart_trial_01')
    parser.add_argument('--output', type=Path, help='MP4 path (default: SCENE/candidate_NNN_no_adapter_collision.mp4)')
    parser.add_argument('--saved-collisions', action='store_true', help='Use saved collision masks for historical comparisons; default disables adapter/object contact')
    parser.add_argument('--show-viewer', action='store_true', help='Also display the live simulation')
    args = parser.parse_args()
    run = args.run
    if run.suffix == '.json' and run.name != 'scene.json':
        # A config maps to its dataset in the newest suite (suite folders are date-prefixed).
        matches = sorted((CONTACT_SELECTION_OUTPUTS / 'suites').glob(f'*/{run.stem}'))
        if not matches:
            parser.error(f'No saved run of {run.stem} under outputs/contact_selection/suites')
        run = matches[-1]
    if run.name == 'scene.json':
        scene = run
    elif args.scene:
        scene = run / args.scene / 'scene.json'
    else:
        scenes = sorted(run.glob('*/scene.json'))
        if len(scenes) != 1:
            parser.error('Choose --scene from: ' + ', '.join(p.parent.name for p in scenes)
                         if scenes else f'No saved scenes in {run}; pass the directory containing your generated run.')
        scene = scenes[0]
    if not scene.is_file():
        parser.error(f'Saved scene not found: {scene}')
    import json
    manifest = json.loads(scene.read_text())
    indices = [c['index'] for c in manifest['candidates']]
    if args.candidate not in indices:
        parser.error(f'Contact {args.candidate} not found; available indices: {indices}')
    suffix = ''
    if not args.saved_collisions:
        suffix += '_no_adapter_collision'
    output = args.output or scene.parent / f'candidate_{args.candidate:03d}{suffix}.mp4'
    if output.suffix.lower() != '.mp4':
        parser.error('--output must end in .mp4')
    from contact_selection.commands.visualize import replay
    replay(scene, args.candidate, args.show_viewer, video_path=output, saved_collisions=args.saved_collisions)


if __name__ == '__main__':
    main()
