#!/usr/bin/env python3
"""Record one saved contact as an MP4, excluding adapter/object collisions by default."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'glfw' if '--show-viewer' in sys.argv else 'egl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Saved run directory, scene.json, or config JSON with a saved run of the same name')
    parser.add_argument('--candidate', type=int, required=True, help='Contact index printed in the plot')
    parser.add_argument('--scene', help='For multi-object runs, e.g. heart_trial_01')
    parser.add_argument('--output', type=Path, help='MP4 path (default: SCENE/candidate_NNN.mp4)')
    parser.add_argument('--saved-collisions', action='store_true', help='Use saved collision masks for historical comparisons; default disables adapter/object contact')
    parser.add_argument('--finger-pitch', type=float, help='Override initial world-Y finger pitch in degrees; positive lifts the adapter')
    parser.add_argument('--show-viewer', action='store_true', help='Also display the live simulation')
    args = parser.parse_args()
    run = args.run
    if run.suffix == '.json' and run.name != 'scene.json':
        run = ROOT / 'outputs/contact_selection' / run.stem
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
    suffix = '' if args.finger_pitch is None else f'_pitch_{args.finger_pitch:+g}'
    if not args.saved_collisions:
        suffix += '_no_adapter_collision'
    output = args.output or scene.parent / f'candidate_{args.candidate:03d}{suffix}.mp4'
    if output.suffix.lower() != '.mp4':
        parser.error('--output must end in .mp4')
    from contact_selection.visualize import replay
    replay(scene, args.candidate, args.show_viewer, video_path=output, finger_pitch_deg=args.finger_pitch, saved_collisions=args.saved_collisions)


if __name__ == '__main__':
    main()
