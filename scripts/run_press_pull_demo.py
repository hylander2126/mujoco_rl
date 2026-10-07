#!/usr/bin/env python3
"""Tip a basic box with the physical IRB120, then ease it back onto the table."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'glfw' if '--show-viewer' in sys.argv else 'egl')


def main() -> int:
    from parameter_estimation.press_pull_demo import BoxDemoConfig, run_demo
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/press_pull_box_demo',
                        help='New output directory; refuses to overwrite an earlier run')
    parser.add_argument('--show-viewer', action='store_true')
    parser.add_argument('--no-video', action='store_true', help='Run without offscreen rendering or ffmpeg')
    parser.add_argument('--object-y', type=float, default=0.0, help='Box center world Y (m); default centers it relative to the robot')
    parser.add_argument('--force', type=float, default=5.0, help='Press force (N)')
    parser.add_argument('--inset', type=float, default=0.006, help='Contact inset from the tipping edge (m)')
    parser.add_argument('--ground-friction', type=float, default=0.5)
    parser.add_argument('--finger-friction', type=float, default=2.0)
    parser.add_argument('--impratio', type=float, default=10.0)
    parser.add_argument('--noslip-iterations', type=int, default=10)
    parser.add_argument('--timestep', type=float, default=0.001)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    cfg = BoxDemoConfig(object_y_m=args.object_y, press_force_n=args.force, edge_inset_m=args.inset,
                        ground_friction=args.ground_friction, finger_friction=args.finger_friction,
                        impratio=args.impratio, noslip_iterations=args.noslip_iterations,
                        timestep=args.timestep)
    result = run_demo(cfg, args.output, video=not args.no_video, viewer=args.show_viewer,
                      verbose=not args.quiet)
    metrics = result['metrics']
    print(f"Feasible: {result['feasible']}; intended tip: {metrics['max_intended_tip_deg']:.2f} deg; "
          f"pivot drift: {1000 * metrics['max_pivot_drift_m']:.2f} mm; "
          f"ARC contact: {metrics['arc_contact_fraction']:.1%}")
    if result['failure_modes']:
        print('Failures: ' + ', '.join(result['failure_modes']))
    print(f'Saved results to {args.output}')
    return 0 if result['feasible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
