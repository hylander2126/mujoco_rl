"""Record the centered near-pivot box demonstration."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from contact_selection.sim.box import BoxDemoConfig, prepare_box
from contact_selection.sim.dataset import write_json
from contact_selection.sim.rollout_evaluator import evaluate_rollout
from contact_selection.sim.video import DemoDisplay

ROOT = Path(__file__).resolve().parents[2]


def run_demo(config: BoxDemoConfig, output: Path, *, video: bool = True,
             viewer: bool = False, verbose: bool = True) -> dict:
    """Run one full press/pull/return and save trace, model, reset state, and report."""
    import json
    model, data, candidate, cfg, metadata = prepare_box(config, verbose)
    output.mkdir(parents=True, exist_ok=False)
    # Reuse the independently defined feasibility checks, without relaxing them.
    thresholds_path = Path(__file__).parents[1] / 'config' / 'box_mu_0p50.json'
    thresholds = json.loads(thresholds_path.read_text())['feasibility']
    metadata['feasibility_thresholds'] = thresholds
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial_state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, initial_state, state_spec)
    metadata['state_spec'] = int(state_spec)
    write_json(output / 'config.json', metadata)
    mujoco.mj_saveModel(model, str(output / 'model.mjb'))
    np.savez_compressed(output / 'initial_state.npz', state=initial_state)
    with DemoDisplay(model, data, output, video, viewer) as display:
        result, arrays = evaluate_rollout(model, data, candidate, cfg, thresholds,
                                          step_callback=display if video or viewer else None)
    result['video_frames'] = display.frames
    result['video_speedup'] = 2.0 if video else None
    np.savez_compressed(output / 'trajectory.npz', **arrays)
    write_json(output / 'result.json', result)
    return result


def main() -> int:
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
