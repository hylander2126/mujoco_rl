"""Opt-in grippy box experiment using the IRB120 and existing PressPullFSM.

These are illustrative contact parameters, not calibrated hardware properties.
The original scene assets and default controller behavior are unchanged.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import mujoco
import numpy as np

from contact_selection.sim.candidate_generator import Candidate
from contact_selection.sim.dataset import write_json
from contact_selection.sim.rollout_evaluator import evaluate_rollout
from mujoco_irb120.robot.controllers.robot import controller
from parameter_estimation.controllers.press_pull_fsm import PressPullConfig, PressPullFSM
from parameter_estimation.scene import load_environment


@dataclass(frozen=True)
class BoxDemoConfig:
    """Parameters changed only on the freshly loaded demo model."""
    object_y_m: float = 0.0  # Align nominal box center with the robot symmetry plane.
    press_force_n: float = 5.0
    edge_inset_m: float = 0.006
    ground_friction: float = 0.5
    finger_friction: float = 2.0
    impratio: float = 10.0
    noslip_iterations: int = 10
    force_drop_fraction: float = 0.1
    timestep: float = 0.001

    def __post_init__(self):
        values = [self.object_y_m, self.press_force_n, self.edge_inset_m, self.ground_friction,
                  self.finger_friction, self.impratio, self.force_drop_fraction, self.timestep]
        if not np.isfinite(values).all():
            raise ValueError('Demo parameters must be finite')
        if not 0 < self.press_force_n < PressPullConfig().force_hard_limit_n:
            raise ValueError('Press force must be positive and below the existing force limit')
        if self.ground_friction < 0 or self.finger_friction < 0 or self.impratio <= 0:
            raise ValueError('Friction must be nonnegative and impratio positive')
        if self.noslip_iterations < 0 or not 0 < self.force_drop_fraction < 1 or self.timestep <= 0:
            raise ValueError('Invalid solver, force-drop fraction, or timestep')


def prepare_box(config: BoxDemoConfig, verbose: bool = True):
    """Return the physical box scene, near-edge contact, and controller config."""
    model, data = load_environment(0)
    payload = int(model.site_bodyid[model.site('site:obj_frame').id])
    joint = int(model.body_jntadr[payload])
    adr = int(model.jnt_qposadr[joint])
    # mj_setConst visits qpos0 and then qpos_spring; keep both reset references aligned.
    model.body_pos[payload, 1] = config.object_y_m
    model.qpos0[adr + 1] = config.object_y_m
    model.qpos_spring[adr + 1] = config.object_y_m
    data.qpos[adr + 1] = config.object_y_m
    model.opt.timestep = config.timestep
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = config.impratio
    model.opt.noslip_iterations = config.noslip_iterations
    ball = model.geom('push_ball_col').id
    table = model.geom('table').id
    model.geom_friction[ball, 0] = config.finger_friction
    model.geom_priority[ball] = 1  # Ball solref/solimp/condim win at its contacts.
    model.geom_friction[table, 0] = config.ground_friction
    mujoco.mj_forward(model, data)
    irb = controller(model, data)
    cfg = PressPullConfig(force_ref_n=config.press_force_n, max_normal_speed=0.005,
                          arc_force_drop_fraction=config.force_drop_fraction, verbose=verbose)
    fsm = PressPullFSM(irb, model, data, cfg)
    top = fsm.object_top_center()
    pivot = data.site_xpos[irb.obj_frame_site].copy()
    width = 2 * (top[0] - pivot[0])
    if not 0 < config.edge_inset_m < width / 2:
        raise ValueError(f'Edge inset must lie between 0 and half the box width ({width / 2:g} m)')
    point = top.copy()
    point[0] = pivot[0] + config.edge_inset_m
    cfg.press_offset_xy = (float(point[0] - top[0]), 0.0)
    local = data.xmat[irb.payload_body_id].reshape(3, 3).T @ (point - data.xpos[irb.payload_body_id])
    candidate = Candidate(0, point.tolist(), [0.0, 0.0, 1.0], local.tolist(), list(cfg.press_offset_xy))
    # Physical metadata come from the compiled model, not the stale object JSON.
    com_world = data.xipos[irb.payload_body_id].copy()
    metadata = {
        'preset': asdict(config), 'controller': asdict(cfg), 'candidate': candidate.to_dict(),
        'mass_kg': float(model.body_mass[irb.payload_body_id]),
        'com_body_m': model.body_ipos[irb.payload_body_id].copy(), 'pivot_world_m': pivot,
        'geometric_balance_angle_deg': math.degrees(math.atan2(com_world[0] - pivot[0], com_world[2] - pivot[2])),
        'collision_policy': 'adapter_object_disabled',
        'geom_contype': model.geom_contype.copy(), 'geom_conaffinity': model.geom_conaffinity.copy(),
        'geom_friction': model.geom_friction.copy(), 'ball_solref': model.geom_solref[ball].copy(),
        'ball_solimp': model.geom_solimp[ball].copy(), 'ball_condim': int(model.geom_condim[ball]),
        'ball_priority': int(model.geom_priority[ball]), 'mujoco_version': mujoco.__version__,
    }
    return model, data, candidate, cfg, metadata


class DemoDisplay:
    """Stream optional video without accumulating frames; optionally show viewer."""
    def __init__(self, model, initial_data, output: Path, video: bool, viewer: bool, *,
                 video_path: Path | None = None, playback_speed: float = 2.0):
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_copyData(self.data, model, initial_data)
        self.video = video
        self.show_viewer = viewer
        self.output = output
        self.video_path = video_path or output / 'demo.mp4'
        self.playback_speed = playback_speed
        self.renderer = self.viewer = self.encoder = self.encoder_log = None
        self.next_frame = 0.0
        self.next_sync = 0.0
        self.frames = 0
        self.start_wall = None
        self.options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.options)
        self.options.sitegroup[:] = 0  # Hide frame/site markers in the demo.
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        pid = int(model.site_bodyid[model.site('site:obj_frame').id])
        self.camera.lookat[:] = self.data.xpos[pid] + np.array([-0.04, 0, 0.035])
        self.camera.distance = 0.85
        self.camera.azimuth = 90
        self.camera.elevation = -12

    def __enter__(self):
        try:
            if self.video:
                if shutil.which('ffmpeg') is None:
                    raise RuntimeError('Video requires ffmpeg; install it or pass --no-video')
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
                self.encoder_log = (self.output / 'ffmpeg.log').open('w')
                self.encoder = subprocess.Popen([
                    'ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', '640x480', '-r', str(30 * self.playback_speed), '-i', '-', '-an', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.video_path)],
                    stdin=subprocess.PIPE, stderr=self.encoder_log)
            if self.show_viewer:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
                for key in ('distance', 'azimuth', 'elevation'):
                    setattr(self.viewer.cam, key, getattr(self.camera, key))
                self.viewer.cam.lookat[:] = self.camera.lookat
                self.viewer.opt.sitegroup[:] = 0
            return self
        except Exception:
            self.__exit__(*sys.exc_info())
            raise

    def __call__(self, model, data, candidate):
        if not self.video and not self.show_viewer:
            return
        if self.start_wall is None:
            self.start_wall = time.monotonic()
        if self.show_viewer and not self.viewer.is_running():
            raise KeyboardInterrupt('Viewer closed')
        if data.time + 1e-9 < min(self.next_frame if self.video else np.inf,
                                self.next_sync if self.show_viewer else np.inf):
            return
        mujoco.mj_copyData(self.data, model, data)
        if self.video and data.time + 1e-9 >= self.next_frame:
            self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
            frame = self.renderer.render()
            self.encoder.stdin.write(frame.tobytes())
            self.frames += 1
            self.next_frame += 1 / 30  # Capture 30 frames per simulation second.
        if self.show_viewer and data.time + 1e-9 >= self.next_sync:
            self.viewer.sync()
            self.next_sync += 1 / 60
            remaining = data.time - (time.monotonic() - self.start_wall)
            if remaining > 0:
                time.sleep(min(remaining, 1 / 60))

    def __exit__(self, exc_type, exc, traceback):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        status = 0
        if self.encoder is not None:
            try:
                self.encoder.stdin.close()
            except BrokenPipeError:
                pass
            status = self.encoder.wait()
        if self.encoder_log is not None:
            self.encoder_log.close()
        if status and exc_type is None:
            raise RuntimeError(f'Video encoding failed; see {self.output / "ffmpeg.log"}')


def run_demo(config: BoxDemoConfig, output: Path, *, video: bool = True,
             viewer: bool = False, verbose: bool = True) -> dict:
    """Run one full press/pull/return and save trace, model, reset state, and report."""
    import json
    model, data, candidate, cfg, metadata = prepare_box(config, verbose)
    output.mkdir(parents=True, exist_ok=False)
    # Reuse the independently defined feasibility checks, without relaxing them.
    thresholds_path = Path(__file__).resolve().parents[1] / 'contact_selection/config/box_mu_0p50.json'
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
