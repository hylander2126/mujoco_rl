"""Stream simulation frames to MP4, with optional interactive viewing."""
from pathlib import Path
import shutil
import subprocess
import sys
import time
import mujoco
import numpy as np


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
