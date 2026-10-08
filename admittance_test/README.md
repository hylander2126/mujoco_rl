# IRB120 Gravity-Compensation Test

Small, isolated MuJoCo demo for applying external forces to the IRB120 tool in free space.

Run from the repository root with the project environment active:

```bash
PYTHONPATH=$PWD python3 admittance_test/run.py --steps 3000
```

For an interactive viewer on a graphical host:

```bash
PYTHONPATH=$PWD python3 admittance_test/run.py --show-viewer
```

In the viewer, `Up` and `Down` change the +X force target by 0.25 N and `Space` releases the force target. Close the viewer to stop.

An SSH terminal needs X11 forwarding and an X server on your local computer. Start a separate forwarded session, then verify that `echo $DISPLAY` is nonempty:

```bash
ssh -Y user@host
cd ~/Documents/github/mujoco_rl
source .venv/bin/activate
PYTHONPATH=$PWD python3 admittance_test/run.py --show-viewer
```

VS Code Remote SSH terminals do not necessarily provide X11 forwarding automatically. `Xvfb` can provide a hidden display for automated smoke tests, but it will not make a visible interactive window.

The demo removes the table and box, replaces the position actuators with torque motors, and applies MuJoCo's bias torque (`qfrc_bias`) at every step. The viewer's body perturbation tools can then apply small forces to the robot tip. Use `--save outputs/admittance_test/trace.npz` to save a diagnostic trace.

No existing controller or scene code is modified.
