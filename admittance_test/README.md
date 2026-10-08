# IRB120 Admittance Test

Small, isolated MuJoCo demo for testing Cartesian force-to-motion control before hardware.

Run from the repository root with the project environment active:

```bash
PYTHONPATH=$PWD python3 admittance_test/run.py --steps 3000
```

For an interactive viewer on a graphical host:

```bash
PYTHONPATH=$PWD python3 admittance_test/run.py --show-viewer
```

In the viewer, `Up` and `Down` change the +X force target by 0.25 N and `Space` releases the force target. Close the viewer to stop.

The default demo loads the box scene, moves the tool just above the top face, holds its orientation, and commands +X motion from a 2 N force target. Use `--target-force` to change the target or `--save outputs/admittance_test/trace.npz` to save the trace.

The controller uses the existing IRB120 wrapper for kinematics and F/T sensing. MuJoCo position actuators receive integrated joint-position targets, so no existing controller or scene code is modified.
