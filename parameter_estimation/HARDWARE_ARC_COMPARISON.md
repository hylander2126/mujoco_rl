# Comparison with irb120_ros2

Inspected `irb120_control/irb120_control/arc_static.py` at source commit
`83ee38617f9ef30b077074c0072a1c289407aa74` and the current sensor/finger assemblies.
Tool provenance and regeneration are documented in
`mujoco_irb120/robot/assets/robot/hardware_tool/README.md`.

Hardware `_publish_arc_step` computes the ball-center XZ arc velocity, adds
radial PI force correction, and explicitly sets `wy = 0.0` for ARC and UNARC.
There is no rolling-contact correction or arc-following wrist rotation. The
simulation now defaults to the same fixed world orientation, with bounded
orientation-error feedback because zero angular velocity alone drifts under the
simulated actuator/contact dynamics. Both rotation modes compensate the rigid
flange-to-ball velocity offset; that is not rolling-contact compensation.

The PI force-controller implementation and shared XZ arc geometry math match
hardware (the simulation additionally extracts the tangent projection helper).
The state machines are not claimed to be identical:

- Hardware low-force exit: 8% of peak, with a 0.1 N floor. The demo retains its
  validated 10%-of-peak exit; the base simulator retains its original threshold.
- Hardware hard force limit is now 20 N to accommodate Servo overshoot; simulation
  retains 15 N. Hardware pre-squash standoff is 30 mm; simulation uses 20 mm.
- Hardware has lateral force correction and detection-derived arc geometry;
  simulation uses its configured/object-site pivot and zero commanded Y velocity.
- Hardware's force-settling LULL code is commented out. The active behavior is
  zero twist and a one-second wait, matching the simulator's phase structure.
- Hardware uses MoveIt Servo; simulation integrates differential IK joint targets
  into position actuators. Ground truth object rotation only labels outcomes.

The hardware estimator still contains a stale 82.25 mm tool-stack constant;
`ft_link` is also still referenced by logging/preprocessing despite being absent
from the current URDF. No ROS files were changed. The imported geometry uses the
actual fixed-joint chain, not those stale constants.

Historical videos/datasets remain historical. `outputs/arc_static_hardware_synced`
contains the new compiled model, configuration, trajectory, results and video.
