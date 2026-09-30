# Why the off-axis plot was asymmetric

The original plot measured a real asymmetric response: the box center was at
**world Y=0.08 m**, although the graph translated the box center to plot Y=0.
That translation did not move the box relative to the robot. The centered
contact therefore had a roughly 0.19° ARC yaw change before considering the
additional effect of contact offset. All three signed yaw responses remained
positive; taking their magnitudes produced a sloping line instead of a V.

The box COM Y and geometric-center Y coincide. This was not a hidden COM offset
or an absolute-value bug. Moving only the box in world Y while keeping contact
positions relative to it fixed isolates a setup-dependent robot/controller
bias. It does not establish which individual controller or actuator term
produces that bias.

![Matched rotation and torque comparison](../outputs/contact_selection/y_offset_diagnostic/off_axis_comparison.png)

## Controlled comparison

All contacts have world X=0.580 m. Table friction is 0.5, finger friction is 2.0,
and adapter–object collisions are disabled in **both** placements. Geometry,
mass/inertia, initial finger orientation, controller settings, and thresholds
are unchanged. The original placement uses three mirrored contacts; the new
centered placement uses five. All eight configured trials pass.

| Contact Y relative to COM | Box at world Y=0.08 m: peak off-axis | Box at world Y=0 m: peak off-axis |
|---:|---:|---:|
| −44 mm | 0.0983° | 0.1142° |
| −22 mm | not tested | 0.0603° |
| 0 mm | 0.1956° | 0.0022° |
| +22 mm | not tested | 0.0608° |
| +44 mm | 0.2664° | 0.1144° |

The centered setup gives the expected V without changing the rotation formula,
subtracting a fitted baseline, or forcing symmetry. Signed yaw reverses across
zero; the remaining center response is small but nonzero. These are single
controlled simulations, not a statistical repeatability study or a guarantee
for other shapes, friction regimes, or robot poses.

## Torque versus rotation

For a force at displacement r from the COM:

`tau_x = r_y F_z - r_z F_y`, `tau_z = r_x F_y - r_y F_x`.

With a symmetric pull and negligible lateral force, changing the sign of r_y
changes the sign of these torque components; their magnitude has the expected
V shape. However, this box is supported by the table. The measured fingertip
roll torque at the old −44 mm contact, near 5° intended tip, is about +0.191 N m;
the table contributes about −0.191 N m. Off-axis rotation is therefore small
and depends on the residual supported motion, not fingertip torque alone.

New traces log fingertip force, fingertip/table/total contact torque **about the
actual COM in world coordinates**, COM position, and the diagnostic phase.
The contact-frame transformation and force sign follow
[MuJoCo's contact convention](https://mujoco.readthedocs.io/en/latest/computation/):
force acts on geom 2; the opposite sign is used when the object is geom 1.
The magnitude panel uses the evaluator metric. Signed-yaw diagnostics from
these saved controller traces differ by less than 0.0001° in magnitude because
the controller logs before its kinematics refresh; new evaluator traces record
the rotation at the same sampling point as the metric.
The plotted net contact torque excludes joint damping and is not described as
the complete rigid-body torque balance.

## Corrections

- The nominal box preset now physically places its center at **world Y=0**.
  `BoxDemoConfig.object_y_m` records this, and `--object-y 0.08` can restore the
  earlier placement in the standalone demo. Box X remains reachable at 0.58 m.
- Both `qpos0` and `qpos_spring` are updated. MuJoCo's
  [constant recomputation](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_setconst.c)
  evaluates the spring reference last; leaving it at Y=0.08 silently moved the
  dataset reset back there. A regression test covers this.
- Off-axis plots use **contact Y minus initial COM Y**, explicitly state the
  world X/Z and ARC-onset rotation reference, and show world placement in the
  figure title. No ARC is treated as unavailable, rather than zero rotation.
- The diagnostic figure shows signed yaw and torque as well as magnitude.
  Historical results retain their original saved physical pose and labels.

The unchanged feasibility metric is
`max_t norm(rotvec(R(t) R(ARC start)^T)[X,Z])` in degrees during ARC.
This is a rotation-vector magnitude, not torque and not simply the object's
absolute yaw in the world frame.

## See it and rerun

The analysis command generates the figure and replayable scene snapshots. It
reuses the eight completed diagnostic trials when their JSON/NPZ files exist:

```bash
.venv/bin/python -m contact_selection off-axis
```

The default source is the archived nominal box when available, otherwise the
current nominal box. `--source RUN` selects a saved box dataset explicitly.

Use a fresh `--output` directory to rerun all eight simulations. The output
includes `off_axis_comparison.png`, per-scene `contacts.png`, and `analysis.json`.

[Centered box, +44 mm contact video](../outputs/contact_selection/y_offset_diagnostic/box_y_0p00/candidate_004_no_adapter_collision.mp4)

```bash
.venv/bin/python -m contact_selection replay outputs/contact_selection/y_offset_diagnostic \
  --scene box_y_0p00 --candidate 4
```
