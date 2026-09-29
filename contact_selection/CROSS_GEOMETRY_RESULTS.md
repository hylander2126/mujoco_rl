# Exploratory contact sweeps on mesh geometries

The box-specific `box_grip` preset is validated only on the box. For the heart
and L shaped collision meshes, `arc_grip` records a separate exploratory setup:
5 N press, finger sliding friction 2.0, matched object/table sliding friction,
elliptic cones, `impratio=10`, 10 no-slip iterations, arc-following wrist motion,
and 10%-of-peak force exit. The heart and L use the explicit near-X geometric
pivot overrides from the earlier pilot. No object mesh or controller law was
changed. These settings are not calibrated to hardware.

| Object | Object/table friction | Feasible | Centre heuristic | Random success |
|---|---:|---:|---|---:|
| Heart | 0.50 | 11/12 | fail | 91.7% |
| L | 0.50 | 12/12 | pass | 100% |
| L | 0.25 | 8/12 | pass | 66.7% |

The two L runs have identical candidate coordinates. Four contacts change from
pass at friction 0.50 to fail at 0.25. At 0.25, all four failures have pivot
drift above the 10 mm limit; one also has an unintended collision. On the
heart, the center candidate fails from an unintended collision while another
candidate passes. Replay reproduced the center failure: the robot's
`ft_and_adapter_link` geom contacted the heart during ARC at 8.366 s.

A small follow-up checked two representative contacts at nearby settings.
Heart center failure and candidate 1 success persisted at matched friction
0.45 and 0.55. For L, center success persisted at 0.23 and 0.27, but candidate
1 changed from failure at 0.23 to success at 0.27. These probes are saved under
`outputs/contact_selection/mesh_robustness_probe.json`; they cover only those
contacts and are not full candidate sweeps.

Full datasets and configs:

- `outputs/contact_selection/mesh_arc_grip_12` from
  [config/mesh_arc_grip.json](config/mesh_arc_grip.json).
- `outputs/contact_selection/L_arc_grip_025_12` from
  [config/L_arc_grip_025.json](config/L_arc_grip_025.json).

Reproduce in fresh output directories:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/generate_contact_dataset.py \
  --config contact_selection/config/mesh_arc_grip.json \
  --output outputs/contact_selection/my_mesh_arc_grip
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/generate_contact_dataset.py \
  --config contact_selection/config/L_arc_grip_025.json \
  --output outputs/contact_selection/my_L_arc_grip_025
```

These scenes provide mixed labels on the training heart and validation L
geometries, but there is no test geometry or common sampled physical-property
distribution across objects. The heart has only one negative, and one L label
is sensitive to a 0.02 friction change. This is enough to exercise the dataset
and selection evaluation path, not enough to claim a geometry-general trained
predictor or calibrated success probabilities. A geometry-only target must
specify which physical conditions its probability averages over; otherwise
the same contact can carry contradictory labels.

## Additional validation and test geometry coverage

The flashlight's near-X geometric pivot is recorded in
[config/flashlight_arc_grip.json](config/flashlight_arc_grip.json). Its top
surface supplied only five valid candidates out of 12 requested; all five
passed at matched object/table friction 0.5. The saved run is
`outputs/contact_selection/flashlight_arc_grip`.

Monitor and soda initially supplied zero reachable contacts because their
asset poses lie around world X=1 m. The experiment therefore records explicit
initial free-joint positions and geometric pivot overrides in
[config/heldout_pose_arc_grip.json](config/heldout_pose_arc_grip.json). The pose
change is applied after MuJoCo constant recomputation and appears in each
saved reset state and manifest. The completed run is
`outputs/contact_selection/heldout_pose_arc_grip_v2`:

| Test object | Reachable candidates | Feasible | Main outcome |
|---|---:|---:|---|
| Monitor | 4 | 0 | Contact loss; no intended rotation |
| Soda | 2 | 2 | Both tip cleanly |

These scenes test abstention and object-level feasibility, but neither contains
both labels. The first attempt at saving the pose override applied it before
`mj_setConst`, which reset `qpos`; that incomplete attempt is under
`outputs/contact_selection/heldout_pose_arc_grip` and must not be used for
training or evaluation. A regression test now checks the saved reset pose and
candidate count.
