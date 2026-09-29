# Contact-selection experiments: first milestone

This package generates a finite set of physically plausible top contacts, runs
the existing press-pull controller independently at each contact, and records
binary feasibility with separate physical diagnostics. It does **not** replace
the controller or parameter estimator. An exploratory geometry-only selector
is available; calibrated prediction and information ranking remain gated on
broader validation.
See [BOX_GRIP_RESULTS.md](BOX_GRIP_RESULTS.md) for the current 25-contact box
sweep, [PHYSICS_SWEEP_RESULTS.md](PHYSICS_SWEEP_RESULTS.md) for the
lower-friction box sweeps, [CROSS_GEOMETRY_RESULTS.md](CROSS_GEOMETRY_RESULTS.md)
for the heart and L mesh sweeps, [LEARNING_BASELINE_RESULTS.md](LEARNING_BASELINE_RESULTS.md)
for the exploratory selector, and [PILOT_RESULTS.md](PILOT_RESULTS.md) for the
earlier legacy-physics pilot.

## Current box contact tests

Contact-dataset generation now defaults to `config/box_grip.json`, reusing the
working [box demo](../parameter_estimation/PRESS_PULL_BOX_DEMO.md) setup directly:
5 N press, finger friction 2.0, table friction 0.5, elliptic cones, `impratio=10`,
10 no-slip iterations, arc-following wrist motion, and 10%-of-peak force exit.
This preset is restricted to the validated basic box; it does not silently
apply those settings to other objects.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/generate_contact_dataset.py \
  --output outputs/contact_selection/my_box_sweep --candidates 25
MPLCONFIGDIR=/tmp/contact-selection-mpl .venv/bin/python \
  scripts/visualize_contact_selection.py outputs/contact_selection/my_box_sweep
```

The demo contact 6 mm inside the tipping edge is proposed first, followed by
the centre heuristic and a spread over the top surface. The reference passes
all the usual geometry/IK filters; its known outcome is never used to bypass
feasibility checks. It counts toward the requested candidate total. A blue
outline diamond marks it in the label plot. `reference_contact_success` in the
summary makes it easy to spot regressions in the positive control.

Every candidate gets identical physics and reset state. Friction/solver fields,
resolved preset, reference identity and controller settings are saved in the
scene manifest and JSONL dataset. Replay loads the saved compiled model, so the
new solver/contact settings survive replay automatically. Source hashes now
include the demo module that supplies the preset.

Use `--config contact_selection/config/experiment.json` for the original
physics, or `--config contact_selection/config/pilot_geometry_pivots.json` for
the earlier mesh-pivot pilot. Those files and previous datasets are unchanged.
The new preset supplies simulator settings suitable for this box experiment;
it remains uncalibrated to real hardware. The exploratory learned selector is
documented below.

For mesh objects, the separate `arc_grip` preset sets finger, object, and table
friction explicitly. Use `config/mesh_arc_grip.json` for heart and L, or
`config/heldout_pose_arc_grip.json` for monitor and soda with recorded initial
pose overrides. These settings are exploratory; the box's validated preset is
still restricted to the box. See [CROSS_GEOMETRY_RESULTS.md](CROSS_GEOMETRY_RESULTS.md).

## Repository findings

* `parameter_estimation/scripts/press_pull_simulation.py` runs
  `PressPullFSM` with the IRB120 `controller`. The phases are SQUASH, LULL,
  ARC, LULL, UNARC, RETRACT, DONE. Approach uses IK and direct configuration
  placement; it is not a collision-free motion plan from the robot's home pose.
* `PressPullFSM.object_top_center()` uses geom AABBs. `press_offset_xy`
  selects the press point relative to this centre. `move_to_pre_squash()`
  places the ball above it, and descent finds the surface using force.
  ARC is constrained to world XZ, pulling toward -X about a world-Y edge.
  The default pivot comes from `site:obj_frame`; `arc_center_xz` is an existing
  override. The controller's success check uses **total rotation magnitude**,
  which can mislabel yaw as successful tipping.
* `parameter_estimation/scene.py` assembles the existing robot and object XMLs
  into a temporary scene; six existing objects are supported. Box collision is
  primitive; the others use convex mesh collision. Mesh coordinates must be
  transformed with compiled geom poses (MuJoCo recentres mesh vertices).
* `push_selection/push_selection_pipeline.py` already ranks support-edge / side
  push pairs. Its arbitrary edge and push-direction outputs cannot drive this
  fixed-plane, top-press controller without changing the interaction. It remains
  the analytical selector for the separate side-push experiment. Here the
  existing **top-centre press** is the comparison heuristic.
* `scripts/simulation.py` provides manual tipping; `shove_simulation.py` and
  `push2twin/controllers` provide other pushing modes. None is duplicated here.
  `push_selection/run_push_selection.py` is a geometry batch runner;
  `robot_learning/scripts/collect_sim_data.py` collects a different task.
* FSM logs contain sensor/world wrench, ball/sensor/object poses, contact flags,
  phase IDs, arc angles and force references. `check_contact()` also accepts rod
  contact; this package separately measures the intended **ball** contact.
* Existing estimation lives in `com_estimation.py` physics functions and an
  offline notebook fit. `ONLINE_ESTIMATOR.md` is explicitly a specification,
  references a fitter outside this repository, and documents failed box tipping.
  There is no validated callable press-pull fitter to invoke blindly. Native
  arrays are preserved and an estimator callback is provided; unavailable
  estimates/errors/information scores are explicit, never zero-valued stand-ins.

## Run

From the repository root (Python 3.10+):

```bash
python -m venv .venv
.venv/bin/pip install -r contact_selection/requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/generate_contact_dataset.py \
  --config contact_selection/config/experiment.json \
  --output outputs/contact_selection/my_pilot --objects 0 10 14 --candidates 25
MPLCONFIGDIR=/tmp/contact-selection-mpl .venv/bin/python \
  scripts/visualize_contact_selection.py outputs/contact_selection/my_pilot
.venv/bin/python -m pytest contact_selection/tests -q
```

Generation is headless and does not allocate a renderer. The output directory
must be new. `--config`, `--objects`, `--candidates`, and `--repeats` are supported.
Counts are upper bounds: impossible scenes and filtered proposals never get
padded with invalid contacts. Empty sets are preserved in the scene report and
included in oracle/coverage denominators. Seeds identify object/state groups;
repeats with fixed randomization ranges should reproduce the same physics.

Replay any saved candidate, optionally opening the viewer:

```bash
.venv/bin/python scripts/visualize_contact_selection.py \
  outputs/contact_selection/my_pilot \
  --replay-scene box_0_481830384 --candidate 0 --show-viewer
```

The first-milestone viewer replays an explicitly chosen candidate. There are no
predicted probabilities or learned-selected contacts until the training gate
passes. Offline plots show actual labels, candidate IDs, the centre heuristic,
intended rotation and ball-contact maintenance on the object footprint.

## Geometry and thresholds

`config/experiment.json` is the complete experiment configuration. JSON avoids
adding a YAML dependency. Controller defaults come from `PressPullConfig`, not
a second copy of its constants. Object-specific controller overrides can be
provided as `controller_by_object: {"heart": {"arc_center_xz": [x, z]}}`.
These must be justified geometrically and are persisted; default assets are
never silently corrected. Overrides change the controller pivot, **not** the
estimator's object-frame convention. Estimation needs explicit frame handling.
`config/pilot_geometry_pivots.json` records an exploratory box/heart/L sweep:
the heart and L pivot XZ coordinates use the near-X bottom hull bounds. This
is a nominal geometric approximation for the mesh support, not evidence that
the object maintains that pivot during execution; drift remains a label check.

Sampling intersects downward rays with the union of transformed convex
collision hulls, requires upward-facing normals, and uses a deterministic
farthest-point proposal order. IK and collisions at the approach pose are
checked with the existing robot and FSM. Constraints also enforce pivot span,
near-X support geometry, and a surface reachable within the SQUASH timeout.
This checks approach placement, not swept robot motion or the complete arc;
uncertain execution outcomes belong in the rollout labels.

New, provisional scientific thresholds are explicit:

| Configuration | Default | Interpretation |
|---|---:|---|
| edge_margin_m | 0.006 m | XY silhouette inset, not a full sphere-contact clearance proof |
| min_upward_normal | 0.95 | Restrict surfaces to near-horizontal tops |
| approach_tolerance_m | 0.002 m | Verify achieved ball placement after IK |
| pivot_tolerance_m | 0.01 m | Support-band / pivot-site compatibility tolerance |
| min_arc_contact_fraction | 0.9 | Required ball contact during ARC |
| max_pivot_drift_m | 0.01 m | Bound movement of the intended material pivot |
| max_off_axis_deg | 3° | Bound world-X/Z rotation during ARC |
| joint_limit_tolerance_rad | 0.01 rad | Allow small soft-limit numerical penetration |

These are assumptions to sensitivity-test, not experimentally validated values.
The rotation threshold (2°), force safety channel/limit (15 N), phase timeouts,
and contact-loss debounce are reused from the controller. Feasibility requires
all of: completed and done, established ball contact, sufficient ARC contact,
at least 2° about **world -Y**, bounded off-axis rotation and pivot drift, no
force-limit violation/abort, no joint violation or unintended robot contact,
and finite simulation state with no new MuJoCo warnings. Warnings conservatively
invalidate a rollout. A controller abort always invalidates apparent completion.

Pivot drift distinguishes translation of the support from legitimate body
translation during tipping. Body translation is also reported separately.
Ball travel in the object frame is a slip/rolling diagnostic, **not** a direct
measurement of tangential slip at the contact patch. Max contact force is from
MuJoCo contact forces; the safety margin uses the same measured force channel as
the FSM. Neither is disguised as an information score. No aggregate reward is
computed. Approach contact, pivot force distribution and full planned-path
reachability remain limitations of this first version.

## Data and reproducibility

Each run stores config and source hashes, software versions, JSONL outcomes,
and a manifest per object/state. Each scene has a compiled MJB model and the
complete `mjSTATE_INTEGRATION` vector. Each rollout receives a fresh `MjData`
copy of the same initial scene, a fresh robot controller and fresh FSM, then
the same approach/bias procedure as the original script. Robot approach differs
by contact, but initial object state and all physics parameters are identical.
Full per-tick FSM arrays plus contact/force/joint/pivot diagnostics are compressed
NPZ files. The initial-state hash connects candidates to their common reset.
When combining experiments, group by `(config_id, candidate_set_id)`; the human
readable scene ID alone can recur under different experiment configurations.
MJB replay requires a compatible MuJoCo version; source hashes and resolved
physical parameters help identify drift. This loader currently uses the shared
temporary XML path in the existing scene loader, so run generation serially.

Splits are assigned by object identity in configuration. All candidates and
repeats of an object share one split. Six geometries are too few for strong
generalization claims. Only conservative mass and commanded-force scaling are
implemented, disabled by default (ranges [1,1]); mass scaling also scales inertia.
Named presets now vary friction, and explicit initial-position overrides are
saved with the reset state. Broad geometry, CoM, pose, and sensor randomization
remain future work. Ground-truth mass/CoM are metadata, never classifier features.

## Training gate and next milestone

`summary.json` reports oracle availability, exact expected random-selection
success, centre-heuristic success and whether scenes have mixed labels.
Inspect spatial structure and repeatability before treating fitted scores as
reliable success probabilities.
The exploratory logistic selector classifies each candidate from pre-action
features, rejects scores below a configurable threshold, and abstains when
none survive its geometry-range check. Scores are uncalibrated; see
[LEARNING_BASELINE_RESULTS.md](LEARNING_BASELINE_RESULTS.md) for its object-held-out
development evaluation and limitations. Only accepted candidates may later be
ranked by a separately validated information-quality provider. Ground-truth
outcome scores are never deployable pre-action inputs.

A small MLP, probability calibration, a validated estimator adapter, and
secondary quality ranking remain future work. More object geometries with mixed
labels and a fresh test set are needed before claiming predictive performance.
