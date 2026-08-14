# Real2Sim pipeline — context notes

**Status: Steven is hand-writing the first implementation.** Chosen location:
top-level **`push2twin/`** (resolves the "directory layout" open question
below). As of 2026-08-13 it has `controllers/pipeline_fsm.py` (adapted from
`parameter_estimation/controllers/press_pull_fsm.py`) and
`scripts/main_genesis_sim.py` (WIP — mid-refactor, inlining
`GenesisRobotController.velocity_shove()` into the script body, per the
Genesis-code-doesn't-belong-in-the-submodule correction below). Per the
ownership split, Claude's role on the reused pieces (estimation, sim control
policy, deployment) stays scaffold/discuss, not hands-on-keyboard, unless
asked otherwise in a given session. This file exists so a future session
doesn't need the goal, scope, and ownership split re-explained. See
`CLAUDE.md` for the existing repo structure this pipeline builds on top of.

## Problem

The shove task (`parameter_estimation/scripts/shove_simulation.py`) currently
fails/chatters/tips on objects the robot has no model of — it has no geometry
and no physical parameters (mass/CoM/friction) for anything it hasn't already
been told about. Today `shove_simulation.py` reads both the approach pose and
the ground-truth mass/CoM straight out of `object_params.json` — there is no
path from "novel object in front of the robot" to a usable model.

## Goal

During the existing planar-push-and-rotate + tip procedure (the press/pull/tip
FSM at `parameter_estimation/controllers/`), capture **both**:

1. **Object geometry** — mesh, via multi-view capture while the procedure
   rotates/tips the object.
2. **Physical parameters** — mass, CoM, friction, via the existing estimator
   (`parameter_estimation/com_estimation.py` + the batch fit in
   `notebooks/main.ipynb`, eventually the windowed estimator in
   `ONLINE_ESTIMATOR.md`) — **unchanged**.

Assemble both into one full object model, export it to MJCF/URDF/USD, use it to
run the shove task in sim with a basic/classical control policy (**no RL, no
policy training**), then deploy back to the real robot and compare against the
no-model baseline (current chattering/tipping behavior).

## How this differs from existing Real2Sim work

Prior work in this space (e.g. Scalable Real2Sim, TwinAligner) identifies
object dynamics via **pick-and-place / grasping** interaction. This pipeline
uses **non-prehensile press/pull/tip** interaction instead — extending
real2sim identification to objects that can't be cleanly grasped (too flat,
too large, no graspable affordance, etc.), which is exactly the object set
this repo already targets (`box, heart, L, monitor, soda, flashlight`).

## Ownership split — read before writing any code in this pipeline

The user wants to write the implementation themselves for anything that reuses
existing estimator, control, or hardware code:

- Parameter estimation stage (mass/CoM/friction fit) — **reuse, unchanged**.
- Sim control policy for the shove task — classical, reuses
  `mujoco_irb120/robot/controllers/robot.py` (`controller` class: FK/IK,
  Jacobians, admittance/OSC) the same way `shove_simulation.py` and
  `press_pull_fsm.py` already do.
- Real-world deployment / no-model-baseline comparison — reuses the real-robot
  ROS 2 workspace (`~/Documents/github/irb120_ws`), out of scope for this repo
  directly.

The genuinely new-to-Steven parts are:

- **Geometry/mesh capture-and-reconstruction** — multi-view capture during the
  rotate/tip motion, reconstruction into a mesh.
- **Multi-format model export** — mesh + inertial parameters (mass, CoM,
  inertia tensor, friction) → MJCF/URDF/USD.

**Default for future sessions:** scaffold/discuss rather than write code for
the reused parts (estimation, sim control policy, real-world deployment/
comparison) unless explicitly asked otherwise in that session. Be more
hands-on — propose and write code — for the geometry/reconstruction and
multi-format export parts, since those are new territory.

## Relevant existing pieces to build on

- **Press/pull/tip FSM** — `parameter_estimation/controllers/press_pull_fsm.py`
  (+ `force_controller.py`, `motion_geometry.py`). Phase sequence
  `SQUASH → LULL → ARC → LULL → UNARC → RETRACT → DONE`. The ARC phase is the
  rotate/tip motion multi-view capture would piggyback on. Contact point comes
  from `site:ball_center`, not `FK()`'s `site:fingertip` (~0.18 m apart).
- **Estimator** — no `estimate(...)` function exists; the batch fit is
  `notebooks/main.ipynb` cells 8–9 (`scipy.optimize.least_squares` over
  `(com_z, mass, mu)`, offline, post-rollout). The windowed/online version is
  **specified but not implemented** — `parameter_estimation/ONLINE_ESTIMATOR.md`.
  Steven is building that separately; do not implement it here unasked.
- **Object model format** — see e.g.
  `mujoco_irb120/robot/assets/objects/box/box_exp.xml`: a MJCF body with
  `<inertial mass=... pos=... diaginertia=...>`, a primitive/mesh `<geom>` with
  `friction=` and `solref=`, plus `site:payload` and `site:obj_frame` (the
  tipping-edge frame that `com_gt` and the ARC pivot are both expressed
  relative to). Any exported model needs to reproduce this frame convention
  (or document the transform) to stay usable by the FSM and estimator as-is.
- **Ground truth / per-object config** — `parameter_estimation/object_params.json`
  (mass, CoM, `theta_star`, approach poses, `force_ref_n` per object). A
  captured/estimated model is the thing meant to eventually stand in for this
  file's contents on a genuinely novel object.
- **Push-face/tip-edge geometry** — `push_selection/push_selection_pipeline.py`
  already does pure-geometry mesh analysis (support polygon, tip edges, push
  faces, CoM projection) with `trimesh`. Likely reusable for reasoning about
  reconstructed meshes, not just the existing hand-authored ones.
- **Two robot controllers, do not merge** —
  `mujoco_irb120/robot/controllers/robot.py` (`controller`, used by
  `parameter_estimation/`) vs. `robot_learning/controller.py` (`Robot`/
  `PositionController`, used by `robot_learning/`). This pipeline is a
  `parameter_estimation/`-side effort, so it uses the submodule's `controller`.
- **Shared utilities** — `util/paths.py` (`REPO_ROOT`, `OUTPUT_ROOT`,
  per-subproject output dirs), `util/runtime.py` (device selection, video
  recorder). Any new subproject should follow the same `outputs/<name>/...`
  convention (gitignored).

## Correction: Genesis code does not belong in the submodule

Flagged 2026-08-13: a past session put a Genesis-backend controller *and* a
full shove-experiment script into the `mujoco_irb120` submodule
(`robot/controllers/genesis_robot.py`, `scripts/genesis_test.py`,
`robot/assets/{robot/genesis_robot.xml, objects/genesis_object.xml}`). That
violates the same rule `CLAUDE.md` already states for the MuJoCo controllers
("do not merge them," keep the submodule to universally-reusable robot code):
**the submodule should only hold assets and genuinely generic controller
code, not sim/experiment construction.**

Concretely, `mujoco_irb120/robot/controllers/genesis_robot.py` mixes two
things that need to split:

- **Generic (fine to stay in the submodule)** — `trapezoid_speed`,
  `smoothstep`, `ellipsoid_speed_scale`, `manipulability_speed_scale`,
  `quat_rotate`, `damped_least_squares_qdot`, `damped_pseudoinverse`,
  `limit_joint_velocity`, and `GenesisRobotController`'s low-level plumbing
  (`step`, `link_local_point_world`, `get_contact_jacobian`,
  `plan_ik_with_constraints`, `stop_velocity`) — these are Genesis-analogues
  of what `robot.py`'s `controller` class already provides for MuJoCo:
  general robot capability, no task baked in.
- **Task-specific (should NOT be in the submodule)** —
  `GenesisRobotController.velocity_shove()` is the entire shove-experiment
  logic (preshove pose, push direction, ramp profile, per-step logging), and
  the `__init__` defaults (`workspace_center/radii`, `contact_local_point`)
  are tuned to one specific box-push setup. This is sim construction, not a
  reusable controller method.

There is also a near-duplicate: `mujoco_irb120/scripts/genesis_test.py` and
`parameter_estimation/scripts/genesis_test.py` are ~95% identical (same
`gs.Scene` setup, same `velocity_shove` call, different path prefixes and a
couple of tuned constants) — confirmed by diff. That duplication is exactly
what "sim construction living in the submodule" produces: every subproject
that wants a Genesis smoke test ends up copying the whole script instead of
importing a shared piece.

**Decision:** going forward, Genesis (and any future backend's) *sim
construction* — scene assembly, task/experiment scripts, tuned per-task
constants — lives outside the submodule, in the main repo. Steven's stated
target for this is a `main_genesis_sim` location as part of the real2sim
work (exact path TBD alongside the rest of the layout in "Not yet decided"
below). The submodule keeps only the generic math/control-theory helpers and
the thin robot-capability wrapper, mirroring how `robot.py`'s `controller`
class is used today.

**Not yet done:** no files have been moved. This is a recorded decision, not
a completed refactor — treat `genesis_robot.py`/`genesis_test.py` as still in
their old, wrong location until a session actually does the split.

## Genesis feasibility check: can it set CoM and inter-object friction?

Asked 2026-08-13, before continuing `push2twin/scripts/main_genesis_sim.py` —
whether Genesis is worth pursuing over MuJoCo for the sim-control-policy
stage. Checked against the **installed** `genesis-world` 1.3.3 source
(`~/.virtualenvs/robot_learning/lib/python3.12/site-packages/genesis/`), not
docs, so this reflects what's actually available:

- **CoM: yes, at runtime.** `rigid_entity.RigidEntity.set_COM_shift(com_shift,
  links_idx_local=None, envs_idx=None)` — an offset from the geometry-computed
  CoM, settable per-link and per-parallel-env. Mass has the same shape:
  `set_mass_shift`, `set_links_inertial_mass`, `set_mass`.
- **Friction: yes to set, but combined the same way MuJoCo does.**
  `entity.set_friction(f)` / `set_friction_ratio(...)` set a per-link/per-geom
  value, but `contact.py`'s `func_set_contact` resolves an actual contact as
  `friction = max(max(friction_a, friction_b), 1e-2)` — **maximum
  combination**, identical to the MuJoCo gotcha already documented below.
  There is no independent, asymmetric, per-*pair* friction coefficient in
  either engine. Genesis additionally hard-clamps the settable range to
  `[1e-2, 5.0]` for stability, a constraint MuJoCo doesn't impose.
- **MuJoCo does the same things, confirmed by direct introspection** (not
  assumed): `model.body_ipos` (CoM offset) and `model.geom_friction` are both
  plain writable NumPy arrays on `MjModel` — `shove_simulation.py` already
  mutates `geom_friction` at runtime today for `MU_TABLE`.

**So on the specific question asked, the two engines are equivalent** — same
runtime CoM/mass control, same max-friction-combination limitation, no
capability gained by switching.

**Recommendation: stick with MuJoCo for this pipeline.** Reasoning:
1. Genesis's actual selling point — GPU-parallel batched rollouts — has no
   payoff here, since the goal explicitly excludes RL/policy training (a
   basic/classical control policy only).
2. The entire estimator + FSM + F/T-sensor pipeline this project reuses is
   verified against *MuJoCo's* contact solver specifically — including a
   currently-open, solver-regime-flavored bug (the box-doesn't-tip issue
   below, suspected `solref`-related). Porting to Genesis means re-deriving
   and re-verifying all of that against a different solver, for a stage
   (`sim_policy`) that was supposed to be a straightforward reuse of existing,
   working robot control code.
3. The existing Genesis code in this repo is itself evidence of the cost:
   half-ported, duplicated across `mujoco_irb120/scripts/genesis_test.py` and
   `parameter_estimation/scripts/genesis_test.py`, and the file open right now
   (`main_genesis_sim.py`) is mid-refactor. None of that exists on the MuJoCo
   side, which already has a working, documented, single controller.

**Not yet confirmed by Steven** — treat MuJoCo-only as a recommendation, not
a decision, until he says so.

## Known open issues that block/affect this pipeline

- **Object 0 (box) does not currently tip** in sim (`ONLINE_ESTIMATOR.md` §7)
  — FSM runs correctly but rotation stays <0.5°, drag force plateaus at ~1.0 N
  against a 6.2 N friction cone for an unestablished reason (suspect contact
  solver regime — box uses a very stiff `solref`). Multi-view capture during
  the ARC/rotate phase needs *some* rotation to be useful, so this is a
  prerequisite, not orthogonal.
- **The 1° tilt mask** in the batch/online estimator means no parameter
  estimate exists until the object is already tipping — relevant if the
  control policy wants a parameter estimate before deciding how hard to
  press/pull.
- **`shove_simulation.py:170`** currently commands index 4 of `[wx,wy,wz,vx,vy,vz]`
  (vy) while its comment says "+x direction" — a live bug in the very baseline
  this pipeline is meant to beat. Worth fixing before running comparisons.
- MuJoCo combines geom friction by **maximum**, not geometric mean — any
  friction values captured/estimated and re-exported into a new object XML
  need `friction=` values chosen with that combination rule in mind, not the
  `sqrt(mu1*mu2)` some existing display code prints.

## Not yet decided

- ~~Directory layout~~ — **resolved**: `push2twin/`, top-level, per Steven.
- Data flow and file formats between capture → estimation → assembly/export →
  sim control → real deployment.
- Which multi-view capture method (real-camera rig vs. sim-rendered synthetic
  views for prototyping) is used first.
- Physics backend for the sim control policy stage — MuJoCo recommended (see
  Genesis feasibility check above), **pending Steven's confirmation**. If
  MuJoCo is chosen, `main_genesis_sim.py` and the Genesis assets/controller
  become dead ends rather than something to finish splitting out of the
  submodule.
- The `genesis_robot.py` split (submodule cleanup) stays deferred either way
  until the backend question above is settled — no point moving Genesis code
  out of the submodule if Genesis itself is dropped.

Steven is now writing `push2twin/` himself; Claude's default stays
scaffold/discuss on the reused pieces (see ownership split at top) unless
asked otherwise in a session.
