# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

MuJoCo simulation work around an **ABB IRB120** 6-DOF manipulator. The repo holds
three related-but-separate subprojects that share one robot model and one set of
mesh assets:

| Subproject | What it does | Scene | End effector |
|---|---|---|---|
| `parameter_estimation/` | Robot presses/tips an object; fits mass, CoM height, and friction from F/T data. | Meshed objects (box, heart, L, monitor, soda, flashlight) | `push_rod` + `fingertip` |
| `robot_learning/` | Behavior-cloning scaffold: collect scripted demos, train a BC policy, evaluate it. | Colored cube + two bins | flat `tray` |
| `push_selection/` | Geometry-only optimizer that picks *where* to push a mesh. | No sim; operates on meshes directly | n/a |

These three do **not** currently share a scene, a task, or a robot wrapper. Merging
them is the ongoing work (see *Current direction* below), not the existing state.

**The end-effector difference is the one that bites.** Estimation loads the
submodule's `robot.xml`, whose tool is a `push_rod` ending in a `fingertip` —
a point contact you can press and tip with. Learning loads
`robot_learning/assets/tray_robot.xml` instead, whose tool is a flat `tray`
(`site:tool`, `site:sensor`, `site:tray_center`) that a cube rides on top of.
Porting press/pull/tip into the learning env therefore means swapping the tool,
not just swapping the objects — and every F/T reading, IK target, and oracle
waypoint on the learning side is expressed relative to the tray.

`mujoco_irb120/` is a **git submodule** (`github.com/hylander2126/mujoco_irb120`)
holding the robot URDF/meshes, object meshes, and its own robot controller.

## Setup

There is **no** `pyproject.toml` or `setup.py` — `pip install -e .` will fail.
The project runs out of a virtualenv:

```bash
source ~/.virtual_environments/robot_learning/bin/activate
```

> `activate_venv.sh` in the repo root points at `~/.virtualenvs/robot_learning`,
> which does not exist. The real path is `~/.virtual_environments/`. The script is
> broken as committed; source the path above directly.

Key dependencies: `mujoco`, `numpy`, `scipy`, `torch`, `trimesh`, `h5py`,
`imageio`/`Pillow` (video), `mediapy` (optional, notebooks).

Headless machines: `env.py` auto-sets `MUJOCO_GL=egl` when no `DISPLAY`/`WAYLAND_DISPLAY`
is present. Override by setting `MUJOCO_GL` yourself.

## Running things

### Import paths matter — read this before running anything

Nothing is installed as a package, so `sys.path` has to be right or imports fail
in non-obvious ways. Two different conventions are in play:

**`robot_learning/` needs BOTH the repo root and `robot_learning/` on the path.**
`main.py` mixes root-relative imports (`from util.paths import ...`,
`from robot_learning.environment.scene import ...`) with sibling imports
(`from scripts.collect_sim_data import ...`). So, from the repo root:

```bash
PYTHONPATH=$PWD python robot_learning/main.py collect --episodes 5
PYTHONPATH=$PWD python robot_learning/main.py train --policy-type goal_conditioned
PYTHONPATH=$PWD python robot_learning/main.py eval --checkpoint outputs/robot_learning/checkpoints/vla_bc.pt
PYTHONPATH=$PWD python robot_learning/main.py diagnose
PYTHONPATH=$PWD python robot_learning/main.py plot
```

`python -m robot_learning.main --help` parses args but **breaks on every
subcommand** (the `from scripts.X` imports won't resolve). `python robot_learning/main.py`
without `PYTHONPATH` fails immediately on `util.paths`. Use the form above.

**`parameter_estimation/` and `push_selection/` need the repo root only:**

```bash
PYTHONPATH=$PWD python parameter_estimation/scripts/shove_simulation.py --no-viewer
PYTHONPATH=$PWD python push_selection/run_push_selection.py --top-k 3
```

### Known-broken entry points

- `parameter_estimation/scripts/photoshoot.py` finds its files now, but
  `scene.load_photoshoot()` composites every object into one scene and each
  object XML declares a site named `site:payload`, so MuJoCo rejects the model
  with `repeated name 'site:payload' in site`. Needs per-object site name
  prefixing to work.
- `simulation.py` defaults to `KEYBOARD_CONTROL = True` and expects a viewer
  window with focus — not usable headless without editing the flag.
- `shove_simulation.py:170` commands `shove_vel[4]`, which is **vy**, while its
  comment says "+x direction". Verified empirically: `v_cmd = [wx,wy,wz,vx,vy,vz]`,
  so index 3 is x and 4 is y. Either the comment or the index is wrong; the shove
  is currently sideways.

### The estimator is a notebook, not a module

This is the single most misleading thing about the repo. `parameter_estimation/com_estimation.py`
contains **only the wrench models** — `model_fwd_wrench`, `model_bkwd_wrench`,
`tau_app_model`, `tau_model`, `F_model`. There is no `estimate(...)` function anywhere.

The actual parameter fit lives in `parameter_estimation/notebooks/main.ipynb`,
cells 8–9: a `scipy.optimize.least_squares` over `(com_z, mass, mu)` run **offline,
in batch, on a loaded `.npz`** after a rollout has finished. Workflow is:

```
shove_simulation.py  →  outputs/parameter_estimation/rollouts/*.npz  →  main.ipynb cells 8-9
```

**Cell 8 masks out all samples below 1° of tilt** (`min_angle_mag = np.deg2rad(1)`;
comment: "model doesn't capture theta=0"). The estimator therefore cannot identify
anything until the object is already tipping. Any plan that feeds estimated
parameters to a controller *before* it decides how to push has to deal with this
first — it is a property of the model, not a tuning issue.

## Architecture

### Two robot wrappers — do not merge them

There are two independent implementations, and which one you get depends on which
subproject you are in:

- **`mujoco_irb120/robot/controllers/robot.py`** (submodule, 741 ln) — class is
  lowercase `controller`. Used by `parameter_estimation/`. Has FK/IK (3 damped-least-squares
  variants), Jacobians, F/T biasing + gravity comp, contact/topple detection,
  `get_payload_pose`, `get_tip_edge`, admittance and operational-space control.
  Its `ft_get_reading(grav_comp, apply_bias)` has **no** `flip_sign` argument.
- **`robot_learning/controller.py`** (local, 617 ln) — classes `Robot` and
  `PositionController(Robot)`. Used by `robot_learning/`. Its
  `ft_get_reading(grav_comp, apply_bias, flip_sign)` **does** take `flip_sign`.

They have diverged deliberately. Do not unify them as a refactor. When the learning
side needs estimator math (notably the **wrench → object-frame Adjoint transform**),
extract that function into `util/` and import it from both, rather than merging classes.

### Shared code

- **`util/paths.py`** — `REPO_ROOT`, `OUTPUT_ROOT`, per-subproject output dirs,
  `resolve_repo_path()`. Prefer this over hand-rolled `parents[N]` (see broken
  entry points above).
- **`util/runtime.py`** — `select_torch_device()` (CUDA with safe CPU fallback),
  `EpisodeVideoRecorder` (headless MP4 writer).
- **`util/rollout_dataset.py`** — HDF5 rollout dataset for BC training.
- **`util/debug_log.py`**, **`util/visualize_robot.py`**.
- **`mujoco_irb120/util/helper_fns.py`** — Modern Robotics wrappers, quaternion
  continuity, screw-theory conversions, Adjoint matrices.

### `robot_learning/`

Task is **bin sorting**, not press/pull/tip: place a colored cube into the matching bin.

- `task.py` — frozen `BinSortTaskSpec` dataclass + `HW1_TASK` default; helpers
  `randomize_bin_pose()`, `swap_bin_colors()` for layout ablations.
- `environment/scene.py` — composites `assets/scene_template.xml` +
  `sort_cube.xml`/`tray_robot.xml` into a generated scene in `$TMPDIR`.
- `environment/env.py` — sim loop, rendering, domain randomization.
  **Observation is 24-D** (`OBS_DIM = 24`), built at `env.py:376`:
  ```python
  ft  = self.irb.ft_get_reading(grav_comp=True, apply_bias=True, flip_sign=True)
  obs = np.concatenate([q, qdot, ft, ee_pos, obj_pos])   # 6 + 6 + 6 + 3 + 3
  ```
  Biased, gravity-compensated F/T is **already in the observation**. A vision-only
  vs. raw-force ablation is a masking config, not new plumbing.
- `hw1_oracle_policy.py` (106 ln) — the only live scripted oracle. Bin-sort only.
  Open-loop: it tracks elapsed time, not robot state.
- `models/policy.py` — `StateOnlyBCPolicy` (proprioceptive baseline),
  `GoalConditionedBCPolicy` (selected-bin pose + progress; the default),
  `TinyVLAPolicy` (image + language + state), plus `CharInstructionEncoder`.
  Selected by `--policy-type {goal_conditioned,vla,state_only}`; validated in
  `scripts/train_bc.py:98`.
- `scripts/` — `collect_sim_data.py`, `train_bc.py`, `eval_policy.py`,
  `diagnose_conditioning.py`, `plot_training.py`.
- `environment/default.yaml` — seed, image/video size, domain randomization,
  dataset/checkpoint paths, training hyperparameters. Note the long comment there:
  `action_noise_std` is deliberately the only randomization enabled, because the
  open-loop expert never demonstrates recovery.

### `parameter_estimation/`

- `scene.py` — `load_environment(num)`; objects keyed by ID:
  `0=box, 10=heart, 11=L, 12=monitor, 13=soda, 14=flashlight`.
- `object_params.json` — ground truth per object under a top-level `"objects"` key
- `com_estimation.py` — wrench models only (see above).
- `plotting.py` (520 ln), `rendering.py` — figures and offscreen render helpers.
- `controllers/` — press-and-pull FSM, ported from the real robot. See below.
- `scripts/press_pull_simulation.py` — press/arc/unarc rollouts (the experiment
  the estimator is fed by).
- `scripts/shove_simulation.py` — flat sliding push. Used for the friction
  estimate, which needs no tipping.
- `notebooks/main.ipynb` — the batch fit. `notebooks/simulation.ipynb`.
- `ONLINE_ESTIMATOR.md` — spec for the sliding-window estimator. **Not yet
  implemented**; Steven is building it. Do not implement it unasked.

#### `parameter_estimation/controllers/` — press-and-pull FSM

Simulation port of the hardware controller `irb120_ws/.../arc_static.py`:

    SQUASH -> LULL -> ARC -> LULL -> UNARC -> RETRACT -> DONE

`press_pull_fsm.py` holds `PressPullFSM` and `PressPullConfig`;
`force_controller.py` and `motion_geometry.py` are verbatim copies of the
hardware modules (pure math, no ROS) and should be kept in sync with them.
`STATE_IDS` matches the hardware log encoding — do not renumber, phase
segmentation downstream keys off those integers.

Four things that bite when working on this, all documented at length in the
module docstring:

- Pose comes from the **`site:ball_center`** site, not `FK()`'s `site:fingertip`
  — they are ~0.18 m apart along the rod.
- `ft_get_reading()` is **sensor-frame**; the arc projections need world frame.
  The hardware's `/netft_data_transformed` is already world-frame, so equivalent-
  looking hardware code is not equivalent.
- Debounce thresholds are **tick counts tuned at 100 Hz**, and the sim runs at
  1000 Hz. `PressPullFSM._ticks()` rescales them; anything new in that style
  must do the same or it fires 10x too early.
- **Slip and tip look identical in the force signal.** Tangential force collapses
  both when the object reaches its balance point and when the finger slides
  across its top face. The FSM disambiguates with the object's ground-truth
  rotation (`min_tip_angle_deg`) purely as an outcome label, never in the control
  law — the hardware equivalent is its vision-based object pitch stream. Never
  fit parameters from a rollout whose `tipped` flag is False.

**Known gap: the FSM does not yet tip object 0.** All phases, transitions, force
regulation and logging verified working, but every configuration tried ends with
`tipped=False` and < 0.5° of object rotation. Friction and actuation have both
been ruled out as the cause; the drag force plateaus at ~1.03 N against a 6.2 N
friction cone, for reasons not yet established. Full measurements and the
next things to try are in `ONLINE_ESTIMATOR.md` §7 — read it before re-deriving
any of this. Rollouts recorded today contain no tip, so nothing can be fit from
them yet.

Two friction facts worth not rediscovering: MuJoCo combines geom friction by
**maximum**, not geometric mean (so `shove_simulation.py:129-131`'s `sqrt(μ₁μ₂)`
"effective mu" printouts are wrong — display only, not used); and press force
appears on *both* sides of the tipping condition, raising available friction and
the restoring moment together, so pressing harder is not a general fix.

### `push_selection/`

`push_selection_pipeline.py` (1376 ln) — pure geometry, no MuJoCo and no robot.
Given a mesh and a 2D CoM projection: extract tip edges from the support polygon,
extract push faces from the top band, pair edges to faces whose horizontal normals
are **parallel** (`tip.inward_normal ≈ push.outward_normal`, i.e. the push face is
on the *opposite* side of the object from the tip edge), optionally check that the
line of action passes within `loa_epsilon` of the CoM, then score and rank.

`score_pair()` weights, defaulted in the function body (not a config file):
`orthogonality 5.0, tipping_ease 4.0, loa_closeness 3.0, leverage 1.5,
edge_stability 1.0`. Note `orthogonality` is always 1.0 by construction — the
perpendicular-slab pairing step upstream guarantees it — so despite being the
"primary ranking key" it does not discriminate between surviving candidates.
`loa_closeness` only contributes when `enforce_loa=True`, which is **not** the
default. `run_push_selection.py` is the CLI.

## Assets

Robot and object assets live in the **submodule**, at
`mujoco_irb120/robot/assets/` — `robot/` (+ `robot/visual/` meshes) and
`objects/{box,flashlight,heart,L,monitor,soda}/`.

Learning-task assets are local: `robot_learning/assets/{scene_template.xml,
sort_cube.xml, tray_robot.xml}`.

Both subprojects **generate their scene XML at runtime into `$TMPDIR`**
(`mujoco_irb120_hw1_binsort.xml`, `mujoco_irb120_parameter_estimation.xml`).
Never hand-edit a generated scene; edit the template it is built from.

## Outputs

Everything writes under `outputs/` (gitignored), namespaced by subproject:
`outputs/robot_learning/{rollouts,checkpoints,figures}`,
`outputs/parameter_estimation/rollouts`, `outputs/push_selection/`.
`.npz`, `.h5`, `.mp4` are all gitignored — rollout data does not survive a clone.

## Current direction

The goal is to fold the estimation work into the learning environment: a policy
that presses/pulls/tips objects, ablated across three observation conditions —
**(A)** no force, **(B)** raw F/T, **(C)** F/T plus *derived* physical parameters
(mass, CoM, friction).

Decisions and constraints an agent should know before proposing changes:

- **Condition C uses online/windowed estimation.** The batch least-squares fit is
  being reformulated to refit over a sliding window during the rollout and emit a
  confidence signal alongside the estimate, so the policy can learn to discount it
  while it is uninformative. Privileged-feature distillation was considered and
  not chosen.
- **The 1° tilt mask is the central open problem** for condition C — the estimate
  does not exist until tipping starts, which is after the push decision. The
  planned way out is to extrapolate: the balance angle θ\* is the *zero crossing*
  of a signal that is linear in tilt, so a line fit over a partial sweep predicts
  it before the sweep gets there. See `parameter_estimation/ONLINE_ESTIMATOR.md`.
- **The horizontal CoM (`com[0:2]`) is assumed known** from a previous trial on
  the same object. Both existing batch fits already assume this. Recovering it
  online is explicitly out of scope.
- **The press/pull/tip FSM has been ported into this repo** at
  `parameter_estimation/controllers/`, from the real-robot ROS 2 workspace
  `~/Documents/github/irb120_ws/src/irb120_ros2/irb120_control/irb120_control/`
  (`arc_static.py` is canonical; `arc_squash_pull.py` is marked deprecated in its
  own header; `adaptive_press.py` adds the escalating-force retry). The older
  `controllers/state_machine.py` deleted at `d5b40e1` is superseded — do not
  resurrect it.
- **`robot_learning/` is a side project**, not the main line. Project 1 (this
  press/pull/tip work) is expected to get its own top-level folder; the bin-sort
  BC scaffold stays where it is. Do not assume the two should converge.
- The object set for Project 1 is the estimator's meshed objects.

## Conventions

- Python 3.12. `from __future__ import annotations` and PEP 604 unions (`str | None`)
  throughout newer modules.
- NumPy for geometry, `float32` for anything crossing into torch.
- Frozen dataclasses for task/config specs; `dataclasses.replace()` for variants.
- Docstrings explain *why*, at length, where a choice is non-obvious — match that
  when the reasoning isn't self-evident from the code (see `task.py:swap_bin_colors`,
  `default.yaml`'s randomization comment).
- Real-robot code (`irb120_ws`) is ROS 2 and a separate repo. Don't import across.
