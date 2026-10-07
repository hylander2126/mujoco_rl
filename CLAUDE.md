# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

Research simulation work around an **ABB IRB120** 6-DOF manipulator: non-prehensile
press/pull/tip interaction and estimating object parameters from it. Subprojects
share one robot model (the submodule's `robot.xml`, a `push_rod` ending in a
`fingertip`) and one set of object meshes:

| Subproject | What it does |
|---|---|
| `parameter_estimation/` | Robot presses/tips an object; fits mass, CoM height, and friction from F/T data. Also the hub the others import from (scene, FSM, controllers). |
| `contact_selection/` | Generates candidate top contacts, runs the press-pull FSM at each, labels feasibility, and trains an exploratory geometry-only selector. CLIs live in top-level `scripts/`. See `contact_selection/README.md`. |
| `push_selection/` | Geometry-only optimizer that picks *where* to push a mesh. No sim. |
| `push2twin/` | Real2sim prototyping (reconstruction + parameter capture). **Genesis-backed**, exploratory. See `REAL2SIM_CONTEXT.md` and `push2twin/README.md`. |
| `bundlesdf_poc/` | Paused BundleSDF proof-of-concept (see its README). |

**Robot learning (BC/VLA, bin-sort, tray end effector) moved out on 2026-09-28**
to `~/Documents/github/irb120_learning`. Dependencies point one way:
`mujoco_irb120` ← `irb120_learning` ← this repo. Learning code that is tied to a
research task (the contact selector, a future press/pull/tip policy, the A/B/C
ablation) stays here. Generic learning machinery lives there. Don't make that
repo import from this one.

`mujoco_irb120/` is a **git submodule** (`github.com/hylander2126/mujoco_irb120`)
holding the robot URDF/meshes, object meshes, and its own robot controllers
(MuJoCo `robot.py` and Genesis `genesis_robot.py`).

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
in non-obvious ways. Run from the repo root with the root on the path:

```bash
PYTHONPATH=$PWD python parameter_estimation/scripts/shove_simulation.py --no-viewer
PYTHONPATH=$PWD python push_selection/run_push_selection.py --top-k 3
```

The top-level `scripts/*.py` (contact selection, press-pull box demo) insert the
repo root into `sys.path` themselves and run from any directory.

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

### Robot wrapper

**`mujoco_irb120/robot/controllers/robot.py`** (submodule): class is lowercase
`controller`. Has FK/IK (3 damped-least-squares variants), Jacobians, F/T biasing +
gravity comp, contact/topple detection, `get_payload_pose`, `get_tip_edge`,
admittance and operational-space control. Its `ft_get_reading(grav_comp, apply_bias)`
has **no** `flip_sign` argument. `irb120_learning` has its own deliberately diverged
`Robot`/`PositionController`. Don't unify them. If both sides need the same math
(e.g. the **wrench → object-frame Adjoint transform**), put it in
`mujoco_irb120/util/helper_fns.py`.

### Shared code

- **`util/paths.py`**: `REPO_ROOT`, `OUTPUT_ROOT`, per-subproject output dirs,
  `resolve_repo_path()`. Prefer this over hand-rolled `parents[N]`.
- **`util/runtime.py`**: `select_torch_device()`, `EpisodeVideoRecorder` (headless
  MP4 writer). An independent copy also lives in `irb120_learning`.
- **`util/visualize_robot.py`**.
- **`mujoco_irb120/util/helper_fns.py`**: Modern Robotics wrappers, quaternion
  continuity, screw-theory conversions, Adjoint matrices.

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

**Tipping object 0 works via the box demo preset.** An earlier gap, where every
FSM configuration ended with `tipped=False` (history in `ONLINE_ESTIMATOR.md` §7),
is resolved for the basic box by `parameter_estimation/press_pull_demo.py`
(`PRESS_PULL_BOX_DEMO.md`, run via `scripts/run_press_pull_demo.py`): press 6 mm
inside the tipping edge, finger friction 2.0, elliptic cones, `impratio=10`,
no-slip iterations. That preset is validated for the box only. The mesh objects
use the exploratory `arc_grip` presets in `contact_selection/config/`.

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

Scenes are **generated at runtime into `$TMPDIR`**
(e.g. `mujoco_irb120_parameter_estimation.xml`, from
`parameter_estimation/scene_template.xml`). Never hand-edit a generated scene;
edit the template it is built from.

## Outputs

Everything writes under `outputs/` (gitignored), namespaced by subproject:
`outputs/parameter_estimation/rollouts`, `outputs/push_selection/`,
`outputs/contact_selection/`. Robot-learning outputs moved to `irb120_learning/outputs/`.
`.npz`, `.h5`, `.mp4` are all gitignored — rollout data does not survive a clone.

## Current direction

The goal is a policy that presses/pulls/tips objects, built here on the
estimation stack (not in `irb120_learning`'s bin-sort env, which uses a tray
tool). It will be ablated across three observation conditions —
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
- **Robot learning is a side project in a separate repo** (`irb120_learning`).
  Project 1 (this press/pull/tip work) is expected to get its own top-level
  folder here. If VLA/world models become research, import them from
  `irb120_learning` rather than moving code back.
- The object set for Project 1 is the estimator's meshed objects.
- **TODO (planned, not started): replace the in-repo estimator with the
  `press_pull_estimator` package** from `~/Documents/github/press-pull-tipping/code`
  (`pip install -e ../press-pull-tipping/code`). That package is the canonical
  core estimator. It reproduces the paper's Table 2 on the 40 real trials (to the last rounded digit), and
  `estimate_press_pull()` takes plain arrays through
  `Trial.from_streams(t_ft, ft, t_pose, pose, state)`, whose state ids match
  `STATE_IDS`. It should supersede `com_estimation.py` and the batch fit in
  `notebooks/main.ipynb` cells 8–9. The online estimator (`ONLINE_ESTIMATOR.md`)
  should build on it rather than on the older wrench models. The package's own
  MuJoCo box sandbox was removed. Simulation lives here.
- **`estimate_press_pull` fits ARC and UNARC separately and averages them.** That
  is intentional and stays. A single fit over all ARC + UNARC samples was tested
  and gave worse results, so don't switch to it.
  - **Open question:** the two sweeps disagree systematically on hardware. Mean over 10 trials, ARC vs UNARC: monitor mass
    5.403 vs 5.145 kg (~5%), flashlight 0.390 vs 0.402 kg and 9.26 vs 9.50 cm,
    heart z_c 11.41 vs 11.61 cm, box 0.704 vs 0.710 kg. Fingertip slip is ruled
    out as the cause. Candidates are controller lag, sensor drift/bias, or pivot
    creep between the sweeps. This matters for the online/windowed estimator: a
    window over only the pull sweep will be biased relative to the full-sweep fit.
    In a noise-free sim rollout of the box the sweeps agree to 0.4% (0.669 vs
    0.667 kg), so the model and geometry alone don't produce the hardware gap.
    Removing the measured F/T offset and a linear drift from the 40 hardware logs
    doesn't change the gap either (monitor +5.1% → +5.2%), so linear drift is ruled
    out too. Load-dependent sensor hysteresis and a resisting torque that reverses
    with the sweep direction remain. See `uncertainty/README.md`.
  - **The fingertip ball rolls on the top face** (the finger holds its orientation), so
    the ball-derived tilt reads `1 − rH/(H² + x0²)` low (~4–6%). The rolling affects
    only θ; the torque and force are measured. `uncertainty/hardware_check.py`
    applies the correction to the 40 trials. It isn't in `press_pull_estimator` yet.

## Conventions

- Python 3.12. `from __future__ import annotations` and PEP 604 unions (`str | None`)
  throughout newer modules.
- NumPy for geometry, `float32` for anything crossing into torch.
- Frozen dataclasses for task/config specs; `dataclasses.replace()` for variants.
- Docstrings explain *why*, at length, where a choice is non-obvious — match that
  when the reasoning isn't self-evident from the code (see the module docstring
  of `parameter_estimation/controllers/press_pull_fsm.py`).
- Real-robot code (`irb120_ws`) is ROS 2 and a separate repo. Don't import across.
