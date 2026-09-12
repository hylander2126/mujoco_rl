# push2twin

Real2sim prototyping folder: capture an object's geometry *and* physical
parameters through non-prehensile push/pull/tip interaction, assemble both
into one object model, and use it to close the loop on the shove task. Full
goal, motivation, and how this differs from grasp-based real2sim work is in
**`REAL2SIM_CONTEXT.md`** at the repo root — read that first if you haven't.
This file is the detailed "what's actually built and what's next" — expect
it to go stale faster than the root doc and be more trustworthy about current
state as a result.

**Everything here is Genesis-backed, exploratory, and not wired into the real
estimator yet.** Genesis was chosen for this prototyping specifically because
Steven wanted to learn it, not because the MuJoCo-vs-Genesis backend question
for the actual pipeline has been settled — `REAL2SIM_CONTEXT.md`'s "Genesis
feasibility check" still recommends MuJoCo for the real thing. If that's
where it lands, the scene-construction code here doesn't carry over — only
`reconstruction/depth.py` (pure trimesh/numpy) and `assemble_model.py` (pure
trimesh/MJCF) do; neither has any Genesis or MuJoCo-sim dependency.

**New dependency:** `open3d` (0.19.0) — installed deliberately, once a
hand-rolled TSDF implementation and an open3d-based one existed side by side
and worked about equally well. No reason to maintain a from-scratch TSDF once
a real one is a `pip install` away; see "History" below.

## Layout

```
push2twin/
├── controllers/
│   ├── pipeline_fsm.py      # verbatim copy of parameter_estimation's press_pull_fsm.py
│   └── velocity_shove.py    # Genesis guarded Cartesian push -- used by main_genesis_sim.py's
│                             # smoke test; not currently called by rotate2construct.py, see below
├── reconstruction/
│   └── depth.py              # PinholeCamera + render_depth: synthetic depth camera,
│                             # ray-cast against a trimesh mesh
└── scripts/
    ├── main_genesis_sim.py  # smoke test: robot + box, one straight push via velocity_shove
    ├── trimesh_vis_test.py  # mesh loading, headless snapshot, ray-cast visibility test --
    │                        # ancestor of depth.py's render_depth
    ├── rotate2construct.py  # the capture + reconstruction prototype -- see below
    └── assemble_model.py    # the model-assembly prototype -- see below
```

### `controllers/pipeline_fsm.py`

Byte-for-byte copy of `parameter_estimation/controllers/press_pull_fsm.py`,
**unmodified**. Built entirely against
`mujoco_irb120.robot.controllers.robot.controller`'s API — none of which
exists on `GenesisRobotController`. **Cannot currently drive or be driven by
anything in `scripts/`.** No adapter exists; copied in as a starting point
and untouched since.

### `controllers/velocity_shove.py`

Genesis-backend guarded Cartesian push: trapezoid speed profile × workspace
ellipsoid scale × manipulability scale → damped-least-squares joint velocity
→ nullspace posture regulation → joint velocity clamp. Lives outside the
`mujoco_irb120` submodule on purpose — task-specific sim construction
shouldn't live in the submodule's controller, see "Genesis code does not
belong in the submodule" in `REAL2SIM_CONTEXT.md`. Still used by
`main_genesis_sim.py`'s single-push smoke test; **not currently used by
`rotate2construct.py`** — see that script's section and "History" below.

### `reconstruction/depth.py`

`PinholeCamera` + `render_depth`: a synthetic depth camera, standing in for
a real one, that ray-casts a pixel grid against a `trimesh.Trimesh` (nearest
hit per ray picked out vectorized — a per-point Python loop was a real ~1fps
bottleneck earlier in this folder's history). This is depth *synthesis*
only — never TSDF fusion itself, which is `open3d`'s job now (see
`rotate2construct.py`).

### `scripts/rotate2construct.py` — the capture + reconstruction prototype

The object rotates about world Z **kinematically** (position fixed, quat
teleported every tick) — **not** pushed by the robot right now, a deliberate
step back (see "History"). The robot is still in the scene for visual/scale
context; it isn't driven. At every tick, a fixed 'onboard camera' — world-
frame, not attached to the object — renders a synthetic depth image
(`reconstruction/depth.py`) and fuses it into an `open3d.pipelines.
integration.ScalableTSDFVolume`.

- The object's own local/canonical frame is where everything happens: since
  the object only rotates about its own origin (never translates) here,
  that frame is trivially stable, and it's also the frame a reconstructed
  model needs to end up in anyway. The TSDF volume lives there; only the
  fixed camera's pose, re-expressed in that frame each tick from the
  object's current orientation, changes between captures.
- `integrate_depth()` translates `PinholeCamera`'s basis vectors into the
  `(intrinsic, extrinsic)` pair open3d expects: `R = column_stack([right,
  up, forward])` is the camera-to-local rotation, so the extrinsic
  (local-to-camera) is `[[R.T, -R.T @ origin], [0,0,0,1]]`. Validated
  standalone against a known box (synthetic multi-view sweep, no Genesis)
  before being trusted here: recovered bounds within about a voxel width,
  ~109% of true volume, watertight.

Run it:

```bash
source ~/.virtualenvs/robot_learning/bin/activate   # see note below, not ~/.virtual_environments/
PYTHONPATH=$PWD python push2twin/scripts/rotate2construct.py --total-deg 360 --deg-per-step 6
```

Headless only. Outputs, all under `outputs/push2twin/` (gitignored):

| File | What |
|---|---|
| `rotate2construct_scene.mp4` | The Genesis scene — robot + object — so you can see what happened. (Short right now — see "Known issues".) |
| `rotate2construct_recon.mp4` | The reconstruction filling in as the sweep progresses. |
| `rotate2construct_mesh.stl` | The final reconstructed mesh, object-local frame. **This is the file `assemble_model.py` consumes.** |

A full 360°/6°-per-step sweep (60 captures) reconstructs all four side faces
of the box cleanly — confirmed visually, a closed "tube" shape — but stays
**open at the top and bottom**, since the camera only sweeps azimuth at a
fixed height and never looks from above or below. Expected, not a bug: the
sweep genuinely never observes those faces. `mesh.is_watertight` will be
`False` because of this; `assemble_model.py` already accounts for it (see
below).

> Aside, unrelated to this folder but discovered while working in it: `CLAUDE.md`
> says the venv is at `~/.virtual_environments/robot_learning` — that path
> doesn't exist on this machine. The real one is `~/.virtualenvs/robot_learning`.

### `scripts/assemble_model.py` — the model-assembly prototype

Combines a mesh (`rotate2construct.py`'s `.stl` by default) with mass/CoM/
friction into one standalone MJCF file, matching the style of every other
object asset in this repo. **First try / MVP — does not solve frame
reconciliation** (see "Known issues"); mass/CoM are taken as given, in
whatever frame the caller asserts, with no verification. Default `--mass` is
`object_params.json`'s **box** ground truth (0.676 kg) — an explicit
stand-in, since no object has been through both `rotate2construct.py` and
the real estimator yet.

Two modes:
- No `--com`: just `mass=` on the geom, same as `heart_exp.xml`/`L_exp.xml`
  — MuJoCo derives CoM (mesh centroid) and inertia from the geometry itself.
- `--com X Y Z`: an explicit `<inertial fullinertia=...>`, computed from the
  mesh (uniform-density assumption) then shifted to the given CoM via the
  parallel axis theorem, since trimesh's inertia tensor is about the mesh's
  *own* center of mass, not an arbitrary point.

**Non-watertight meshes use the convex hull for mass properties, not the raw
mesh — found out why the hard way.** `rotate2construct.py`'s meshes are
never watertight (open top/bottom, see above), and testing `--com` against
one caught a real bug: `trimesh.Trimesh.moment_inertia` on an open mesh
doesn't just get "less accurate" — the volume/inertia integrals assume a
closed surface, and can come back with **negative diagonal entries**, which
isn't a physically valid tensor at all. MuJoCo correctly rejected it
outright (`inertia must have positive eigenvalues`). Fix: when
`mesh.is_watertight` is `False`, mass properties are computed from
`mesh.convex_hull` instead — always closed by construction, so always valid,
at the cost of treating any concavity as solid.

Run it (defaults to `rotate2construct.py`'s last output):

```bash
PYTHONPATH=$PWD python push2twin/scripts/assemble_model.py
PYTHONPATH=$PWD python push2twin/scripts/assemble_model.py --com 0.0 0.0 0.0 --name my_object
```

Loads the result back through `mujoco.MjModel.from_xml_path` before exiting
and prints what MuJoCo actually computed — proof it's a real, loadable
model. Both modes verified this way against a real `rotate2construct.py`
mesh. Output: `outputs/push2twin/<name>.xml` (+ an **absolute** `meshdir` in
its `<compiler>` tag — simplest correct thing for a first-try prototype, not
portable across machines/checkouts).

## History: why the robot doesn't push anything right now

Earlier version of `rotate2construct.py` had the robot actually pushing the
object near its base, off-center, to induce rotation (`controllers/
velocity_shove.py`, still there, still used by `main_genesis_sim.py`).
Getting even one push to land took real debugging (a stale-height bug, and a
link-vs-contact-point offset bug in `GenesisRobotController`), and it
worked — push 1 in a run reliably made real contact and moved the object.
But push 2+ almost always missed: push 1 alone slides the object ~0.2–0.3 m,
past what's comfortably reachable for the next off-center approach (and, for
the camera used in the reconstruction, likely also enough to start losing
the object from frame). So reconstruction quality was bottlenecked on the
manipulation, not the reconstruction pipeline — which made it hard to tell
whether a bad reconstruction meant "the fusion is wrong" or "the object
barely moved."

Deliberate step back, both to isolate that and because Steven asked for it
directly: rotate the object kinematically (known-good motion, same idea as
the very first version of this script, before pushing existed) so the
reconstruction pipeline could be developed and trusted against a controlled
input first. Nothing about the push controller was thrown away —
`velocity_shove.py` and the geometry insight (off-center, near-base contact;
recompute approach pose from the object's *current* pose, not its spawn
pose) are exactly what's needed whenever this gets re-coupled to
`rotate2construct.py`.

## Known issues / open threads

- **Robot interaction is paused, not solved.** See "History" — re-coupling
  `rotate2construct.py` to a real push needs the push itself tuned (shorter/
  gentler, so the object doesn't leave reach after one hit) before it's
  worth trying again.
- **Frame reconciliation is still unsolved.** The estimator's CoM is
  relative to whatever frame `press_pull_fsm.py` uses (`site:obj_frame`, the
  tipping edge); the reconstructed mesh is in `rotate2construct.py`'s
  object-local capture frame. `assemble_model.py`'s `--com` takes a value in
  that frame on faith, with zero verification that a real estimate would
  already be expressed there.
- **No object has been through both stages yet.** `rotate2construct.py`'s
  object is a primitive Genesis box; the estimator has only ever run against
  `parameter_estimation`'s mesh objects (box/heart/L/monitor/soda/flashlight)
  in MuJoCo. `assemble_model.py`'s default mass is object_params.json's box
  ground truth used as a stand-in for exactly this reason.
- **Geometry mismatch, already worked around, but re-trip-able:**
  `genesis_object.xml`'s box (0.05 0.05 0.2 half-extents) is **not**
  `box_exp.stl` (0.05 0.05 0.15 half-extents) — different boxes.
  `rotate2construct.py` builds its local mesh from `genesis_object.xml`'s own
  dimensions (`box_full_extents()`), specifically to avoid assuming they
  match. Don't carry that assumption into a script that samples `box_exp.stl`
  directly.
- **The reconstructed mesh is always open at the top/bottom** with the
  current camera sweep (azimuth-only, fixed height) — see
  `rotate2construct.py`'s section. `assemble_model.py` handles the resulting
  non-watertight mesh correctly now (convex hull for mass properties), but
  the geometry itself is still missing real top/bottom detail, not just
  "technically not watertight." An elevation sweep, not just azimuth, would
  fix the geometry; nothing here does that yet.
- `mujoco_irb120/robot/controllers/genesis_robot.py`'s submodule cleanup
  (from `REAL2SIM_CONTEXT.md`) is only partly done — `velocity_shove()`
  itself got pulled out into this folder, but the duplicated
  `genesis_test.py` (submodule `scripts/` vs. `parameter_estimation/scripts/`)
  is still sitting there untouched, and cleaning it up is moot anyway if
  MuJoCo ends up the chosen backend.

## Next steps

- **Add an elevation sweep** to `rotate2construct.py`'s camera, not just
  azimuth, so the reconstruction actually closes (see known issues above).
- **Re-couple to a real push** once it's tuned to survive multiple attempts
  (see "History").
- **Solve frame reconciliation** between the estimator's CoM frame and the
  reconstruction's mesh frame.
- **Run a real object through both stages** — pick one of
  `parameter_estimation`'s existing objects (box is the obvious one, already
  the default `--mass`), get it into both `rotate2construct.py`'s Genesis
  scene and the real estimator, and see whether the two outputs actually
  combine into something sane.
- **URDF/USD export**, per `REAL2SIM_CONTEXT.md`'s original goal — MJCF was
  the deliberate first target since it's what this repo already speaks
  everywhere else; the other formats are unstarted.

The tentative full pipeline layout (`capture/`, `estimation/`,
`model_assembly/`, `sim_policy/`, `deployment/`) sketched in
`REAL2SIM_CONTEXT.md` before any of this existed is still just a sketch, not
approved — this folder grew organically instead, and `assemble_model.py` is
roughly what that sketch called `model_assembly/`. Worth reconciling the two
at some point, rather than letting the sketch keep drifting from reality.
