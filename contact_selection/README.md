# Contact selection

This package answers one question: *where on top of an object should the
press-pull controller press so that the object tips cleanly?*

## How it works

1. **Candidates.** Sample points on the object's top surface. Each one has to
   pass top-surface, edge, pivot, approach-IK and collision checks.
2. **Label by simulation.** Run the fixed press-pull FSM (static wrist, 5 N press)
   at every candidate, starting from the same saved reset state. A contact is
   *feasible* when the object tips toward world −Y with sustained fingertip
   contact, bounded pivot drift and off-axis rotation, and no controller, force,
   joint, collision or numerical failure.
3. **Robust label.** Each object is run under several configs (table friction,
   mass ×0.5/×2, force ×2.6). A contact counts as *robust* only if it is
   feasible in **every** config. Table friction is what moves these labels;
   mass and force change nothing at μ = 0.5.
4. **Selector.** A logistic regression over pre-action geometry only. It uses
   16 features: contact position and normal, pivot offsets `pivot_dx_m` and
   `pivot_dz_m`, the pivot ray angle `atan2(dx, dz)`, bounding box, and IK
   margins. No outcome or ground-truth physics goes into it. It picks the
   highest-scoring candidate above 0.5 that lies within the training feature
   range, and otherwise abstains. Scores are uncalibrated. Splits are by
   object.

Main failure mode: the press pushes the pivot sideways in proportion to the
tangent of the ray angle. At low friction the pivot slides. Objects whose
centre of mass sits far out from the pivot, like the heart, don't follow this
pattern.

Run these commands from the repository root after installing `requirements.txt`
in `.venv`. Video recording also requires `ffmpeg` on PATH. Saved outputs are
local artifacts; on a fresh checkout, generate a sweep or run the full suite
before replaying it. See [available configs](config/README.md).

## Watch a contact

Record one contact from a saved run as a normal-speed MP4:

```bash
.venv/bin/python -m contact_selection replay outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p20 --candidate 2
```

Open `outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p20/box_trial_01/candidate_002_no_adapter_collision.mp4`.
The command prints the video path and outcome. Contact numbers are the numbers
on the contact plot. Add `--scene heart_trial_01` for a multi-object run, or
`--show-viewer` to also watch live. `--output clip.mp4` chooses the video path.
You can also pass `contact_selection/config/box_mu_0p20.json` to find the newest
saved run of that config under `outputs/contact_selection/suites/`.

This reruns only the selected contact from its saved model, full reset state,
and controller settings. By default, it disables **adapter–object collisions**
in that model. Fingertip–object, object–table, and all other pair eligibility
remain unchanged. `--saved-collisions` preserves the saved collision settings
for historical comparisons. It uses the saved configuration, not edits to the
source config. It saves an MP4 and a small JSON outcome beside it; rerunning
replaces that contact's video. You do not need to generate the sweep again. New scenes use the same adapter
exclusion. Collision masks/policy are recorded in new dataset metadata and
preserved in MJB snapshots; replay JSON records whether saved masks were used.
This is a simulation simplification: adapter overlap is permitted. Contact
selection heuristics are unchanged; no historical failures are relabeled.

## Generate a new full sweep (data only)

Use this only to collect new results for all contacts. It can take minutes and
saves numerical traces, not videos. From the repository root:

```bash
.venv/bin/python -m contact_selection generate --candidates 25
.venv/bin/python -m contact_selection plot outputs/contact_selection/sweeps/YYYY-MM-DD_box_mu_0p50
```

Then use `.venv/bin/python -m contact_selection replay outputs/contact_selection/sweeps/YYYY-MM-DD_box_mu_0p50
--candidate 0` to record a chosen point.

The default config is [box_mu_0p50.json](config/box_mu_0p50.json), which uses the
working box demo's 5 N press and contact physics. `--config` accepts the
exploratory mesh and friction variants in `config/`; `--objects`,
`--candidates`, and `--repeats` override their configured values. Box-grip
physics applies only to the box. Mesh `arc_grip` presets record their own
friction and, where needed, pivot and initial-pose overrides.

Scene folders use object and repeat names, such as `box_trial_01`; the
random seed stays in the manifest. `mu` in box dataset names is table sliding
friction (for example, `box_mu_0p20` means 0.20).

Saved runs include JSONL labels, scene manifests, compiled MJB models, full
reset states, rollout arrays, and source/config provenance. Replay one candidate
with `.venv/bin/python -m contact_selection plot OUTPUT --replay-scene SCENE
--candidate INDEX`; this also saves an MP4 by default. Scene preparation is serial; `--workers 8` evaluates independent rollouts in
parallel from saved snapshots, without sharing mutable simulation state.

## Off-axis rotation: corrected setup

The default box now sits at **world Y=0**, physically centered relative to the
robot. Previously only the plot axes were centered; the box itself was at
world Y=0.08 m. A controlled five-contact sweep now shows the expected V-shaped
off-axis response. See [the analysis and corrected figure](OFF_AXIS_RESULTS.md).
All seven active configs have now been rerun. `simulation_preset.parameters.object_y_m` sets the box center explicitly.

## Results

All active results were refreshed on 2026-09-30: **122 main rollouts, 33
sensitivity trials, retrained selection, regenerated plots, 13 main replay
videos, and both standalone timestep demonstrations**.
[Full rerun summary and video links](RERUN_RESULTS.md).

| Experiment | Current outcome |
|---|---|
| [Nominal box](BOX_GRIP_RESULTS.md) | 25/25 pass; maximum off-axis rotation 0.118°. |
| [Box friction sweep](PHYSICS_SWEEP_RESULTS.md) | 20/25 at friction 0.2; 10/25 at 0.15. Failures now reflect motion/execution rather than adapter collision. |
| [Meshes](CROSS_GEOMETRY_RESULTS.md) | Heart 12/12, L 8/12 at 0.25, flashlight 5/5, monitor 0/4, soda 2/2. |
| [Retrained selector](LEARNING_BASELINE_RESULTS.md) | Heart selection changes; flashlight now falsely abstains at the unchanged 0.5 threshold. |
| [Mass/force sweeps, ray-angle feature](MASS_FORCE_RESULTS.md) (2026-10-05) | Mass ×0.5/×2 and a 13 N press change no robust label at friction 0.5; mass matters only at low friction, where its sign flips. Under one friction envelope for every object, heart and flashlight have no robust contact. The new feature avoids one failing pick there but falsely abstains on the flashlight under the original labels. |

To rerun the complete active suite into a fresh folder, including probes,
selector evaluation, plots, and representative videos:

```bash
.venv/bin/python -m contact_selection rerun --name my_label --workers 8
```

Outputs land in dated folders under `outputs/contact_selection/` unless `--output`
is given: `rerun` → `suites/YYYY-MM-DD_NAME/`, `generate` and `off-axis` →
`sweeps/YYYY-MM-DD_NAME/`. `train`, `refeature` and `compare` write next to the
datasets they read, e.g. `geometry_selector_YYYY-MM-DD/`.

The suite now runs 19 configs: friction plus the mass and force variants. Each
rollout saves about 20 MB of trajectory, so a full suite needs about 9 GB free.
`compare SUITE` fits every feature set × label scope and
scores each against every scope, including leave-one-object-out. `refeature`
recomputes features for saved datasets after a feature change, without
re-simulating. `train --exclude-features NAME` fits an ablation.

The static-wrist baseline selector is
`outputs/contact_selection/suites/2026-09-30_static_wrist/geometry_selector`.
The 16-feature ray-angle selector is
`outputs/contact_selection/suites/2026-10-05_mass_force/geometry_selector_ray`. It is
better on some held-out objects and worse on others; see
[its comparison](MASS_FORCE_RESULTS.md#5-selector-comparison) before adopting it.
This is supervised learning from simulator outcomes, not RL policy training.
[How it relates to bandits and RL](LEARNING_BASELINE_RESULTS.md#relationship-to-rl).

The older unmodified-physics pilot had no feasible box contacts, rejected
heart/flashlight pivot sites, and found one marginal L success only after
explicit pivot overrides. It motivated the recorded physical presets and
execution diagnostics; its configs and long-form result report were retired.

## Method and limits

Candidates pass top-surface, edge, pivot, approach-IK, and collision checks.
The centre top press is the heuristic baseline. Feasibility requires completed
intended world -Y tipping, sustained ball contact, bounded pivot drift and
off-axis rotation, and no controller, force, joint, collision, or numerical
failure. The default thresholds are recorded in each experiment config. An
approach check does not prove a collision-free path from robot home.

Ball-contact fraction is the fraction of sampled ARC timesteps with a detected
fingertip–object contact (shown as a percentage). It measures contact continuity,
not grip strength or absence of slip. Friction sweeps test that sensitivity.

Labels describe a contact executed with the fixed controller and zero initial
finger pitch. Orientation search is retired. Unexpected collisions still fail
a rollout; only adapter–object collision is excluded. Each collision records
its first time and controller phase.

Object-level splits keep candidate siblings together. Geometry-only features
exclude outcome labels and ground-truth physical properties. The saved sweeps
are small and use different physical settings across objects, so they do not
establish generalization or a calibrated success probability. The selector
requires fresh held-out objects with mixed outcomes before deployment.

## Scope and future work

- **Pre-contact, physics-free.** The selector runs before the robot touches the
  object, so friction, mass and CoM height are unknown to it. It never takes
  them as inputs. The friction, mass and force scenarios enter only through the
  robust AND label, as a prior range of physics a chosen contact should survive.
  Friction is per-object and is an *output* of the estimator: separating it from
  the other parameters needs both pushing and tipping.
- **2D CoM is given.** In the full system, the horizontal CoM would be recovered
  in an object frame during a planar-pushing phase. That phase is outside this
  paper, so the 2D CoM is provided as an input to the whole stack, and features
  may use it.
- **Future work: real2sim and active learning.** Use the outcome of each real
  interaction (the estimated parameters and whether the object tipped) to update
  the sim and the selector, instead of training once on a fixed simulated
  envelope.
