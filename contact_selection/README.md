# Contact selection

This package samples reachable top contacts, runs the existing press-pull
controller from the same saved reset state at each contact, and records
feasibility plus physical diagnostics. It also contains an exploratory
geometry-only logistic selector. Its scores are uncalibrated; there is no
validated estimator-quality ranking.

Run these commands from the repository root after installing `requirements.txt`
in `.venv`. Video recording also requires `ffmpeg` on PATH. Saved outputs are
local artifacts; on a fresh checkout, generate a sweep or run the full suite
before replaying it. See [available configs](config/README.md).

## Watch a contact

Record one contact from a saved run as a normal-speed MP4:

```bash
.venv/bin/python -m contact_selection replay outputs/contact_selection/box_mu_0p20 --candidate 2
```

Open `outputs/contact_selection/box_mu_0p20/box_trial_01/candidate_002_no_adapter_collision.mp4`.
The command prints the video path and outcome. Contact numbers are the numbers
on the contact plot. Add `--scene heart_trial_01` for a multi-object run, or
`--show-viewer` to also watch live. `--output clip.mp4` chooses the video path.
You can also pass `contact_selection/config/box_mu_0p20.json` to find its saved
run under `outputs/contact_selection/box_mu_0p20`.

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
.venv/bin/python -m contact_selection generate \
  --output outputs/contact_selection/my_box_sweep --candidates 25
.venv/bin/python \
  -m contact_selection plot outputs/contact_selection/my_box_sweep
```

Then use `.venv/bin/python -m contact_selection replay outputs/contact_selection/my_box_sweep
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
All seven active configs have now been rerun. Earlier snapshots are archived
under `outputs/contact_selection/archive_pre_centering_20260930`. `simulation_preset.parameters.object_y_m` sets the box center explicitly.

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

To rerun the complete active suite into a fresh folder, including probes,
selector evaluation, plots, and representative videos:

```bash
.venv/bin/python -m contact_selection rerun --output outputs/contact_selection/my_full_rerun --workers 8
```

The current selector is `outputs/contact_selection/geometry_selector_centered`.
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
