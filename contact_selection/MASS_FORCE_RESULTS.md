# Pivot ray-angle feature, mass and force sweeps

Completed 2026-10-05. This adds one interaction feature to the geometry-only selector,
two robustness sweeps (unknown payload mass and the force-ramp ceiling), and a
`compare` command that scores every selector variant against every label scope.
The model is still the same L2 logistic regression. Its fitting rule, the 0.5
threshold and the abstention rule were not changed.

**Short version.**

- The new feature is physically right. Where friction is the binding limit it
  predicts which contacts fail on the box, flashlight and soda, approximately on
  the L, and not on the heart.
- The requested mass and force sweeps **changed no robust label**, so the retrained
  selector is identical to one trained on friction only.
- The selector's real weakness turned out to be elsewhere: objects were labeled
  under different friction envelopes. Under a matched envelope, both existing
  baselines pick a failing heart contact, and the current one also picks a
  failing flashlight contact. The new feature avoids one of those
  failures but costs a false abstention under the old labels.
- Two diagnostics that were not in the original plan are what show this: low
  friction crossed with mass, and the mesh objects at low friction.

## 1. Feature: `pivot_ray_angle_rad`

The hardware press selector (`irb120_perception/contact_point_selector.py`,
`_press_band`) scores `dz + w * (cross(r, -z) . axis)`. With its press pivot on
the −X support edge, `axis = −Y` and that term is `−dx`. The score is
`dz − w·dx`: height above the pivot minus inboard distance. Here
`pivot_dz_m` and `pivot_dx_m` are already features, so that linear score is in
the span a logistic model can already express, and adding it would change nothing.

What a linear model cannot express is their ratio. The FSM regulates force
**radially toward the fixed pivot** and moves tangentially (see
`press_pull_fsm._arc_step`). The press therefore has no moment about the pivot,
but it pushes the object along the pivot→contact ray. `tan(angle) = dx/dz` is
the horizontal/vertical force ratio the press puts onto the table under the
pivot before any pull. When the object weighs little compared with the press, the pivot slides
once `tan(angle) > μ_table`. The hardware score with `w = 1` is positive exactly when
`angle < 45°`. The ratio is what transfers a threshold learned on the 30 cm box
to the 15 cm L.

**Sign check against labels.** Every saved low-friction failure is
`unintended_pivot_or_sliding`. Finger friction is 2.0, so the fingertip never
slips. Contacts sorted by angle at matched friction (`+` pass):

| Object | Angles (°) | μ 0.20 (atan 11.3°) | μ 0.15 (atan 8.5°) |
|---|---|---|---|
| Box | 1×5, 5×3, 6×2, 8×2, 9×2, 12×2, 14×4, 17×5 | 18/25, cut at 14° | 10/25, cut 6°→8° |
| L | 6 6 10 10 13 17 17 20 23 23 26 29 | `++++-++-----` | `++--+++-----` |
| Flashlight | 8 8 8 9 9 | 5/5 | **0/5** |
| Soda | 9 11 | 2/2 | 1/2 (9° passes, 11° fails) |
| Heart | 4 4 7 7 10 13 13 15 18 18 21 23 | 4/12, cut at 10° | **0/12** |

Box, flashlight and soda cut within one candidate spacing of `atan(μ)`. At
μ 0.25 the L cuts at 17° against 14°, and its pattern is non-monotone at lower
friction. The heart slides even at 4°. The angle ignores the pull needed to lift
the COM: the pull's horizontal component grows with `k = cx/|r|`, the COM's
horizontal lever over the ray length. The heart has k = 0.27, against 0.16 for
the box. A k feature needs the horizontal COM, which this selector does not
have before the first tip (follow-up 3).

`contact_selection/tests/test_box_grip.py::test_ray_angle_orders_reference_before_failing_center`
pins the box case. The demo reference contact sits at 1.1°, below atan 0.15. The top
center sits at 9.5°, between atan 0.15 and atan 0.20, matching its fail-at-0.15,
pass-at-0.20 labels.

**Rejected first version.** The first implementation measured the angle between
the ray and the *surface normal* (`press_tilt_rad`). That is the fingertip-slip
angle. It is identical on flat tops, but it scored the curved-top flashlight
lower: held-out flashlight brier 0.194 → 0.369, top score 0.504. Since the
fingertip never slips in simulation, the table-slip angle is the right one.
The switch was made after seeing that validation-object result, which counts
as a model-selection decision on validation data. Monitor and soda (test) were
not used to choose. The run is kept at
`suites/2026-10-05_mass_force/selector_comparison.json`.

## 2. Mass and force sweeps

Twelve new active configs pin the existing `randomization` scales on each
nominal-friction config: mass ×0.5, mass ×2.0, and press force ×2.6
(5 N → 13 N, the `force_ref_max_n` ceiling). Ranges and rationale are in
[config/README.md](config/README.md). The simulated FSM declares, but does not
run, the adaptive ladder, so 13 N is a single attempt at the ladder's ceiling.

All 19 active configs pass (+) for every contact except:

| Scenario | Box | Heart | L | Flashlight | Soda | Monitor |
|---|---:|---:|---:|---:|---:|---:|
| Nominal μ 0.50 | 25/25 | 12/12 | 12/12 | 5/5 | 2/2 | 0/4 |
| Mass ×0.5 | 25/25 | 12/12 | **11/12** | 5/5 | 2/2 | 0/4 |
| Mass ×2.0 | 25/25 | 12/12 | 12/12 | 5/5 | 2/2 | 0/4 |
| Force ×2.6 | 25/25 | 12/12 | 12/12 | 5/5 | 2/2 | 0/4 |

The one new failure is L contact 5, at 29°. It is the only contact past
atan 0.5 = 26.6°, and lighter objects approach exactly that limit. That contact
already fails at L μ 0.25. **No robust label changed**, so the AND over
friction, mass and force gives the same targets, weights and selections as
friction alone.

Costs of the 13 N press that the labels do not show:

- Passing contacts end within **1.05–1.9 N** of the 15 N hard limit, against
  8.9–9.8 N of margin at 5 N.
- The ARC exits much earlier. Median peak tip angle falls from 22.5° to 6.8° on
  the L, 20.0° to 5.4° on the heart, and 10.2° to 5.6° on the flashlight. The
  tip-angle threshold is still met, but a sweep this short may not reach the
  balance angle the estimator needs.
- On the monitor, 2.5 kg at 5 N and 5 kg at 13 N exceed the force limit
  (64–76 N peak contact force). The monitor has no feasible contact anyway.

## 3. Diagnostic: mass and force at low friction

The requested sweeps hold friction at 0.5, where every contact has margin.
Nine extra datasets crossed the thin-margin frictions with the same three
scales. Each is its friction config with `randomization.<scale>` pinned, saved in
`outputs/contact_selection/probes/2026-10-05_mass_force_low_friction/configs/`.

| Scenario | Nominal | Mass ×0.5 | Mass ×2.0 | Force ×2.6 |
|---|---:|---:|---:|---:|
| Box μ 0.20 | 18/25 | 16 | **21** | 16 |
| Box μ 0.15 | 10/25 | 10 | **6** | 10 |
| L μ 0.25 | 6/12 | 8 | **4** | 8 |

- **Mass matters only when friction is tight, and its sign flips.** In the
  quasi-static balance, extra weight helps when `k(cos θ + μ sin θ) < μ` and
  hurts otherwise. For the box, k = 0.16 lies between 0.15 and 0.20: heavier
  hurts at 0.15 and helps at 0.20, as observed. At 0.15, double mass also breaks
  the best 1° and 5° contacts through off-axis rotation. The L
  (k = 0.22–0.25 at μ 0.25) is within about 5% of the boundary. The model
  predicts a marginal help, but heavier hurt. Treat that case as unexplained.
- **Force ×2.6 and mass ×0.5 give identical label vectors in all three cases.**
  Only the ratio `mg/N` enters the quasi-static force balance, so a force-ceiling
  sweep largely duplicates a light-mass sweep.
- Adding these to the AND lowers the robust counts to box 6/25 and L 4/12.

## 4. Diagnostic: the friction envelope differs by object

Robust labels were never comparable across objects. The box is labeled over μ
{0.15, 0.20, 0.50}, the L over {0.25, 0.50}, and everything else only at 0.50.
The selector partly reconciles this through object-identity features: width,
depth and local z carry the largest weights (|w| ≈ 0.64). Six extra datasets ran
heart/L, flashlight and monitor/soda at μ 0.20 and 0.15, with table and object
friction set together as in `l_mu_0p25.json`. They are saved in
`outputs/contact_selection/probes/2026-10-05_matched_friction/configs/`. Under the matched envelope plus the
interaction scenarios, the robust labels become:

| Object | Split | Friction only | + mass, force | + interaction | + matched envelope |
|---|---|---:|---:|---:|---:|
| Box | train | 10/25 | 10/25 | 6/25 | 6/25 |
| Heart | train | 12/12 | 12/12 | 12/12 | **0/12** |
| L | validation | 6/12 | 6/12 | 4/12 | 2/12 |
| Flashlight | validation | 5/5 | 5/5 | 5/5 | **0/5** |
| Monitor | test | 0/4 | 0/4 | 0/4 | 0/4 |
| Soda | test | 2/2 | 2/2 | 2/2 | 1/2 |

Whether μ 0.15 is a realistic table friction is a hardware question. This
diagnostic only shows that the current per-object labels answer different
questions.

## 5. Selector comparison

From `.venv/bin/python -m contact_selection compare` (Section 6). Each held-out
cell reads selected-pass / selected-fail / abstained (of which false), followed by
mean brier over the four held-out objects (L, flashlight, monitor, soda). Lower brier
is better. Every model below was trained on its stated scope and evaluated under
each label scope.

| Model | Original friction labels | + mass, force (requested) | + interaction | Matched envelope (strictest) |
|---|---|---|---|---|
| `geometry_selector_centered` (documented baseline, older rotating-wrist controller) | 2/0/2 (1), 0.381 | same | 2/0/2 (1), 0.387 | 2/0/2 (0), 0.454 |
| `suites/2026-09-30_static_wrist/geometry_selector` (same code as this run) | 3/0/1 (0), 0.358 | same | 3/0/1 (0), 0.372 | **2/1/1 (0), 0.502** |
| 15 features, trained on + mass, force | 3/0/1 (0), 0.358 | same | 3/0/1 (0), 0.372 | 2/1/1 (0), 0.502 |
| **Ray angle, trained on + mass, force (new)** | **2/0/2 (1), 0.386** | same | 2/0/2 (1), 0.375 | **2/0/2 (0), 0.408** |
| Ray angle, trained on matched envelope | 2/0/2 (1), 0.271 | same | 2/0/2 (1), 0.287 | 1/1/2 (1), 0.342 |

The documented baseline was trained on the older rotating-wrist run, so its row
shows its saved weights scored on these datasets. The static-hardware model, the
"15 features" row, and a 15-feature model trained here on friction only have
identical weights, because the code and labels are identical.

**What the new feature changed, held-out:**

- L: same pick (contact 11, which passes under every scope). Brier improves
  0.212 → 0.159 under the original labels, and 0.351 → 0.128 under the strictest.
- Flashlight: **worse under the original labels.** Every contact passes, but no
  score reaches 0.5, so the selector abstains falsely (brier 0.194 → 0.361; the
  centered baseline had the same false abstention). **Better under the matched
  envelope:** all five contacts fail at μ 0.15, so abstaining is correct. The
  15-feature model commits to contact 4, which fails.
- Soda and monitor: unchanged. Soda picks contact 1, which passes under every
  scope. Monitor still abstains as geometry out of range.

**Leave-one-object-out** refits once for each of the six objects (train/validation/test
split ignored) and scores the held-out object under the strictest labels.
With the ray angle, brier goes 0.551 → 0.577 (**worse**). The pass/fail/abstain
count is the same (3/2/1): both feature sets still pick a failing heart and a
failing flashlight contact when those objects are held out of a fit whose
labels never saw them at low friction.

**Training on the matched envelope does not fix this either.** With only two
training geometries, the heart's 0/12 makes heart-like geometry negative, and the flashlight
then looks box-like. That model picks a failing flashlight contact at score 0.81
and abstains on soda, which still has one robust contact.

**Net.** Under the labels the selector was built for, the feature is a wash on
choices, slightly worse on brier, and adds a false abstention. It is the only
variant that makes no failing choice under a physically consistent envelope.
That case was reached by a diagnostic chosen after looking at the flashlight,
so treat it as one favorable case, not as evidence of generalization. The
dominant problem is the training set: two training geometries, and before the
envelope diagnostic, no heart contact with a negative label.

## 6. Reproduce

The active suite (19 configs, datasets, probes, selector, plots, 29 replay
videos, demos) was produced as below. Each output path must not exist yet. Its datasets were generated while `features.py` still held the
rejected normal-based feature, so the features were recomputed, without
re-simulation, into a label-only copy:

```bash
S=outputs/contact_selection/suites/2026-10-05_mass_force
.venv/bin/python -m contact_selection rerun --output $S --workers 8
.venv/bin/python -m contact_selection refeature $S/{box,heart_l,flashlight,monitor_soda}_mu_0p50 \
  $S/box_mu_0p{20,15} $S/l_mu_0p25 $S/*_mass_x* $S/*_force_x* --output $S/features_ray
.venv/bin/python -m contact_selection train $S/features_ray/*/ --output $S/geometry_selector_ray
.venv/bin/python -m contact_selection compare $S/features_ray \
  --extra outputs/contact_selection/probes/2026-10-05_mass_force_low_friction outputs/contact_selection/probes/2026-10-05_matched_friction \
  --baseline outputs/contact_selection/suites/2026-09-30_static_wrist/geometry_selector/model.json \
  --output $S/features_ray/selector_comparison.json
```

A fresh `rerun` now writes ray-angle features directly. `refeature` is only
needed for datasets generated before a feature change. The interaction and envelope
datasets were generated with the ray-angle feature, by `generate --config` on
the saved configs. Their trajectory `.npz` files were deleted after labeling
because the disk was full (2026-10-05: about 5 GB free of 228 GB, with an active
suite taking about 9 GB). They cannot be replayed from saved trajectories, but
`replay` re-simulates from each scene's `model.mjb` and reset state.

The new selector is `suites/2026-10-05_mass_force/geometry_selector_ray`. The
`geometry_selector` beside it was fitted inside `rerun` with the rejected
normal-based feature and is superseded. The 15-feature ablation is
`train ... --exclude-features pivot_ray_angle_rad`. Older 15-feature
`model.json` files still load and score.

The friction datasets reproduce `suites/2026-09-30_static_wrist` exactly
(candidates, labels and metrics), as do all 33 probe outcomes, so the comparison
above isolates the feature and label changes.

## 7. Follow-up

1. **Pick one friction envelope for every object from hardware.** Measure the
   real table/object friction range and run every object across it. This changes labels
   far more than mass does, and decides whether the low-friction failures above
   matter at all.
2. **Cross mass with that envelope, not with μ 0.5.** Mass only changed labels
   where friction was tight. Mass ×0.5 can stand in for the force ceiling there,
   per the identical labels in §3.
3. **COM lever feature `k = cx/|r|`.** It explains the heart's early sliding and
   sets the sign of the mass effect, but needs the horizontal COM. If the first
   tip's estimate, or a vision prior, can supply it, it is the next feature to try.
4. **Simulate the adaptive ladder,** rather than a single attempt at its ceiling,
   and check whether the 13 N run's short ARC sweeps still reach the balance
   angle the estimator needs.
5. **More training geometries with mixed labels.** With two training objects,
   object-identity features carry the largest weights, and no feature change
   can fix that.
