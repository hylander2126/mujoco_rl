# Exploratory geometry-only contact selector

> **2026-10-05:** A 16th feature, `pivot_ray_angle_rad`, mass and force-ceiling
> scenarios in the robust AND, and a matched-friction diagnostic are reported in
> [MASS_FORCE_RESULTS.md](MASS_FORCE_RESULTS.md). The 15-feature description
> below remains accurate for `geometry_selector_centered`.

Updated 2026-09-30: retrained from all seven corrected datasets, with COM Y=0
and adapter–object collisions disabled. No threshold or fitting rule was changed.

The first learned baseline is L2-regularized logistic regression implemented
with NumPy and SciPy. It uses the 15 pre-action geometry/reachability features
already saved with each candidate. Candidates from the same object and physical
setting are aligned by index and exact candidate geometry. A candidate's
**robust label** is positive only if it passed every supplied physical setting
for that object. This avoids assigning contradictory targets to the same
geometry, but the physical envelopes differ across objects and have not been
chosen from a calibrated operating distribution. Scores are uncalibrated.

The fit uses box and heart as training geometries. L and flashlight are
validation geometries; monitor and soda are test geometries. No contacts from
a held-out object enter the fit. Training weights each object equally, so the
25 box contacts do not dominate the 12 heart contacts. The model standardizes
features with physical scale floors to avoid huge extrapolation from nearly
constant training dimensions. A candidate is eligible only if its score is at
least 0.5 and every standardized feature is within 10 scale units of the
training mean. The selection API returns the top eligible candidate or an
explicit abstention reason.

| Split | Object | Robust contacts | Center | Selected contact | Outcome |
|---|---|---:|---|---:|---|
| Train | Box | 10/25 | fails | 18 | passes |
| Train | Heart | 12/12 | passes | 7 | passes |
| Validation | L | 8/12 | passes | 11 | passes |
| Validation | Flashlight | 5/5 | passes | abstain | all scores below 0.5 |
| Test | Monitor | 0/4 | fails | abstain | geometry out of range |
| Test | Soda | 2/2 | passes | 1 | passes |

The rerun changes the heart selection from 5 to 7. It also introduces a **false
abstention on the flashlight**: all contacts are feasible, but the largest score
is about 0.497, below the unchanged 0.5 threshold. The threshold was not retuned
to hide this result. This shows why unchanged aggregate success counts do not
imply unchanged selection behavior. Candidate indices can also change when the
candidate set is regenerated; use coordinates when comparing older runs.

Retrain from the saved datasets. Without `--output`, the model is written to
`geometry_selector_YYYY-MM-DD/` beside the first dataset:

```bash
.venv/bin/python -m contact_selection train \
  outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p50 \
  outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p20 \
  outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p15 \
  outputs/contact_selection/suites/2026-09-30_static_wrist/heart_l_mu_0p50 \
  outputs/contact_selection/suites/2026-09-30_static_wrist/l_mu_0p25 \
  outputs/contact_selection/suites/2026-09-30_static_wrist/flashlight_mu_0p50 \
  outputs/contact_selection/suites/2026-09-30_static_wrist/monitor_soda_mu_0p50
```

For pre-action selection, load `model.json`, generate candidates and geometry
with `generate_candidates()`, then call
`contact_selection.selector.predict_and_select(model, candidates, geometry)`.
It returns the selected candidate, a score and feature-range diagnostic for
each proposal, or `no_valid_candidates`, `geometry_out_of_range`, or
`low_score`. A CLI can apply the same API to any saved pre-action scene manifest:

```bash
.venv/bin/python -m contact_selection select \
  outputs/contact_selection/suites/2026-09-30_static_wrist/geometry_selector/model.json \
  outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p15/box_trial_01/scene.json
```

The API does not read rollout labels or simulator ground-truth mass/friction.

## Limits of this result

This is a development baseline, not a validated predictor. A first version
using raw training standard deviations saturated to scores of 0 or 1 on held-out
shapes and accepted the failed monitor. The physical scale floors fixed the
numerical extrapolation; the geometry-range abstention rule was added after
inspecting that first test run. The monitor outcome is therefore **not an
untouched test result**. The training set has only two geometries, and the
test scenes have no mixed labels (all monitor contacts fail; both soda contacts
pass). The selected-contact table shows pipeline behavior, not statistical
evidence that the model generalizes or beats a geometry heuristic on new
objects. More object geometries with mixed outcomes, a prespecified physics
distribution, and a fresh held-out test are required before calibrating scores
or using them as success probabilities. The MLP and secondary estimator-quality
ranking remain unimplemented.

## Relationship to RL

The current method is **simulation-supervised contact selection**: sample surface
points, execute a fixed press–pull controller, label outcomes, and fit a classifier.
The sampler is not a learned exploration policy and there is no policy-gradient
or value-learning update. A learned point-selection policy receiving one reward
per attempt would be a contextual bandit (a one-step RL formulation). Learning
force, motion, or orientation choices throughout the interaction would instead
be sequential RL. Neither RL variant is implemented in this baseline.
