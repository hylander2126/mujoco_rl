# Exploratory geometry-only contact selector

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

| Split | Object | Robust contacts | Centre | Selected contact | Outcome |
|---|---|---:|---|---:|---|
| Train | Box | 10/25 | fails | 18 | passes |
| Train | Heart | 11/12 | fails | 5 | passes |
| Validation | L | 8/12 | passes | 11 | passes |
| Validation | Flashlight | 5/5 | passes | 4 | passes |
| Test | Monitor | 0/4 | fails | abstain | no feasible contact |
| Test | Soda | 2/2 | passes | 1 | passes |

The saved exploratory checkpoint and detailed per-contact scores are under
`outputs/contact_selection/geometry_selector_v3`. Reproduce from the existing
saved datasets in a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_contact_selector.py \
  outputs/contact_selection/box_grip_25 \
  outputs/contact_selection/box_grip_table_02 \
  outputs/contact_selection/box_grip_table_015 \
  outputs/contact_selection/mesh_arc_grip_12 \
  outputs/contact_selection/L_arc_grip_025_12 \
  outputs/contact_selection/flashlight_arc_grip \
  outputs/contact_selection/heldout_pose_arc_grip_v2 \
  --output outputs/contact_selection/my_geometry_selector
```

For pre-action selection, load `model.json`, generate candidates and geometry
with `generate_candidates()`, then call
`contact_selection.selector.predict_and_select(model, candidates, geometry)`.
It returns the selected candidate, a score and feature-range diagnostic for
each proposal, or `no_valid_candidates`, `geometry_out_of_range`, or
`low_score`. A CLI can apply the same API to any saved pre-action scene manifest:

```bash
.venv/bin/python scripts/select_contact.py \
  outputs/contact_selection/geometry_selector_v3/model.json \
  outputs/contact_selection/box_grip_table_015/box_0_481830384/scene.json
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
