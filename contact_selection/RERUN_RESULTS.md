# Complete rerun after centering and adapter exclusion

Completed 2026-09-30. All seven active configs now place the initial object COM
at world Y=0 and disable only adapter–object collision pairs. The press–pull
controller, friction settings, thresholds, train/validation/test assignments,
and selector fitting/abstention rules were retained.

Recomputed: **122 main rollouts across nine scenes**, **33 sensitivity trials**,
the selector fit and evaluation, all corresponding plots, **13 main replay
videos**, and the standalone demo at 1 ms and 0.5 ms timesteps.

## Main results

| Scene | Previous passes | Current passes |
|---|---:|---:|
| Box, table friction 0.50 | 25/25 | 25/25 |
| Box, table friction 0.20 | 20/25 | 20/25 |
| Box, table friction 0.15 | 10/25 | 10/25 |
| Heart, friction 0.50 | 11/12 | 12/12 |
| L, friction 0.50 | 12/12 | 12/12 |
| L, friction 0.25 | 8/12 | 8/12 |
| Flashlight | 5/5 | 5/5 |
| Monitor | 0/4 | 0/4 |
| Soda | 2/2 | 2/2 |

The heart center now passes. Current box failures are physical motion/execution
failures, not adapter-collision negatives. No main rollout reports an
unintended collision. The nominal box maximum off-axis rotation falls from
0.270° to 0.118°.

Candidate sets were regenerated, not relabeled. Lateral recentering can change
IK margins and farthest-point ordering. Comparing by **body-local contact
coordinates**, 113/122 current main records match older sampled positions;
the matched heart center changes fail→pass. Nine records represent positions
not sampled in the corresponding older run (two per box setting and three
heart points). Equal pass counts therefore do not imply identical candidates.

## Selection changed

The retrained selector still chooses a passing box point and a passing L point,
but selects heart contact 7 instead of 5. It now **abstains on the flashlight**:
all five contacts pass, but the maximum predicted score is approximately 0.497,
below the unchanged 0.5 threshold. Monitor abstention remains due to geometry
outside the training feature range. Soda selection still passes.

This is a useful limitation of the supervised baseline, not a reason to alter
the threshold after seeing the evaluation. See the
[full selection report](LEARNING_BASELINE_RESULTS.md).

## Watch current outcomes

- [Nominal box reference](../outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p50/box_trial_01/candidate_000_no_adapter_collision.mp4)
- [Low-friction box center: fails](../outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p15/box_trial_01/candidate_001_no_adapter_collision.mp4)
- [Selected box point: passes](../outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p15/box_trial_01/candidate_018_no_adapter_collision.mp4)
- [Heart center: now passes](../outputs/contact_selection/suites/2026-09-30_static_wrist/heart_l_mu_0p50/heart_trial_01/candidate_000_no_adapter_collision.mp4)
- [Selected heart point](../outputs/contact_selection/suites/2026-09-30_static_wrist/heart_l_mu_0p50/heart_trial_01/candidate_007_no_adapter_collision.mp4)
- [Selected L point](../outputs/contact_selection/suites/2026-09-30_static_wrist/l_mu_0p25/L_trial_01/candidate_011_no_adapter_collision.mp4)
- [Flashlight center: feasible despite selector abstention](../outputs/contact_selection/suites/2026-09-30_static_wrist/flashlight_mu_0p50/flashlight_trial_01/candidate_000_no_adapter_collision.mp4)
- [Monitor center: fails](../outputs/contact_selection/suites/2026-09-30_static_wrist/monitor_soda_mu_0p50/monitor_trial_01/candidate_000_no_adapter_collision.mp4)
- [Soda selected point](../outputs/contact_selection/suites/2026-09-30_static_wrist/monitor_soda_mu_0p50/soda_trial_01/candidate_001_no_adapter_collision.mp4)

The 13 replay videos decode successfully, and every replay's metric dictionary
and failure list exactly match its saved main rollout. All scene COM Y values
and collision masks are checked against the corrected policy.

## Sensitivity and retained history

All 16 box boundary checks retain their pass/fail pattern. The eight mesh
checks now show both heart contacts passing at friction 0.45 and 0.55; the L
contact-1 boundary between 0.23 and 0.27 persists. The nine earlier small
friction checks were rerun and are now saved as complete replayable datasets.
See [box physics](PHYSICS_SWEEP_RESULTS.md) and
[mesh results](CROSS_GEOMETRY_RESULTS.md).

The corrected runs, probe snapshots and plots are in `outputs/contact_selection/suites/2026-09-30_static_wrist`.

## RL relationship

The current pipeline uses simulation to generate supervised labels for contact
selection. It does not learn an exploration policy or a sequence of control
actions. Keeping the controller fixed and learning which point to try from a
single resulting reward is a contextual-bandit formulation. Learning actions
throughout press–pull would be sequential RL. The present implementation is
the simulator-data and supervised-baseline foundation adjacent to those methods.
