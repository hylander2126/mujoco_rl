# Mesh contact sweeps: corrected full rerun

Updated 2026-09-30. All five mesh geometries were rerun with their **initial COM
at world Y=0** and adapter–object collisions disabled. X/Z placements and pivot
overrides, force/controller settings, split assignments, and thresholds remain
as configured. Centering changes the robot-relative pose; it does not make an
asymmetric mesh geometrically symmetric.

| Object | Object/table friction | Feasible | Center |
|---|---:|---:|---|
| Heart | 0.50 | 12/12 | pass |
| L | 0.50 | 12/12 | pass |
| L | 0.25 | 8/12 | pass |
| Flashlight | 0.50 | 5/5 | pass |
| Monitor | 0.50 | 0/4 | fail |
| Soda | 0.50 | 2/2 | pass |

The **heart center now passes**; the old collision-based negative is gone.
The L's four failures at friction 0.25 remain pivot-drift failures. The monitor
still loses contact and fails to tip. There are no unintended-collision labels
in these current main sweeps. Candidate generation still yields fewer reachable
points for the narrow flashlight/soda surfaces and the monitor.

![Heart](../outputs/contact_selection/suites/2026-09-30_static_wrist/heart_l_mu_0p50/heart_trial_01/contacts.png)

![L at friction 0.25](../outputs/contact_selection/suites/2026-09-30_static_wrist/l_mu_0p25/L_trial_01/contacts.png)

![Flashlight](../outputs/contact_selection/suites/2026-09-30_static_wrist/flashlight_mu_0p50/flashlight_trial_01/contacts.png)

![Monitor](../outputs/contact_selection/suites/2026-09-30_static_wrist/monitor_soda_mu_0p50/monitor_trial_01/contacts.png)

![Soda](../outputs/contact_selection/suites/2026-09-30_static_wrist/monitor_soda_mu_0p50/soda_trial_01/contacts.png)

## Watch

- [Heart center: now passes](../outputs/contact_selection/suites/2026-09-30_static_wrist/heart_l_mu_0p50/heart_trial_01/candidate_000_no_adapter_collision.mp4)
- [Flashlight center: passes](../outputs/contact_selection/suites/2026-09-30_static_wrist/flashlight_mu_0p50/flashlight_trial_01/candidate_000_no_adapter_collision.mp4)
- [Monitor center: fails](../outputs/contact_selection/suites/2026-09-30_static_wrist/monitor_soda_mu_0p50/monitor_trial_01/candidate_000_no_adapter_collision.mp4)
- [Soda contact 1: passes](../outputs/contact_selection/suites/2026-09-30_static_wrist/monitor_soda_mu_0p50/soda_trial_01/candidate_001_no_adapter_collision.mp4)

```bash
.venv/bin/python -m contact_selection replay outputs/contact_selection/suites/2026-09-30_static_wrist/heart_l_mu_0p50 --scene heart_trial_01 --candidate 0
```

## Nearby friction checks

All eight mesh robustness trials were rerun. Heart contacts 0 and 1 now pass
at both 0.45 and 0.55. For L, center success persists at 0.23 and 0.27, while
contact 1 fails at 0.23 and passes at 0.27. See
`outputs/contact_selection/suites/2026-09-30_static_wrist/mesh_robustness_probe.json`; each entry points to a
complete replayable probe dataset.

The active configs are [heart/L](config/heart_l_mu_0p50.json),
[L at 0.25](config/l_mu_0p25.json), [flashlight](config/flashlight_mu_0p50.json),
and [monitor/soda](config/monitor_soda_mu_0p50.json). `center_com_y: true` applies
the COM correction after any requested pose override and MuJoCo constant
recomputation. Scene manifests record both requested and resolved positions.

These outcomes are still a small development dataset with different physical
envelopes across objects. The test geometries do not contain mixed labels, and
these reruns do not establish geometry-general performance.
