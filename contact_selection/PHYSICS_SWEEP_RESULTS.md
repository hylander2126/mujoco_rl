# Box friction sweeps: corrected full rerun

Updated 2026-09-30. Each setting tests 25 regenerated contacts with COM Y=0 and
adapter–object collisions disabled. Controller, thresholds, and finger friction
2.0 are held fixed. Candidate geometry is identical across the three current
friction settings.

| Table friction | Feasible contacts | Center | Reference |
|---:|---:|---|---|
| 0.50 | 25 / 25 | pass | pass |
| 0.20 | 20 / 25 | pass | pass |
| 0.15 | 10 / 25 | fail | pass |

The aggregate counts match the earlier sweeps, but failure causes do not.
At 0.20, all five far-X contacts (world X=0.624 m) fail from insufficient tip,
pivot drift, force/joint limits, and controller failure; two also exceed the
off-axis limit. **There are no unintended collisions in these reruns.**
At 0.15, all contacts at X≥0.573714 m fail; the near-pivot reference passes.

![Table friction 0.20](../outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p20/box_trial_01/contacts.png)

![Table friction 0.15](../outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p15/box_trial_01/contacts.png)

## Watch the current outcomes

- [Far point at friction 0.20: fails](../outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p20/box_trial_01/candidate_002_no_adapter_collision.mp4)
- [Center at friction 0.15: fails](../outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p15/box_trial_01/candidate_001_no_adapter_collision.mp4)
- [Near-pivot reference at friction 0.15: passes](../outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p15/box_trial_01/candidate_000_no_adapter_collision.mp4)

```bash
.venv/bin/python -m contact_selection replay outputs/contact_selection/suites/2026-09-30_static_wrist/box_mu_0p15 --candidate 1
```

## Rerun sensitivity checks

All 16 previously reported boundary trials were rerun: contacts 0, 1, 2, and 4
at friction 0.14 and 0.16, and at friction 0.15 with mass/inertia scaled to
0.9 and 1.1. Contacts 0 and 4 pass in every condition; 1 and 2 fail.
The result is still a local, deterministic boundary check, not repeatability
statistics. Records: `outputs/contact_selection/suites/2026-09-30_static_wrist/boundary_probe.json`.

The nine smaller checks are now saved as complete, replayable runs too:
at table friction 0.1, reference/center/far corner all fail; at 0.17, reference
and center pass while the far corner fails. With finger friction 0.2 and table
friction 0.5, all three fail. Records:
`outputs/contact_selection/suites/2026-09-30_static_wrist/small_friction_probe.json`.

Each probe record points to a dataset under `outputs/contact_selection/suites/2026-09-30_static_wrist/probes`,
with its own model, reset state, config, traces, and contact plot. The main
configs remain [0.20](config/box_mu_0p20.json) and [0.15](config/box_mu_0p15.json).

Friction sensitivity remains important. A geometry-only predictor needs an
explicit physical envelope; success is not an intrinsic label of the point.
These tests do not calibrate friction or grip to hardware.
