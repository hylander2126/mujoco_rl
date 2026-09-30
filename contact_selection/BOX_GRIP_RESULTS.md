# Nominal box: corrected full rerun

Updated 2026-09-30. All 25 contacts were regenerated and simulated with the box
COM physically at world Y=0 and adapter–object collisions disabled. Finger
friction is 2.0; table friction is 0.5. The press–pull controller and feasibility
thresholds are unchanged. This remains an uncalibrated, grippy simulation preset.

| Measurement | Result |
|---|---:|
| Feasible contacts | 25 / 25 |
| Intended tip | 10.57–11.22° |
| ARC time touching | 100% for every contact |
| Maximum pivot drift | 0.626 mm |
| Maximum off-axis rotation | 0.118° |
| Maximum fingertip contact force | 5.887 N |
| Near-pivot reference (0) | pass |
| Center contact (1) | pass |

![Current contact outcomes](../outputs/contact_selection/box_mu_0p50/box_trial_01/contacts.png)

The reference is at world (0.536, 0.000, 0.350) m. The box center is at X=0.58 m,
Y=0. Plot X/Y coordinates are translated to the geometry center; the rotation
panel uses contact Y relative to the actual initial COM. Off-axis rotation is
the peak world-X/Z rotation-vector magnitude relative to ARC onset.
[The matched-offset analysis](OFF_AXIS_RESULTS.md) explains why physically
centering the setup produces the expected V shape.

ARC contact percentage measures touching time, including sliding. It does not
establish sticking or hardware grip robustness. Feasibility also checks
rotation, pivot drift, forces, joints, collisions, and controller completion.

## Watch

[Reference contact video](../outputs/contact_selection/box_mu_0p50/box_trial_01/candidate_000_no_adapter_collision.mp4)

```bash
.venv/bin/python -m contact_selection replay outputs/contact_selection/box_mu_0p50 --candidate 0
```

The current dataset is `outputs/contact_selection/box_mu_0p50`; its config is
[box_mu_0p50.json](config/box_mu_0p50.json). All candidates share a saved reset
state. Parallel workers load separate copies of that compiled model and state.
The recorded reference video exactly matches the saved rollout metrics.

This nominal scene still has only positive labels, so it alone cannot show a
selection advantage or train a binary classifier. The friction sweeps supply
the mixed labels used by the current selector.

Earlier results are preserved under
`outputs/contact_selection/archive_pre_centering_20260930/box_mu_0p50`.
