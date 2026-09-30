# Contact-selection configurations

Use `python -m contact_selection generate --config PATH --output DIRECTORY`.
`python -m contact_selection rerun --output DIRECTORY` runs all seven active
configs, then creates plots, selector results, sensitivity probes, and videos.

| Config | Objects | Table sliding friction | Current accepted contacts |
|---|---|---|---|
| `box_mu_0p50.json` | Box | 0.50 | 25 |
| `box_mu_0p20.json` | Box | 0.20 | 25 |
| `box_mu_0p15.json` | Box | 0.15 | 25 |
| `heart_l_mu_0p50.json` | Heart, L | 0.50 | 12 |
| `l_mu_0p25.json` | L | 0.25 | 12 |
| `flashlight_mu_0p50.json` | Flashlight | 0.50 | 5 |
| `monitor_soda_mu_0p50.json` | Monitor, soda | 0.50 | 4, 2 |

All active experiments center the initial object COM in world Y and exclude
adapter–object collisions. Other contact pairs retain their collision policy.
Feasibility thresholds and physical presets are explicit in each config.

`experiment.json` contains only legacy feasibility thresholds needed by the
shared `parameter_estimation` demo. It is a compatibility file, not a runnable
contact-selection experiment. Superseded pilot configs have been removed.
