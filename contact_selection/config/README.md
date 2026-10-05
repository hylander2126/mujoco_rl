# Contact-selection configurations

Use `python -m contact_selection generate --config PATH --output DIRECTORY`.
`python -m contact_selection rerun --output DIRECTORY` runs all 19 active
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

Friction coverage differs between objects. The box goes down to 0.15, the L to
0.25, and the rest are tested only at 0.50, so their robust labels are looser.
When every object is tested down to 0.15, the heart and flashlight have no
robust contact
([details](../MASS_FORCE_RESULTS.md)).

### Unknown mass and force-ramp variants (added 2026-10-05)

Each copies the nominal-friction config for the same objects (`*_mu_0p50`) and
pins one existing `randomization` scale to a single value. Candidate geometry is
unchanged, so `load_robust_contacts` aligns these with the friction variants and
the selector's robust label is the AND over **all** supplied scenarios.

| Config pattern | Objects | Varied | Value |
|---|---|---|---|
| `{box,heart_l,flashlight,monitor_soda}_mass_x0p5.json` | as named | payload mass and inertia | ×0.5 |
| `{box,heart_l,flashlight,monitor_soda}_mass_x2p0.json` | as named | payload mass and inertia | ×2.0 |
| `{box,heart_l,flashlight,monitor_soda}_force_x2p6.json` | as named | press force reference | ×2.6 (5 N → 13 N) |

Why these values:

- **Mass ×0.5 / ×2.0.** The selector runs before the object has been tipped,
  so mass is known only to a prior. A factor-of-two band either side of the
  modeled mass is about what a shape/category guess can promise for household
  objects. A wider band mostly tests whether 5 N is enough force, which is the
  adaptive controller's job, not the contact selector's.
- **Force ×2.6.** `PressPullConfig.force_ref_max_n` is 13 N, the adaptive
  ladder's declared ceiling (5 N × 1.25ⁿ, capped), and still below the 15 N hard
  limit. The nominal configs already cover the 5 N floor. A contact that fails
  at 13 N (slip, adapter collision, joint limit, overshoot past the hard limit)
  cannot be rescued by ramping, so it should not be committed to. The simulated
  FSM declares the ladder fields but does not run the ladder. These are
  single-attempt rollouts at the two ends of its range, not a simulated ramp.
- **One factor at a time, at nominal friction.** This is a union of one-axis
  sweeps, not a grid: no scenario combines low friction with high mass.
  Quasi-statically, scaling mass by k behaves like scaling press force by 1/k
  (the radial press ratio mg/N is what enters the force balance). At low
  friction, force ×2.6 and mass ×0.5 gave identical labels, and at friction
  0.5 none of these variants changed a robust label. See
  [MASS_FORCE_RESULTS.md](../MASS_FORCE_RESULTS.md).

The ×2.0 mass sets monitor to 10 kg and soda to 1 kg. The monitor already has
no feasible contact at nominal mass, so its robust label cannot change.

All active experiments center the initial object COM in world Y and exclude
adapter–object collisions. Other contact pairs retain their collision policy.
Feasibility thresholds and physical presets are explicit in each config.

`experiment.json` contains only legacy feasibility thresholds needed by the
shared `parameter_estimation` demo. It is a compatibility file, not a runnable
contact-selection experiment. Superseded pilot configs have been removed.
