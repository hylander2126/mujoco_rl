# Uncertainty analysis (simulation only)

Which input uncertainties matter for the press-pull estimate of (m, z_c, μ), and
how robust contact selection is to reconstructed-geometry error.

```bash
source ~/.virtual_environments/robot_learning/bin/activate
python scripts/run_uncertainty_analysis.py --noise measured  # ~10 min; writes outputs/uncertainty/DATE_analysis/
PYTHONPATH=$PWD python -m uncertainty.hardware_check         # the 40 hardware trials, ~15 s
PYTHONPATH=$PWD python -m uncertainty.contact_resim --output out.json  # real sim labels, ~15 min on 8 cores
PYTHONPATH=$PWD python -m pytest uncertainty/tests           # synthetic sanity checks, ~30 s
```

The first run simulates the validated box press-pull demo once (~75 s) into
`outputs/uncertainty/nominal_box/`. After that, every estimate is a refit of that
noise-free rollout with perturbed inputs, using the canonical
`press_pull_estimator.estimate_press_pull`. That package is imported from
`../press-pull-tipping/code` until it is pip-installed. Nothing is re-simulated.
`--noise` picks the σ preset in `estimator_mc.NOISE_PRESETS`:
`assumed` (the original guesses), `measured` (F/T and pivot levels from the hardware
logs), or `retared` (measured, but with the F/T re-tared just before SQUASH, so only
the drift within a trial remains). Other options: `--samples`, `--contact-configs`, `--selector-model model.json` (use a
trained selector's scores instead of the proxy), `--seed`.

The prompt for checks that need the robot is in `LAB_PC_PROMPT.md`.

## What is perturbed

| Source | Enters the estimator as | Assumed σ | Measured σ |
|---|---|---|---|
| `ft_white` | per-sample sensor wrench noise | 0.05 N, 0.002 N·m | 0.01 N, 0.25 mN·m |
| `ft_bias` | constant wrench offset per trial; also the push slip force (μ) | 0.05 N, 0.002 N·m | 0.035 N, 1 mN·m (re-tared: 0.012 N, 0.4 mN·m) |
| `tilt_white` / `tilt_bias` | ball rotated about the pivot, so only θ changes | 0.1° / 0.2° | same (not measurable) |
| `ball_pos` | FK position noise, ball and sensor together | 0.1 mm | same |
| `tool_offset` | ball→sensor calibration along the tool axis | 1 mm | same |
| `pivot` | the `pivot` argument (camera), x and z | 2 mm | 1.5 mm |
| `com_x` | the known horizontal CoM | 2 mm | same (not measurable) |

The "measured" column comes from `hardware_check.py` (next sections). The rest are
still assumptions; edit `NoiseSpec`. μ is
`(μm)_push / m̂`. The shove itself is not propagated; only the F/T force bias on
the slip force and the mass error are.

Contacts: hull vertices (1 mm), rigid object translation (2 mm) and yaw (1°),
and the pivot (2 mm) are perturbed. The geometry checks from `generate_candidates`
then rerun at the same fixed XY candidates, followed by `extract_features` and the
score.

## Results (2026-10-06, box, assumed σ, 200 draws)

Ground truth m = 0.663 kg, z_c = 0.1378 m, μ = 0.5. The noise-free fit gives
0.668 kg / 0.1427 m (+0.8% / +3.6%) with a near-zero residual. The section after
the contact results traces that bias to two geometric effects
(`bias_breakdown.json`).

| Source alone | std m | std z_c | std μ |
|---|---|---|---|
| ft_bias | 4.8% | 5.7% | 5.0% |
| com_x | 3.9% | 4.0% | 3.8% |
| pivot | 3.0% | 3.7% | 3.0% |
| tool_offset | 1.4% | 1.7% | 1.4% |
| tilt_bias | 1.1% | 1.2% | 1.0% |
| tilt_white, ft_white, ball_pos | ≤0.5% | ≤0.5% | ≤0.5% |
| **all** | **7.1%** | **8.1%** | **7.3%** |

- **Constant errors dominate; white noise averages out** over ~10k samples. The F/T
  term is almost all *force* bias: 0.05 N acting over the ~0.3 m sensor→pivot
  lever arm gives 5.0% on m, while 0.002 N·m of torque bias gives 0.55%.
- **The camera/prior geometry is the next tier.** `com_x` and `pivot` errors of
  2 mm each cost 3–4%. Each scales roughly linearly with σ (`sweeps.png`).
- **The parameters are strongly coupled.** All-source correlations are
  corr(m, z_c) = −0.91 and corr(m, μ) = −0.97; μ inherits m's error through
  μ = μm/m̂.
- **NLS covariance agrees with MC to within 5% for white F/T noise.** The
  condition numbers are 7 (ARC) and 5 (UNARC), with corr(m, z_c) = 0.67. It
  underestimates the all-source spread by about 60× because constant errors are
  invisible to the residual Jacobian. Don't use it as the condition-C confidence
  signal on its own.
- **Short sweeps are badly conditioned.** Fitting only to 4° instead of 15.5°
  raises std z_c 15×, |corr(m, z_c)| from 0.73 to 0.93, and the condition number
  from 7 to 330. This is the quantitative version of the 1° tilt-mask problem
  in `ONLINE_ESTIMATOR.md`.

### Contact robustness (500 draws)

- **Box, μ = 0.2.** The nominal pick (candidate 0, 6 mm inside the pivot edge,
  which is the best friction margin) stays feasible in only 59% of draws. The
  robust pick, candidate 17 at the centreline 25 mm in, holds 100%.
- **L, μ = 0.25.** The nominal pick (6) holds 77%; the robust pick (2) holds 100%.
- **Boundary slack predicts robustness.** Spearman ρ between a candidate's
  distance to the nearest feasibility boundary and its P(feasible) is 0.96 (box)
  and 0.90 (L).
- At 4× the default geometry noise, even the robust pick drops to 0.31 (box) and
  0.14 (L).

Feasibility here is the geometric checks **plus** a proxy, `tan(ray angle) < μ_table`.
That is the failure mechanism documented in `contact_selection/features.py`,
not the simulated label. It is conservative: it passes 14/25 box and 5/12 L
contacts, where the saved sim sweeps passed 20/25 and 8/12. `p_geometry_ok` in
`contacts.csv` drops the proxy. Without `--selector-model`, the score is the
friction margin μ − tan(angle), so the "nominal pick" is a proxy pick too.

## Where the noise-free bias comes from

The sim forces are not the problem. Over the sweep, the F/T reading equals the
fingertip contact force to 3e-4 N. Torques about the CoM balance to 2.5e-4 N·m
against a 0.32 N·m finger torque, so the quasi-static assumption holds too. The bias
is geometric:

| Step | θ* (truth 19.95°) | m | z_c |
|---|---|---|---|
| as estimated | 19.31° | +0.8% | +3.6% |
| tilt corrected for ball rolling | 20.15° | +0.7% | −1.1% |
| + reference pre-tilt (−0.22°) | 19.93° | −0.3% | +0.1% |

1. **The ball rolls; it isn't carried rigidly.** The finger holds its orientation, so
   the ball rolls along the top face as the object tips, and its centre moves `r·θ`
   across the face. The estimator (`object_tilt`) reads tilt as the ball centre's
   rotation about the pivot, which assumes the centre is fixed to the object. It sees
   `φ = θ − atan((x0 + rθ)/H) + atan(x0/H) ≈ θ(1 − rH/(H² + x0²))`. For the box that
   factor is 0.958, matching the measured 0.959. The no-slip check (`|p_ball − pivot|`
   constant) can't see it, because the distance barely changes.
   `estimator_mc.rolling_corrected_tilt` inverts it exactly.
2. **The press pre-tilts the object before the tilt reference is taken.** The
   estimator calls the first LULL pose θ = 0. By then the 5 N press has rocked the box
   −0.22° through contact compliance.

**Rolling only corrupts the tilt.** The fit compares the measured torque about the
pivot, τ(θ), with the gravity model at the same θ. The torque is the measured wrench
moved from the sensor to the pivot with the measured sensor pose (`-Ad_{T_SO}^T w_S`).
That transform is exact. Its tilt-dependent rotation is about the tip axis, and it
doesn't change the torque about that axis. So the moving contact point, which really
does shorten the lever arm, is already in the measurement. The force vector is
measured too. The only wrong input is θ. The model is evaluated at compressed angles,
so the zero crossing θ* comes out about 4% low, and z_c = com_x / tan θ* comes out
high. The bias grows with r/H, so it is largest for short objects.

## Hardware check (40 logged trials, `hardware_check.py`)

**The finger holds its orientation on hardware too.** It rotates only 0.16–0.25° over
ARC/UNARC, so the ball must roll. The predicted tilt slopes are box 0.958, heart 0.939,
flashlight 0.940 and monitor 0.974. The logs can't confirm this directly: the vision
object pose has no samples during contact in any trial. Refitting every trial with
the corrections gives:

| Object | published m / z_c | rolling-corrected m / z_c | + F/T offset removed m / z_c |
|---|---|---|---|
| box | +4.6% / +0.7% | +4.5% / −3.8% | **+2.2% / −0.5%** |
| heart | +1.1% / +15.1% | +1.0% / +7.4% | **+1.3% / +11.1%** |
| flashlight | +2.4% / 0.0% | +2.4% / −6.3% | **−1.6% / −0.7%** |
| monitor | +4.6% / +5.3% | +4.6% / +2.4% | **+4.2% / +2.9%** |

The F/T offset correction subtracts the offset at SQUASH start, plus a linear drift
to the end of RETRACT.

- **With both corrections, mean |error| falls from 3.2% to 2.3% on m and from 5.3%
  to 3.8% on z_c.** The F/T offset alone pushes z_c *up*, while rolling pushes it
  down. The published box and flashlight z_c were accurate partly because the two
  errors cancelled. This is consistent with the rolling model but doesn't prove it.
  An independent tilt measurement is still needed (`LAB_PC_PROMPT.md`, B3).
- **The F/T levels in the logs:**

  | Quantity | Measured |
  |---|---|
  | Offset at SQUASH start, relative to the tare | 0.026–0.081 N, 1.1–2.3 mN·m |
  | Drift within the trial | 0.012–0.023 N, 0.3–0.7 mN·m |
  | White noise, per axis | 0.006–0.011 N |

  The tare happens once at program start, so the offset includes everything since
  then. It is very consistent within each object (± 0.005–0.009 N).
- **The F/T offset doesn't explain the ARC/UNARC gap.** Removing the offset and a
  linear drift leaves it unchanged (monitor m +5.1% → +5.2%, flashlight −3.1% → −3.4%).
  The gap also changes sign between objects. What remains is load-dependent sensor
  hysteresis or a resisting torque that reverses with the sweep direction, such as
  pivot-edge friction or rolling resistance. Both need robot tests.
- **The monitor fails the no-slip check.** Its deviation is 8.4 mm against a 4 mm
  tolerance, and its trajectory pivot sits 32 mm from ARC_CENTER, so its arc isn't a
  clean rotation about the commanded edge. The other objects' fitted pivots are
  1.0–1.7 mm inside ARC_CENTER.

## Measured noise levels (box rollout, `--noise measured`)

| Source alone | assumed σ: std m / z_c | measured σ | re-tared |
|---|---|---|---|
| ft_bias | 4.8% / 5.7% | 3.3% / 4.0% | 1.1% / 1.4% |
| com_x (2 mm) | 3.9% / 4.0% | 3.9% / 4.0% | 3.9% / 4.0% |
| pivot | 3.0% / 3.7% | 2.2% / 2.8% | 2.2% / 2.8% |
| tool_offset | 1.4% / 1.7% | 1.4% / 1.7% | 1.4% / 1.7% |
| tilt_bias | 1.1% / 1.2% | 1.1% / 1.2% | 1.1% / 1.2% |
| **all** | **7.1% / 8.1%** | **5.8% / 6.5%** | **5.0% / 5.4%** |

- **With the measured levels, F/T offset and `com_x` are tied** at 3–4% each.
- **Re-taring just before SQUASH would cut the F/T term to about 1%.** After that,
  `com_x` and the pivot are the remaining big terms, and both come from the camera.

## Confidence vs sweep reached (`confidence.png`)

This analysis puts all sources at the measured σ and truncates the sweep to θ_max. It
covers the sim box and exact synthetic trials with each hardware object's geometry.

| Object | std m | std z_c |
|---|---|---|
| monitor | 4.4% | 4.7% |
| box | 6.0% | 6.6–7.4% |
| heart | 9.3% | 11.6–13% |
| flashlight | 11.3% | 13% |

**The spread is nearly flat from 3° to the full sweep.** Constant errors dominate, and
a longer sweep doesn't average them out. The NLS covariance shrinks 10–30× over the
same range, and it is 100–300× below the real spread.

The objects differ mainly through com_x: the same 2 mm prior error is 7% of the
flashlight's com_x but 3% of the monitor's.

Two consequences for condition C:

- For this error budget, a short sweep isn't what limits the estimate. Once θ passes
  a few degrees, the estimate is about as good as it will get. That supports the
  extrapolation plan in `ONLINE_ESTIMATOR.md`.
- The confidence signal should be a per-object constant floor (from the priors' σ),
  plus the NLS term, which only matters below ~4°.

## Contact robustness: real sim labels (`contact_resim.py`)

Each pick was re-simulated 20 times, with the true object moved by the camera-sized
error: 2 mm translation, 1° yaw, plus 2 mm on the believed arc centre. The robot
pressed at the believed point.

| Pick | Proxy P(feasible) | Sim P(feasible) | Max tilt |
|---|---|---|---|
| box 0 (nominal) | 0.59 | **1.00** | 14–18° |
| box 17 (robust) | 1.00 | **1.00** | 13.5–17.5° |
| L 6 (nominal) | 0.77 | **1.00** | |
| L 2 (robust) | 1.00 | **1.00** | |

Pivot drift stayed at most 2.7 mm against the 10 mm threshold. **At this noise level,
geometry error doesn't make the picks fail; the friction proxy was too pessimistic.**
The proxy's mechanism, `tan(ray angle) < μ`, only bites near the friction limit, and
the real contacts have more margin than it assumes. The ranking by boundary slack is
still right, but the absolute P(feasible) from the proxy shouldn't be trusted.

## Analysis

- **What limits accuracy is constant offsets, not noise.** Per-sample noise
  contributes well under 0.5%. Everything above 1% is a per-trial constant: F/T
  offset, `com_x`, pivot, tool offset and tilt offset. Model bias (rolling) adds a
  systematic −4 to −8% on z_c, depending on r/H.
- **On hardware, rolling and F/T offset partly cancel.** Fixing one without the other
  can make the published numbers look worse; that is what happens to the box and
  flashlight z_c. Fix them together.
- **The F/T offset matters because of the lever arm.** The sensor sits 0.2–0.5 m above
  the pivot, so 0.05 N of force offset becomes ~0.015 N·m of spurious torque. The
  tare is the cheapest fix available.
- **Every error lands on m, z_c and μ together** (|corr| 0.9–0.98). With `com_x`
  fixed, the fit identifies θ* and the amplitude m·|r_com|. Treat (m, z_c, μ) as one
  uncertain vector.
- **The ARC/UNARC gap isn't linear drift.** That leaves a direction-dependent effect.
  It matters for the windowed estimator, because a pull-only window would carry half
  the gap as bias.
- **Contact selection isn't fragile to 2 mm/1° camera error** for these picks. The
  open question there is the selector's accuracy, not geometry robustness.

## Next steps

Simulation items 3–6 of the previous list are done; their results are above. In
priority order:

1. **Validate rolling with an independent tilt on hardware** (`LAB_PC_PROMPT.md`, A1 and B3).
   First get the vision pitch logged during contact, then compare it with the ball
   tilt: the slope should be ~0.96 for the box and ~0.94 for the heart. If it holds,
   add `rolling_corrected_tilt` to `press_pull_estimator.object_tilt` (it needs r, and
   x0 and H from the first contact sample) and regenerate Table 2.
2. **Re-tare the F/T at the pre-squash pose, right before SQUASH** (A3, B5). It's a
   one-line controller change, and the MC says it takes the F/T term from 3.3% to 1.1%.
3. **Characterise the F/T on the robot** (B1 drift without load, B2 hysteresis under
   load). B2 decides whether the remaining ARC/UNARC gap is the sensor.
4. **Find the ARC/UNARC gap's cause with a speed test** (B4). If it's a Coulomb-like
   torque that reverses with sweep direction, the windowed estimator should fit one
   offset term ±τ_f that flips sign with sweep direction, rather than average sweeps
   it doesn't have yet.
5. **Improve the `com_x` prior.** It's now the largest single term (4% at 2 mm), and
   the worst for small-com_x objects (flashlight). Measure the camera's actual com_x
   error, and consider fitting it from two press locations.
6. **Investigate the monitor's no-slip failure** before trusting its numbers (B4).
7. **Selection rule.** The robust pick (`contact_robustness.robustness`, best score with
   P(feasible) ≥ 0.95) costs nothing, but the re-simulation shows it isn't needed at
   2 mm/1°. Keep it as a guard for worse reconstructions (≥4× gives 0.3 on the proxy;
   re-simulate there before relying on it). It lives in `uncertainty/`, not
   `contact_selection/`, to avoid the conflict with the remote branch.

## Outputs

`outputs/uncertainty/2026-10-06_{analysis,measured,retared}/` hold, for each preset:

- tables: `bias_breakdown.json`, `sensitivity.csv`, `sweeps.csv`, `sweep_geometry.csv`,
  `confidence.csv`, `contacts.csv` and `mc_all_samples.csv`, each with a matching JSON;
- `nls_vs_mc.json`, `sanity_checks.json` and `meta.json`;
- plots: `mc_distributions.png`, `sensitivity.png`, `sweeps.png`, `covariance.png`,
  `confidence.png` and `contact_robustness.png`.

`2026-10-06_analysis/` also has `hardware_check.json` and `contact_resim.json`.

## Not done

Robot experiments (see `LAB_PC_PROMPT.md`), Bayesian inference, learned uncertainty,
active probing, shove propagation, and a constant-torque-offset term in the fit (step
4 decides whether it's worth it).
