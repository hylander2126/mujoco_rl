# Prompt for Claude Code on the lab PC

Paste everything below the line into Claude Code, started in `~/Documents/github/irb120_ws`
on the lab PC.

---

You are helping me (Steven) validate error sources in the press-and-pull estimator on
the real IRB120. The canonical estimator is `press_pull_estimator`, in
`~/Documents/github/press-pull-tipping/code`. Its function `estimate_press_pull()`
fits ARC and UNARC separately and averages them; that is intentional, so keep it.
The controller is `src/irb120_ros2/irb120_control/irb120_control/arc_static.py`.
Its logs are in `runtime_logs/<object>/arc_squash/*.npz` and `*.json`.

## What is already known (from the 40 logged trials and a MuJoCo study, 2026-10-06)

- **The fingertip ball rolls on the object's top face.** The finger holds its
  orientation through the arc (wrist pitch varies only 0.16–0.25° over ARC/UNARC), so
  the ball can't be carried rigidly. It rolls, and its centre moves `r·θ` along the
  face. The estimator reads tilt as the ball centre's rotation about the pivot. That
  gives `φ = θ − atan((x0 + rθ)/H) + atan(x0/H) ≈ θ·(1 − rH/(H² + x0²))`, where:
  - `H` is the ball-centre height above the pivot;
  - `x0` is its horizontal offset from the pivot;
  - `r = 0.01325` m is the URDF radius, not measured.

  Predicted slope φ/θ: box 0.958, heart 0.939, flashlight 0.940, monitor 0.974. In
  sim the measured slope matched the prediction (0.959 vs 0.958). Correcting it there
  removed the whole noise-free bias.
- **It only corrupts the tilt.** The applied torque about the pivot comes from the
  measured wrench moved to the pivot with the measured sensor pose. The moving contact
  point is therefore already in the measurement, and neither the moment arm nor the
  force vector is affected.
- **Correcting rolling on the logged hardware trials:**

  | Object | z_c error, published | z_c error, corrected |
  |---|---|---|
  | box | +0.7% | −3.8% |
  | heart | +15.1% | +7.4% |
  | flashlight | 0.0% | −6.3% |
  | monitor | +5.3% | +2.4% |

  With the SQUASH-start F/T offset also removed, the z_c errors are +0.5 / +11.1 /
  −0.7 / +2.9% and the m errors are +2.2 / +1.3 / −1.6 / +4.2%. The published errors
  were m +4.6 / +1.1 / +2.4 / +4.6%. This suggests the published box and flashlight z_c
  were accurate partly because the rolling and F/T-offset errors cancelled. That still
  needs an independent tilt measurement to confirm.
- **F/T levels in the logs**, all |vector|, as mean over the 10 trials per object:

  | Quantity | Measured |
  |---|---|
  | Offset at SQUASH start, relative to the tare | 0.026–0.081 N; 1.1–2.3 mN·m |
  | Drift within a trial | 0.012–0.023 N; 0.3–0.7 mN·m |
  | White noise, per axis std | 0.006–0.011 N; 0.2–0.3 mN·m |

  The offset at SQUASH start is very consistent within each object. Removing the
  offset and a linear drift does **not** change the ARC/UNARC disagreement: monitor
  m +5.1% → +5.2%, flashlight −3.1% → −3.4%. So the cause of that disagreement is
  still open.
- **The vision object pose has no samples during ARC/UNARC in any of the 40 logs**
  (`obj_time_s` holds 0–12 samples per trial, all outside contact). That is why the
  rolling effect hasn't been checked against an independent tilt yet.
- **The monitor fails the no-slip check.** Its peak-to-peak `|p_ball − pivot|` is
  8.4 mm against a 4 mm tolerance, and its trajectory-fit pivot is 32 mm from
  ARC_CENTER. The other objects are within 1–2 mm.

## Rules

- **Never move the robot without me.** Prepare the scripts and configs, then stop and
  tell me exactly what you're about to run. Wait for my go-ahead before every robot
  motion. Keep `REQUIRE_OPERATOR_CONFIRM` on.
- Don't change the estimator's logic in `press-pull-tipping`. New analysis goes in new
  scripts, under `irb120_control/irb120_control/estimation/` or `scripts/`, unless the
  repo's conventions say otherwise.
- Write every new log to `runtime_logs/uncertainty_checks/<YYYYMMDD>/<experiment>/`,
  in the same npz/json format `arc_static.py` already writes.
- Report the numbers you get, even if they contradict the predictions above.

## Tasks

### A. Offline (do these first, no robot)

1. **Vision during contact.** Find out why `/object_detector/detections` produces no
   poses during contact. Candidates: the finger occludes the object, the detector's
   rate or confidence gating, or TF drops (look for `Dropping detection` warnings).
   Start from `_on_detection` in `arc_static.py` and the `irb120_perception` package.
   Report the cause and the smallest fix that gives object pitch at ≥10 Hz during
   ARC/UNARC. Options include a fiducial on the side face away from the finger, or an
   IMU taped to the object.
2. **Ball radius.** Find the CAD or URDF source for `finger_ball_center` and the ball
   diameter. Tell me to measure it with calipers if nothing authoritative exists.
3. **Tare timing.** `tare_netft()` runs once at program start in `main()`, before the
   move to pre-squash, and for the batch scripts possibly once per batch. Report how
   long each logged trial ran between the tare and SQUASH, and whether the robot
   orientation was the same at both. Then propose a re-tare at the pre-squash pose,
   immediately before SQUASH, as a minimal diff to `arc_static.py` and
   `arc_static_batch.py`. Don't apply it until I approve.

### B. Robot experiments (prepare, then run with me)

1. **F/T drift with no load.** Tare at the pre-squash pose with the finger in its arc
   orientation, then hold still for 120 s and log the F/T at full rate. Do this 3
   times. Report the offset against time: whether it is linear, its rate in N/s and
   mN·m/s, and its size at 40–50 s, which is the length of a trial.
2. **F/T load hysteresis.** Tare, then press a rigid block with SQUASH force control
   at 5 N, hold for 40 s, retract, and log 10 s unloaded. Report the unloaded offset
   after minus before. Repeat 3× at 5 N and 3× at the monitor's press force. This tests
   whether the offset is caused by the load, which a linear-in-time correction can't
   remove.
3. **Rolling validation.** Once A1 gives object pitch during contact, run 3 box and 3
   heart trials. For each trial:
   - Compare the vision pitch θ with the ball-derived tilt φ from `object_tilt()`.
     Report the slope φ/θ over ARC, against the predictions above.
   - Report the object's pitch at the first LULL sample relative to before contact,
     which is the pre-tilt. Sim gives about −0.2° for the box at 5 N.
   - Refit with the vision θ in place of φ, and report m and z_c against ground truth.
4. **ARC/UNARC disagreement.** Run the box and the monitor at `ARC_TANGENTIAL_SPEED` × 0.5
   and × 2 (3 trials each). For each speed, report the ARC − UNARC difference in m and z_c.
   - If it scales with speed, it is rate-dependent: controller lag or damping.
   - If it is constant, it is a Coulomb-like resisting torque, such as friction at
     the pivot edge or rolling resistance. A torque that resists the motion reverses
     between the two sweeps, so it pushes their estimates apart in opposite
     directions. That may be why averaging the two sweeps works better than a single
     fit.

   For the monitor, also report the no-slip deviation, and find out why its
   trajectory pivot is 32 mm from ARC_CENTER. Look for slip, a rounded base edge, or
   the wrong ARC_CENTER for that object.
5. **Re-tare before SQUASH** (after A3 is approved). Run 5 box trials and compare them
   with the published Table 2 row: m 0.707 ± 0.012 kg, z_c 15.10 ± 0.28 cm.

### C. Deliverable

Write `runtime_logs/uncertainty_checks/<YYYYMMDD>/REPORT.md` with one table per
experiment and the paths to the logs. The per-trial checks above are implemented in
`mujoco_rl/uncertainty/hardware_check.py` on my desktop, if you want to reuse them. They are:

- the F/T offset at SQUASH start;
- the drift from RETRACT to SQUASH;
- the finger rotation;
- the predicted rolling slope;
- the refits with rolling and offset corrections.

That script reads the gzipped CSVs that `tools/export_npz_to_csv.py` writes.
