# Sliding-window parameter estimator — build notes

**Status: not implemented. This is a spec for you to build, not a description of
existing code.** Everything it depends on — the press-and-pull rollout, the
logged channels, the frame conventions — is in place and noted below.

Goal: turn the offline batch fit into something that produces an estimate *and a
confidence* at every timestep of a rollout, so a policy can consume it as a
feature under condition C.

---

## 1. What exists today

Two batch implementations, neither online:

| Where | What it fits | Notes |
|---|---|---|
| `notebooks/main.ipynb` cells 8–9 | `(com_z, mass, mu)` jointly | `scipy.optimize.least_squares` on a loaded `.npz`. Masks out `|pitch| < 1°`. |
| `irb120_ws/.../estimation/estimate_params.py` `_fit_phase()` | two methods, see below | More developed than the notebook. Start here, not from the notebook. |

`estimate_params.py` runs **two** fits per phase and reports both:

- **Method A** — find θ\* from a *zero crossing*, then get `com_z` from geometry
  and `mass` from a 1-parameter torque fit.
- **Method B** — joint 2-parameter `least_squares` for `(mass, com_z)` from
  torque balance.

**Build the online version from Method A.** Method B is a nonlinear solve that
you would have to re-run every window; Method A's core is a *straight-line fit*,
which has a recursive form, a closed-form confidence interval, and — the reason
this matters — **can be extrapolated before the sweep reaches θ\***.

## 2. The idea

Method A, from `estimate_params.py:211-216`:

```python
# torque-corrected tangential signal; zeroes at theta* for any contact geometry
gx = w_app_O[sel, 3] - (r0[0] / r0[2]) * w_app_O[sel, 5]
fx_coeffs   = np.polyfit(y_pitch_deg, gx, 1)      # gx ≈ m·θ + b
theta_fx_deg = -fx_coeffs[1] / fx_coeffs[0]       # zero crossing = θ*
com_z_fx     = COM_GT[0] / np.tan(np.deg2rad(abs(theta_fx_deg)))
```

`gx` is linear in pitch and crosses zero at the balance angle. So:

> You do not have to *reach* θ\* to estimate it. Fit the line to whatever arc
> you have so far and extrapolate the crossing.

That is the whole reformulation. Refit the line as the window grows; the fit's
own covariance gives you the confidence channel for free, with no extra
machinery and no privileged information.

Early on the line is short and noisy, so the extrapolated crossing has a wide
interval — which is exactly the honest answer, and exactly what the policy needs
in order to learn to discount the estimate while it is uninformative.

## 3. Assumption you agreed to

**`com[0:2]` (the horizontal CoM) is known**, carried over from a previous trial
on the same object. Both batch methods already assume this — they pass
`COM_GT[0]` in and only solve for `mass` and `com_z`. Do not treat recovering
`com_x` online as in scope here.

If you later want it self-contained, `push_selection/push_selection_pipeline.py`
computes a CoM projection from the mesh and is the natural source.

## 4. Suggested shape

```
parameter_estimation/online_estimator.py

class WindowedTipEstimator:
    def __init__(self, com_x, r0, window_sec=None, min_samples=..., ...)
    def update(self, t, pitch_rad, w_app_O) -> Estimate | None
    def reset(self)

@dataclass
class Estimate:
    theta_star_deg: float
    com_z: float
    mass: float
    confidence: float        # in [0,1]
    n_samples: int
    window_span_deg: float   # pitch range the fit actually covers
```

`update()` per tick, returning `None` until `min_samples` is reached. Keep it a
pure function of what a *robot* can measure — no `mujoco` import, no ground
truth — or it will not survive the move to hardware.

### Fit

Either recursive least squares on `gx = m·θ + b`, or a plain `np.polyfit` over a
deque. Start with the deque: a rollout is ~27k samples and `polyfit` on a few
thousand points per tick is still cheap offline, and you can swap in RLS once the
behaviour is right. Do not optimise before the estimator works.

### Confidence

The quantity you want an error bar on is a *ratio*, `θ* = -b/m`, so propagate
through it rather than reporting the slope's error alone:

```python
coeffs, cov = np.polyfit(theta_deg, gx, 1, cov=True)
m, b = coeffs
# delta method on f(m,b) = -b/m
J = np.array([b / m**2, -1.0 / m])
var_theta = J @ cov @ J
```

Then map `var_theta` to `[0,1]` however you like — a saturating
`1/(1+var/var_ref)` is fine. Two failure modes to guard against explicitly:

- **`m ≈ 0`** — a flat line has no crossing. The variance blows up correctly,
  but `theta_star` itself goes to ±inf, so clamp before returning.
- **Extrapolating far outside the observed range.** Track `window_span_deg` and
  penalise confidence when the predicted crossing sits well outside it. A line
  fit over 2° of sweep predicting a crossing at 18° is an extrapolation of 9×,
  and should say so.

## 5. Data contract

`scripts/press_pull_simulation.py` writes an npz with these channels, at 1000 Hz:

| Key | Shape | Meaning |
|---|---|---|
| `t_hist` | (N,) | sim time |
| `state_id_hist` | (N,) | phase: 1 SQUASH, 2 LULL, 3 ARC, 4 UNARC, 5 RETRACT, 6 DONE |
| `w_hist` | (N,6) | wrench, **sensor** frame, `[fx,fy,fz,tx,ty,tz]` |
| `w_world_hist` | (N,6) | wrench, **world** frame, same ordering |
| `arc_angle_hist` | (N,) | contact angle about the tipping edge (rad) |
| `f_radial_hist`, `f_tangent_hist` | (N,) | force projected into the arc frame |
| `obj_pose_hist` | (N,4,4) | object pose — **ground truth, for scoring only** |
| `quat_hist` | (N,4) | object quaternion `(x,y,z,w)` |
| `ball_pose_hist`, `sens_pose_hist` | (N,4,4) | contact point and F/T sensor poses |
| `con_bool_hist` | (N,) | pusher-payload contact flag |
| `attempt_force_refs`, `attempt_tipped`, `attempt_max_tip_deg` | (A,) | per-attempt force ladder and outcome |

Segment on `state_id_hist == 3` (ARC) exactly as `estimate_params.py:184` does.

## 6. Traps

**Wrench ordering differs between codebases.** `w_hist` here is `[f, t]` —
force first. `estimate_params.py` builds `w_meas_S = hstack((tau, f))`, the
Modern Robotics `[tau, f]` convention, and `model_bkwd_wrench` expects *that*.
So `w_app_O[:, 3]` is `f_x` and `w_app_O[:, 5]` is `f_z` in the Method A snippet
above. Reorder before calling into `com_estimation.py`, or you will fit torque
against force and get a plausible-looking wrong answer.

**Sensor frame vs world frame.** `ft_get_reading()` returns the sensor frame.
`w_world_hist` is the pre-rotated version (see
`PressPullFSM._world_wrench`). The hardware topic `/netft_data_transformed` is
already world-frame, so hardware code that looks equivalent is not.

**The 1° mask is a property of the model, not a tuning knob.** The torque model
degenerates at θ=0 — that is why the batch fit drops those samples. Your window
will inevitably contain them at the start. Do not silently include them; either
exclude them from the fit and let confidence stay low, or handle the degeneracy
explicitly. Quietly fitting through θ=0 is the one thing that will make this look
like it works when it does not.

**Only fit rollouts where the object actually tipped.** When the finger slips
and slides across the top face, tangential force collapses in almost exactly the
same way it does at the true balance point — the force signature alone cannot
distinguish them. The FSM labels this: check `attempt_tipped` / the `tipped`
flag, and see §7. A slip rollout will produce a confident, meaningless θ\*.

**Tick counts are not durations.** The sim runs at 1000 Hz, the hardware at
100 Hz. If you port any windowing constant expressed in samples, scale it —
`PressPullFSM._ticks()` does this for the FSM's own debounces.

## 7. Getting data

```bash
source activate_venv.sh
PYTHONPATH=$PWD python parameter_estimation/scripts/press_pull_simulation.py --object 0 --adaptive
```

### ⚠ Open issue: the box does not currently tip

**The FSM runs the full sequence correctly but does not yet tip object 0.** Every
configuration tried so far ends with `tipped=False` and the object rotating
< 0.5°. Do not build the estimator against these rollouts expecting a real θ\* —
there is no tip in them. Resolving this is the prerequisite.

What has been measured, so none of it needs redoing:

- **Contact is correct.** One clean ball↔payload contact on the box's top face
  at z=0.3497, vertical normal, normal force tracking the reference (4–5 N at a
  5 N setpoint). The force controller works.
- **Friction is not the limit.** MuJoCo combines geom friction by **maximum**,
  not geometric mean, so the finger↔box contact runs at μ=1.5. Available
  tangential force is ~6.2 N; measured is ~1.03 N, i.e. 24% of the cone. Raising
  the box's own friction from 0.1 → 1.0 changed the result by 0.03°.
  (Note: `shove_simulation.py:129-131` computes "effective mu" as `sqrt(μ₁μ₂)`.
  That is the wrong combination rule for MuJoCo. It is only printed, not used.)
- **Actuation is not the limit.** The position servos have no `forcerange` cap
  and kp=200/100.
- **The torque balance is genuinely adverse.** About the pivot, with a 5 N press
  at the top-face centre:

  | term | moment | sense |
  |---|---|---|
  | press force, 5 N at 0.052 m from pivot | +0.262 N·m | restoring |
  | gravity, 6.5 N at 0.050 m | +0.325 N·m | restoring |
  | drag, 1.03 N at 0.300 m | −0.309 N·m | tipping |
  | **net** | **+0.278 N·m** | **restoring** |

  Tipping needs ~1.96 N of drag; the contact delivers ~1.03 N. Note that the
  *press itself* supplies almost half the restoring moment, because the contact
  normal is vertical rather than along the arc radius.

- **Pressing harder does not fix it, and cannot.** Press force appears on both
  sides — it raises the friction available *and* the restoring moment. The
  adaptive ladder (5 → 12.21 N) moves rotation monotonically but only from 0.02°
  to 0.44°. Pressing nearer the pivot (`--press-offset-x -0.03/-0.04`) helps in
  the right direction and is still not enough.

The unexplained part, and where to look next: **why the drag force plateaus at
~1.0–1.1 N when the friction cone permits 6.2 N and the servos can push.** The
ball slides across the top face while the box stays put, which should mean
saturated sliding friction. Suspect the contact solver regime — the box carries
`solref="0.0001 1.0"` (very stiff) and the sliding speed is only 0.008 m/s.
Worth trying: a softer `solref`, `condim`/`solimp` on the payload geom, or a
larger `--speed-scale` to leave the creep regime.

Once it tips, `--adaptive` becomes meaningful, and the force ladder is worth
keeping as a measurement in its own right: the lowest normal force that carries
an object over bounds its restoring moment, it is recorded in
`attempt_force_refs`, and unlike θ\* it is known *before* any fit converges. It
is the obvious thing to hand the policy during the window where the windowed
estimate is still worthless.

## 8. Validating

Ground truth for box/heart/lshape/flashlight is in `object_params.json`
(`mass_gt`, `com_gt_onshape`, `com_gt_offset`; effective CoM is the difference).
Monitor and soda have no entry — they will `KeyError`.

The test that matters is not final accuracy — the batch fit already has that.
It is: **plot estimated θ\* and its confidence band against sweep angle, and
find how early the band tightens enough to be useful.** If the answer is "only
after the object is past θ\* anyway", the extrapolation is not buying anything
and the approach needs rethinking before it goes anywhere near a policy.

Box is the object to develop against: it is the only one with a `theta_star`
entry (18.435°), its geometry is a primitive box rather than a mesh, and its
`site:obj_frame` sits exactly on the tipping edge that `com_gt_onshape` is
measured from.
