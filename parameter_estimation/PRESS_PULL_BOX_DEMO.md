# Working near-edge press-pull box demo

Run from the repository root:

```bash
# Interactive viewer; choose a fresh output directory for each run.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_press_pull_demo.py \
  --show-viewer --no-video --output outputs/box_live

# Headless run that writes a small, streamed MP4 at 2x playback speed.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_press_pull_demo.py \
  --output outputs/box_video

# Physics-only run, without a display, renderer, or ffmpeg.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_press_pull_demo.py \
  --no-video --output outputs/box_headless
```

Use the repository `.venv` created for contact selection, or install
`contact_selection/requirements.txt`. Video additionally requires `ffmpeg` and a
working MuJoCo rendering backend. The output directory must not already exist;
this prevents overwriting an earlier experiment. Closing the viewer interrupts
the run rather than reporting a completed interaction.

## What works

The existing physical IRB120 presses on the original basic box at world
`(0.536, 0.080, 0.350)` m, **6 mm inside its near-X tipping edge**. It presses
with 5 N, settles for 1 s, ramps the pull over 2 s, tips, reverses the arc, and
retracts. The original `PressPullFSM`, robot controller, scene loader, and
independent rollout evaluator are reused. No new manipulation policy, slider
robot, welded contact, or per-step object teleportation is introduced.

The successful default trial measured:

| Diagnostic | Result |
|---|---:|
| Intended -Y tip during ARC | 10.57° |
| ARC fingertip contact | 100% |
| Maximum pivot displacement during ARC | 0.47 mm |
| Maximum off-axis rotation during ARC | 0.20° |
| Maximum ball contact force over the sequence | 6.00 N |
| Completion / existing feasibility checks | Pass |

At half the timestep (0.5 ms), the tip was 10.56° with 0.48 mm pivot drift and
100% ARC contact. The default 1 ms rollout also reproduced with video enabled.
This is controlled partial tipping and return, not a full overturn.

Saved demo: `outputs/press_pull_box_demo_clean/demo.mp4`.
Results, full native/controller diagnostics, resolved configuration, initial
integration state, and compiled model are saved alongside the video. MJB replay
requires a compatible MuJoCo version; these trials used 3.14.0.

## Changes that enable it

All physical settings are applied to a fresh demo model only. Original XML
assets and existing script defaults are unchanged.

* Elliptic friction cone, `impratio=10`, `noslip_iterations=10`.
* Fingertip sliding friction 2.0 and geom priority 1, so its existing contact
  softness and dimensionality govern the finger contact.
* Table sliding friction 0.5. Originally the table coefficient is 0 and the box
  coefficient is 0.1. Improving fingertip grip alone produced about **11 cm of
  pivot drift** in an exploratory 8 N trial: adequate support friction matters
  as well. The successful demo uses 5 N and a closer-to-edge contact.
* Optional wrist pitching at the commanded arc rate. The command compensates
  `omega × (ball - tool0)` because the robot Jacobian is at the flange, keeping
  the requested ball-centre trajectory consistent during wrist rotation.
* Optional stop when tangential-force magnitude falls below 10% of its ARC peak,
  gated by the existing minimum sweep and debounce. Existing force limits,
  contact-loss checks, angle cap, and return phases remain active. Radial force
  correction is capped at 5 mm/s in this preset.

`rotate_with_arc=False` and `arc_force_drop_fraction=None` are the controller
class defaults, retaining the original behavior. A complete legacy trial was
compared against its pre-change saved time, wrench, object-pose, ball-pose and
phase arrays: they matched exactly. The default fixed-orientation robot
controller itself was not changed.

Options for controlled experiments:

```bash
# Contact inset is in metres; allowed interval is (0, 0.05).
.venv/bin/python scripts/run_press_pull_demo.py --inset 0.015 \
  --no-video --output outputs/box_inset15

# Ablate wrist rotation or support grip explicitly.
.venv/bin/python scripts/run_press_pull_demo.py --world-fixed-finger \
  --no-video --output outputs/box_fixed_wrist
.venv/bin/python scripts/run_press_pull_demo.py --ground-friction 0 \
  --no-video --output outputs/box_original_ground
```

`--force`, `--finger-friction`, `--impratio`, `--noslip-iterations`, and
`--timestep` are also exposed. These alternative configurations are experiments,
not all validated successful presets. A failed feasibility check yields exit 1
and preserves the failure reasons and trajectory.

## Scope and interpretation

These are **demonstration contact parameters, not a calibration to hardware**.
The box remains physically simulated; mass and CoM metadata come from the
compiled model (0.663 kg), not the inconsistent object-parameter JSON.

The 10%-of-peak exit precedes the force zero crossing and **is not an estimate of
balance angle**. The demo makes no estimator-accuracy claim. Finger centre travel
in the object frame includes rolling and must not be interpreted as pure slip.
World-fixed and rolling-wrist trials should remain separately identified when
building future datasets. No learned classifier is trained by this script.

Validation:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  parameter_estimation/tests contact_selection/tests -q
```

32 tests pass, including contact placement, unchanged scene defaults, wrist
velocity compensation, peak-drop stopping, and the previous evaluator tests.
