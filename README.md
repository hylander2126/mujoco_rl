# VLA Homework Series

This sub-project is the simulation-only VLA path for the IRB120 MuJoCo project,
organized as a homework series. Each homework should be small enough to finish,
measure, and render, while still teaching one real part of VLA training.

## Homework 1 - Tray Bin-Sorting

### What we're building

A robot arm carries a tray as its end-effector with a colored cube on it. Given a
language prompt, for example `"sort the cube into the corresponding bin"`, the
robot moves its arm to the correct bin and tips the tray so the cube tumbles in.
The prompt is intentionally color-agnostic so the policy must read the cube's
color from the camera image rather than from the instruction text — see
[Debugging: Why The Policy Drops Cubes Between The Bins](#debugging-why-the-policy-drops-cubes-between-the-bins)
for why that distinction mattered.

Initial scope:

- two bins
- two colors
- fixed scene
- MuJoCo-only execution
- no teleoperation

### Goals

**Primary:** Understand how a VLA works end-to-end by building one, not just
reading about it.

**HW1 target:** reach 95% success rate with a fine-tuned policy on red/blue cube
sorting.

**Secondary:** Produce a visually compelling demo: rendered MuJoCo video plus
language prompt leading to the robot executing the correct sort. This should be
portfolio-site friendly.

### Architecture

```text
Language Prompt + Camera Image
         |
         v
    [VLA Policy]
         |
         v
    Joint Actions
         |
         v
  MuJoCo Environment
```

The VLA policy takes two main inputs every timestep: a camera image of the scene
and the language prompt string. It outputs joint velocities, joint targets, or
end-effector deltas depending on the action interface we choose.

The starter scaffold currently uses a tiny local policy so the training loop is
easy to inspect. The intended next version uses a pretrained vision-language
backbone, such as OpenVLA or a small CLIP-based encoder, with a lightweight
action head fine-tuned on demonstrations.

## Implementation Plan

### Step 1 - Environment

Status: implemented as a first pass.

The code now centers on the VLA environment at `environment/env.py`. HW1 task
defaults live in `task.py`, and HW1 scene generation lives in
`environment/scene.py`.

Current HW1 environment pieces:

- robot with tray end-effector
- red and blue bins
- one free cube, colored red or blue at reset
- cube population onto the tray using MuJoCo ground-truth tray site pose
- fixed `vla_cam` camera for image recording
- success/reward helper based on whether the cube reaches the matching bin
- robot home joint pose default
- domain-randomization config hook for future variation in:

- starting/home joint noise
- object position
- camera position
- light position
- action noise

### Step 2 - Demo Collection

Status: implemented as a first pass.

HW1 does not use teleoperation yet. Instead, `robot/controllers/hw1_oracle_policy.py`
is a ground-truth scripted state machine that uses MuJoCo state and robot IK to:

- move the tray above the correct bin
- tip the tray toward that bin
- hold long enough for the cube to fall
- return toward home

The collection script alternates red and blue episodes and records
behavior-cloning tuples:

```text
(image, prompt, state, action, cube_color, success)
```

The current `state`/observation vector is 24 floats:

```text
[joint_positions(6), joint_velocities(6), force_torque(6), ee_position_xyz(3), cube_position_xyz(3)]
```

The state vector itself does not include cube color or the language instruction.
The default BC trainer now uses the recorded image and prompt alongside this
state vector, so color/task information can enter through those inputs. The
`state_only` baseline is still available, but it cannot reliably choose the
correct bin by color.

This controller is allowed to be explicit and GT-based. Its job is to create the
first dataset, not to be impressive.

### Step 3 - Fine-Tune a Pretrained VLA

Training a VLA from scratch is not the target. The practical path is to start
from a small pretrained model, then fine-tune it on the collected demonstrations
with behavior cloning.

OpenVLA is the most relevant open VLA direction to investigate. A smaller
CLIP-style encoder plus action head is also a good intermediate step if OpenVLA
is too heavy for iteration.

### Step 4 - Eval + Render

Run the fine-tuned policy in MuJoCo, record rollouts, measure success rate, and
render a clean demo video.

## Why Fine-Tune Instead Of Training From Scratch?

Training a VLA from scratch requires massive compute and data. Fine-tuning a
pretrained model on a narrow task like bin sorting is tractable on a single GPU
and still teaches the meaningful parts:

- how the action head works
- how language conditioning flows through the model
- how behavior-cloning loss behaves
- why the data distribution matters

The learning value is almost the same. The compute cost is not.

## What You'll Actually Learn

- How visuomotor policies are structured and trained
- How language conditioning is injected into a policy network
- The gap between scripted demos and generalized behavior
- Why distribution shift hurts behavior-cloned policies
- How domain randomization can make policies less brittle

## Current Folder Map

- `task.py`: centralized HW1 task spec: colors, bin locations, camera name,
  prompts, home pose, and scripted expert timing.
- `environment/`: VLA-specific MuJoCo env, currently `VLAIRB120Env`.
  `environment/scene.py` owns the generated HW1 MuJoCo XML.
- `scripts/`: runnable collect, train, evaluate, and shared CLI helpers.
- `robot/controllers/hw1_oracle_policy.py`: HW1 scripted expert for demo generation.
- `robot/controllers/robot.py`: shared IRB120 robot wrapper, IK, force/torque
  helpers, and the simple position actuator controller used by HW1.
- `environment/default.yaml`: sim, data, training, and domain-randomization defaults.
- `models/`: starter VLA-shaped policy.
- `outputs/`: default location for rollouts, checkpoints, and figures.

The old RL sandbox, phase-controller experiments, object-library loader, and
checked-in generated XML were removed so this repo can stay focused on the VLA
homework path.

## Basic BC Train/Eval Commands

Run these from the repo root, `mujoco_rl/`.

If you are running without a desktop OpenGL context, prefix MuJoCo commands with
`MUJOCO_GL=egl`. On a normal desktop viewer run, you can omit it.

F/T biasing is disabled in this simulation implementation, so the F/T channels
are gravity-compensated but not per-reset zeroed.

Collect the default dataset used by training:

```bash
MUJOCO_GL=egl python3 main.py collect \
  --episodes 20 \
  --max-sim-time 5.0 \
  --output outputs/rollouts/sim_vla_rollouts.npz
```

Train the image + language + state behavior-cloning policy:

```bash
python3 main.py train \
  --dataset outputs/rollouts/sim_vla_rollouts.npz \
  --epochs 20 \
  --batch-size 64
```

This writes:

```text
outputs/checkpoints/vla_bc.pt
```

Evaluate the trained checkpoint headlessly:

```bash
MUJOCO_GL=egl python3 main.py eval \
  --checkpoint outputs/checkpoints/vla_bc.pt \
  --episodes 10 \
  --max-sim-time 5.0
```

Evaluate headlessly and save one MP4 per episode under `outputs/videos/`:

```bash
python3 main.py eval \
  --checkpoint outputs/checkpoints/vla_bc.pt \
  --episodes 1 \
  --max-sim-time 5.0 \
  --render
```

`--render` writes native 720x720 videos using MuJoCo's offscreen renderer, so it works
over VS Code SSH without `DISPLAY`. On headless hosts the code automatically selects EGL. Both
training and evaluation prefer CUDA when PyTorch can initialize it and print
the selected device; an incompatible driver falls back to CPU. The requirements
pin PyTorch's CUDA 12.1 build, which supports the project's RTX 4060 host with
its NVIDIA 535 driver.

For a much faster evaluation, run the policy every five physics steps and hold
each target between updates:

```bash
python3 main.py eval \
  --checkpoint outputs/checkpoints/vla_bc.pt \
  --episodes 5 \
  --render \
  --max-sim-time 5.0 \
  --control-stride 5
```

On the RTX 4060 host, a one-second rollout benchmark dropped from 1.34 seconds
at stride 1 to 0.29 seconds at stride 5. MuJoCo physics still advances at its
1 ms timestep; this reduces the expensive camera renders and model inferences.
Stride 1 remains the default because it most closely matches dense training
data. Larger strides are faster but can change policy behavior.

The default trainer now uses the recorded camera image, prompt string, and robot
state, so cube color can affect the policy. To run the old proprioceptive-only
baseline, add `--policy-type state_only`; that writes
`outputs/checkpoints/bc_only_states.pt`.

## Choosing Episodes And Epochs

`--episodes` during collection means how many full expert demonstrations to
record. One episode is one reset-to-done rollout: cube starts on the tray, the
expert moves to the selected bin, tips, and returns or times out. More
collection episodes means more training data and more variation across red/blue
tasks, but collection takes longer. For smoke tests, use 2 episodes. For a first
real VLA-shaped BC run, try 20-50 episodes. For judging whether the policy is
actually learning, prefer 100+ successful expert episodes.

`--epochs` during training means how many full passes the learner makes over the
recorded dataset. One epoch uses every training sample once, split into batches.
More epochs let the model fit the demonstrations better, but too many can
overfit, especially with a tiny dataset. For smoke tests, use 1-5 epochs. For a
first real run, try 20-50 epochs and watch train vs validation loss. If both are
still falling, more epochs may help. If train loss keeps falling but validation
loss rises, collect more data or stop earlier.

`--episodes` during eval means how many policy rollouts to run in MuJoCo. Use 1
episode for a quick render check, then 10-20 episodes to get a rough success
rate once the policy looks plausible.

## Debugging: Why The Policy Drops Cubes Between The Bins

After training the `vla` policy on images + the language prompt + state, the
robot reliably moves and tips the tray, but the cube lands between the red and
blue bins instead of in the correct one. This looks like classic behavior-cloning
mode averaging: when a model is trained with MSE regression against a
multimodal target (here, "go left" vs "go right" depending on color), and the
model isn't actually using the signal that disambiguates the mode, MSE pushes
it toward predicting the mean of the two valid actions, i.e. the point between
the bins.

The first diagnostic step is to check whether that's even the right hypothesis:
does the policy's predicted action actually change when color/instruction
conditioning changes, or is it just not receiving that signal at all? Run:

```bash
python3 main.py diagnose \
  --checkpoint outputs/checkpoints/vla_bc.pt \
  --dataset outputs/rollouts/sim_vla_rollouts.npz
```

This loads the checkpoint, takes paired samples from both colors, and for each
one swaps only the instruction, only the image, or both, then measures how
much the predicted action shifts relative to the dataset's typical per-dimension
action std (`scripts/diagnose_conditioning.py`). A near-zero fraction means that
channel has essentially no effect on the decision.

On this project's checkpoint, the result was:

```text
  instruction swap only : 6.0823
  image swap only       : 0.0004
  both swapped          : 6.0823
```

This was a useful surprise. The **vision branch is effectively dead** — swapping
in an image of the other-colored cube changes the predicted action by a
fraction of a percent of typical action variation, meaning `TinyVLAPolicy`
learned to ignore the camera image entirely and is making the color decision
almost purely from the instruction string. The instruction swap, on the other
hand, swings the action by over 6x the typical action std — that's not a
healthy, well-calibrated decision boundary, it's a sign the tiny char-level
language encoder (`CharInstructionEncoder`) is producing large, somewhat
unstable shifts in the joint output rather than a clean two-way switch.

So the "averaging between bins" symptom isn't simply "MSE always averages
multimodal actions" in the abstract — it's that the model found language as
its dominant (and only working) conditioning channel, and that channel's
influence is large but not robust. Likely contributors:

- the vision tower's `AdaptiveAvgPool2d` + 64-dim global pooling may be too
  lossy, or the conv stack just never got gradient signal strong enough to use
  it once the language path already explained the color-dependent variance
- with only `~20-50` collected episodes split across 2 colors, there is very
  little data to teach a CNN to extract a cue as simple as "what color is the
  cube," so the language string was the lower-effort path during training
- MSE regression has no mechanism to commit to one mode over a blended one;
  even where language conditioning is mostly correct, residual averaging
  across nearby timesteps/episodes can still nudge the trajectory toward the
  midpoint

### Fix Options, In Order Of Effort

1. **Strengthen vision supervision / give it an easier signal.** Crop or
   highlight the cube region, increase image resolution, or pretrain the small
   CNN against a color-classification auxiliary loss. Cheap structural change:
   add an auxiliary `cube_color` classification head off the vision embedding
   and weight it into the training loss, so the conv stack is forced to learn a
   color-relevant representation directly instead of only through the action
   loss.
2. **Move conditioning later/stronger in the network (e.g. FiLM).** Right now
   image, language, and state embeddings are concatenated once before the
   final MLP. FiLM-style conditioning (use the language embedding to scale/shift
   the vision features) would make it harder for the network to find a path
   that ignores vision.
3. **Replace MSE with a loss that supports multiple modes** — discretized
   action bins + cross-entropy, or a small Gaussian-mixture action head. This
   is the standard fix for BC mode-averaging in general, independent of the
   vision/language imbalance found above.
4. **Collect more, more varied episodes.** At 20-50 episodes total split
   across 2 colors, there isn't much data forcing a CNN to discover the color
   cue. More episodes (and domain randomization on cube position/lighting)
   would push the vision branch to generalize past "always look at the prompt
   string."
5. **Action chunking** if the issue persists after the above — predicting a
   short trajectory chunk instead of one step at a time prevents
   per-timestep re-averaging near the decision boundary.

Given the diagnostic above, the most promising next step here is #1 or #2
(fix the obviously broken vision conditioning) before reaching for a heavier
multimodal-action-head solution in #3 — the policy currently isn't really
multimodal in its inputs at all, so it never gets the chance to need a
multimodal output.

### Implemented Fix: Forcing The Vision Branch To Carry Color

Two changes closed the gap above:

1. **Removed the color word from the instruction template** (`task.py`):
   the prompt is now the fixed, color-agnostic
   `"sort the cube into the corresponding bin"`. This removes the shortcut
   entirely — the language branch can no longer answer the task by itself,
   since every episode now produces the identical instruction string.
2. **Gave the vision tower a reason and the capacity to encode color**
   (`models/policy.py`):
   - replaced the `AdaptiveAvgPool2d((1, 1))` global average (which let the
     two large, constant-colored bins dominate the pooled signal and washed
     out the small, color-varying cube) with `AdaptiveMaxPool2d((2, 2))`,
     keeping a little spatial structure and favoring locally-salient
     activations over a global blur
   - added an auxiliary `color_head` that predicts cube color directly from
     the vision embedding, trained jointly with the action loss
     (`scripts/train_bc.py`, weighted by `color_loss_weight`, default `0.5`).
     This gives the conv stack a direct, undeniable reason to encode color
     instead of relying on the action loss to discover it indirectly.

Re-running the diagnostic after these two changes (on a small smoke-sized
dataset, just to confirm the mechanism rather than to get a calibrated
number) gave:

```text
  instruction swap only : 0.0000
  image swap only       : 208.2731
  both swapped          : 208.2731
```

Instruction swap is exactly zero now, which is expected — the instruction
text is literally identical across colors, so there's nothing left for that
swap to change. Image swap went from 0.0004 to over 200x typical action std:
the policy is now unambiguously deciding bin choice from the camera image.
The auxiliary color classifier also reached 100% train/validation accuracy
within a few epochs on that smoke run, confirming the vision tower can now
actually extract color, not just default to it.

The magnitude (208x) is large enough to be a yellow flag on its own — on a
real, larger dataset this is worth re-checking; a properly calibrated policy
should shift smoothly across the decision boundary, not swing by two orders
of magnitude. If that holds up on the full dataset, options #2 (FiLM-style
conditioning) and #3 (a real multimodal action head) from the list above are
still worth revisiting to make the color-to-action mapping less brittle.

## Debugging: Does The Policy Track Bin Position, Or Memorize It?

Once the color-sorting fix above made the policy actually use the camera
image, the next question is what it's really doing with that image: is it
finding the correct-colored bin wherever it currently is, or has it just
memorized "see red → execute this one fixed joint trajectory" the same way
it previously memorized "hear red → execute this one fixed joint
trajectory"? Both bins were always in the exact same two physical spots
throughout data collection, so a model could ace training and eval without
ever learning anything about *position* at all.

The cheapest way to find out is to break the assumption the model might be
relying on: physically swap which color occupies which bin slot, and see if
the cube still ends up in the correct-colored bin. `task.swap_bin_colors()`
builds a mirrored task variant (the two bins' XY positions, the scripted
expert's pre-drop hover poses, and its wrist-tip directions are all
mirror-symmetric across the two colors by construction, so "swap" is just
trading those dict values between the two color keys). `VLAIRB120Env.reset()`
accepts a `swap_bins` option (`True`/`False`/`"random"`) that rebuilds the
scene with the bins physically relocated accordingly — the camera image, the
live bin site positions used for the success check, and the scripted
expert's IK targets all automatically agree, since they all read from
whichever task variant is active for that episode.

Run it with:

```bash
python3 main.py eval \
  --checkpoint outputs/checkpoints/vla_bc.pt \
  --episodes 16 \
  --bin-layout random
```

`--bin-layout` is `normal` (always the trained layout), `swapped` (always
mirrored), or `random` (50/50 per episode, the actual generalization test).
Eval prints a per-layout success-rate summary at the end. Result on this
project's checkpoint:

```text
Success rate (normal bin layout, n=4): 100.00%
Success rate (swapped bin layout, n=12): 0.00%
```

Not a partial degradation — a complete failure. Every swapped episode timed
out (`steps=5000`, the full episode budget) rather than landing near the
wrong bin or missing narrowly. That pattern means the policy isn't tracking
the bin's *position* from the image at all; it's still mapping "color X" to
one fixed, hardcoded Cartesian trajectory, the same shortcut-learning failure
as before, just one level further down (object color was no longer a
shortcut, but a fixed scene layout still was).

Fix options, same shape as the color problem before it:

1. **Add bin-position domain randomization to data collection**, not just
   eval. Without this, no architecture change will generalize — the model
   has literally never seen the bin anywhere else.
2. **Give the state/action representation a reason to use bin position
   explicitly** — once bin position varies in training data, consider adding
   bin or tray-target position into the state vector (if you want this to be
   a proprioception-grounded skill) or trust the vision branch to localize it
   directly from pixels (more end-to-end, harder, more honest to the "VLA"
   framing).
3. **Re-run the same conditioning diagnostic logic against bin position**
   once randomized training data exists — `scripts/diagnose_conditioning.py`
   currently swaps color-related image/instruction pairs; the same swap-and-
   measure idea extends directly to swapping bin-layout while holding color
   fixed, to confirm the fix actually closed the gap the way it did for color.

### Implemented Fix: Randomizing Bin Layout During Collection

`collect_sim_data` now accepts `randomize_bin_layout`, exposed as
`--randomize-bin-layout` on `main.py collect`. When set, episodes deterministically
cycle through all 4 `(color, swap_bins)` combinations in turn, so every
combination gets exactly the same number of demonstrations regardless of
total episode count — reusing the exact `swap_bin_colors` mechanism already
validated for eval, so the oracle expert still targets the correct physical
bin in every combination. The recorded layout per sample is saved as a new
`swap_bins` array in the dataset, in case you want to slice or inspect by
layout later.

(An earlier version of this sampled `swap_bins` independently at random per
episode instead of cycling deterministically. With only 20 collected
episodes, that produced a 2/4/6/8 split across the 4 combinations purely by
chance — see the regression below for why that mattered.)

```bash
MUJOCO_GL=egl python3 main.py collect \
  --episodes 50 \
  --max-sim-time 5.0 \
  --randomize-bin-layout \
  --output outputs/rollouts/sim_vla_rollouts_randlayout.npz
```

Important caveat: this only teaches the model **two** discrete layouts (4
total color × layout combinations), not a continuous notion of "find the bin
wherever it is." It directly targets the failure mode actually observed
(hardcoded left/right lookup), but doesn't yet prove the policy can handle a
bin position it has never seen at all. If `--bin-layout random` eval after
retraining on this data comes back at or near 100% on both layouts, the next
honest test is a *third*, never-trained bin position (e.g. a small XY offset
rather than a full mirror) to check whether it's actually localizing or just
learned a 2-entry lookup table instead of a 1-entry one. True continuous
generalization would need the oracle's hover/tip targeting generalized from
the current fixed per-color table to a formula computed relative to the
bin's live position — a larger change, intentionally deferred for now since
a subtly broken oracle would silently corrupt every demonstration collected
with it.

### Regression: "Goes To The Same Spot Regardless Of Color"

After recollecting with `--randomize-bin-layout`, retraining, and re-evaluating,
the policy got *worse* in an unexpected way: it drove to the same spot every
time, ignoring color entirely, not just bin layout. Before assuming the fix
was wrong, the diagnostic tools built earlier in this doc were the right
first move rather than guessing at architecture changes.

First check: did vision conditioning actually collapse again?

```bash
python3 main.py diagnose \
  --checkpoint outputs/checkpoints/vla_bc.pt \
  --dataset outputs/rollouts/sim_vla_rollouts.npz
```

```text
  instruction swap only : 0.0000
  image swap only       : 2.6758
  both swapped          : 2.6758
  bin layout swap only  : 3.9873
```

(`bin layout swap only` is a new diagnostic dimension added alongside this
regression — same idea as the color swap test, but holding color fixed and
swapping between a normal-layout and swapped-layout image instead.)

Both numbers are clearly nonzero — the vision branch hasn't gone dead the way
it did originally. So the conditioning pathway itself is intact; something
else is wrong. The next check was the data, not the model:

```python
import numpy as np
data = np.load("outputs/rollouts/sim_vla_rollouts.npz", allow_pickle=True)
# group episodes by (cube_color, swap_bins) and count
```

That turned up the real problem: the recollected dataset had only **20
total episodes**, split across the now-4 required `(color, swap_bins)`
combinations as `{(red, swapped): 6, (blue, swapped): 8, (red, normal): 4,
(blue, normal): 2}` — as few as 2 demonstrations for one entire combination.
Randomizing bin layout doubled the number of distinct behaviors the policy
needs to learn (4 instead of 2), but the dataset size didn't scale to match,
and the split across combinations was left to chance (independent random
sampling of `swap_bins` per episode) rather than guaranteed even. With 2-8
examples per combination, the rarest ones get drowned out during training,
and the policy falls back toward whatever the dominant combination's
trajectory looks like — which reads as "ignores color, always goes to the
same spot."

Fix applied: `collect_sim_data.py` now cycles deterministically through all
4 `(color, swap_bins)` combinations (see above) instead of sampling
independently, guaranteeing an even split for any episode count. The
remaining piece is scale — collect meaningfully more episodes than 20 total
now that there are 4 modes to learn, e.g. 80-160 (20-40 per combination), not
20, before retraining and re-running both diagnostics and `--bin-layout
random` eval.

### Is It Time For TinyVLA / SmolVLA Fine-Tuning?

Not yet, and this regression is a good illustration of why. Every failure in
this debugging arc so far — color mode-averaging, bin-position memorization,
and this latest regression — came from the same root causes: too little (or
too imbalanced) data for the number of distinct behaviors being asked of the
model, and/or insufficient explicit supervision for the policy to ground a
factor in the image. None of those are fixed by a bigger pretrained backbone.
Swapping in TinyVLA or SmolVLA now would mean debugging the exact same
classes of bug, except slower to iterate on (bigger model, heavier
fine-tuning infra, more compute per experiment) and harder to instrument
(the diagnostic tools here are simple specifically because the policy is
tiny and fully owned).

The stronger signal to bring in a pretrained VLA is when the *task* needs
something a frozen pretrained vision-language representation actually buys —
generalizing to colors, objects, bins, or instruction phrasings never seen
during this project's data collection, transferring to a new scene layout
without per-task data collection, or handling open-ended language rather than
one fixed template. That's a meaningfully different capability jump, not a
debugging fix, and it's worth treating as the next homework step once this
toy task's two known generalization axes (color, bin position) are both
confirmed solid with the current small model — not before, since a fine-tune
run won't be able to tell you whether a failure is "the new backbone" or
"the same data problem as last time."

## Current Limitations

This is not yet an OpenVLA fine-tuning pipeline. Right now it is a readable
VLA-shaped simulation scaffold:

- image input from MuJoCo is used by the default `vla` BC trainer
- prompt string input is used by the default `vla` BC trainer
- robot state input: joint positions, joint velocities, force/torque,
  end-effector position, and cube position
- 6D joint-delta action output, converted back to absolute joint targets for
  `env.step()`
- the optional `state_only` baseline has no color/task conditioning, so it can
  learn to drop cubes but cannot reliably choose the correct color bin
- behavior-cloning training loop
- scripted expert demonstrations

The scripted expert is a first pass. The next important validation step is to run
rollouts, inspect video, and tune bin placement, home pose, tray approach height,
and tilt angle until the expert itself is reliable before training a policy.
