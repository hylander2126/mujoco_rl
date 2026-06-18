# VLA Homework Series

This sub-project is the simulation-only VLA path for the IRB120 MuJoCo project,
organized as a homework series. Each homework should be small enough to finish,
measure, and render, while still teaching one real part of VLA training.

## Homework 1 - Tray Bin-Sorting

### What we're building

A robot arm carries a tray as its end-effector with a colored cube on it. Given a
language prompt, for example `"sort the red object into the red bin"`, the robot
moves its arm to the correct bin and tips the tray so the cube tumbles in.

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

It does not include cube color or the language instruction. That means the
current state-only BC policy can learn a generic physical skill like moving and
dropping the cube, but it has no classifier signal for red-vs-blue sorting. To
sort by color, the policy needs color information from the image, the prompt, a
one-hot color label, or some other explicit task-conditioning input.

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

F/T biasing is off by default in simulation. If you collect a dataset with
`--ft-bias`, evaluate checkpoints trained from that dataset with `--ft-bias` too
so the observation normalization matches.

Collect a quick smoke-test dataset:

```bash
MUJOCO_GL=egl python3 main.py collect \
  --episodes 2 \
  --max-sim-time 2.0 \
  --output outputs/rollouts/smoke_bc_rollouts.npz
```

Collect the default dataset used by training:

```bash
MUJOCO_GL=egl python3 main.py collect \
  --episodes 20 \
  --max-sim-time 5.0 \
  --output outputs/rollouts/sim_vla_rollouts.npz
```

Optionally collect with per-reset F/T biasing enabled:

```bash
MUJOCO_GL=egl python3 main.py collect \
  --episodes 20 \
  --max-sim-time 5.0 \
  --output outputs/rollouts/sim_vla_rollouts_ft_bias.npz \
  --ft-bias
```

Train the basic state-only behavior-cloning policy:

```bash
python3 main.py train \
  --dataset outputs/rollouts/sim_vla_rollouts.npz \
  --epochs 20 \
  --batch-size 64
```

This writes:

```text
outputs/checkpoints/bc_only_states.pt
```

Evaluate the trained checkpoint headlessly:

```bash
MUJOCO_GL=egl python3 main.py eval \
  --checkpoint outputs/checkpoints/bc_only_states.pt \
  --episodes 10 \
  --max-sim-time 5.0
```

If the checkpoint was trained from an `--ft-bias` dataset, evaluate with:

```bash
MUJOCO_GL=egl python3 main.py eval \
  --checkpoint outputs/checkpoints/bc_only_states.pt \
  --episodes 10 \
  --max-sim-time 5.0 \
  --ft-bias
```

Evaluate with the MuJoCo viewer open:

```bash
python3 main.py eval \
  --checkpoint outputs/checkpoints/bc_only_states.pt \
  --episodes 1 \
  --max-sim-time 5.0 \
  --render
```

The current basic BC trainer is state-only: it learns joint deltas from robot
state and does not consume image, language, or color labels. It is useful for
testing the data/training/eval pipeline, but it is not a full task-conditioned
VLA policy yet.

## Choosing Episodes And Epochs

`--episodes` during collection means how many full expert demonstrations to
record. One episode is one reset-to-done rollout: cube starts on the tray, the
expert moves to the selected bin, tips, and returns or times out. More
collection episodes means more training data and more variation across red/blue
tasks, but collection takes longer. For smoke tests, use 2 episodes. For a first
real state-only BC run, try 20-50 episodes. For judging whether the policy is
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

## Current Limitations

This is not yet an OpenVLA fine-tuning pipeline. Right now it is a readable
VLA-shaped simulation scaffold:

- image input from MuJoCo recorded in the dataset, not used by the current
  state-only BC trainer yet
- prompt string input recorded in the dataset, not used by the current
  state-only BC trainer yet
- robot state input: joint positions, joint velocities, force/torque,
  end-effector position, and cube position
- 6D joint-delta action output, converted back to absolute joint targets for
  `env.step()`
- no color/task-conditioning in the current state-only trainer, so it can learn
  to drop cubes but cannot reliably choose the correct color bin
- behavior-cloning training loop
- scripted expert demonstrations

The scripted expert is a first pass. The next important validation step is to run
rollouts, inspect video, and tune bin placement, home pose, tray approach height,
and tilt angle until the expert itself is reliable before training a policy.
