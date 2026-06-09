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

HW1 does not use teleoperation yet. Instead, `controllers/HW1BinSortExpert`
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
- `controllers/`: HW1 scripted expert state machine for demo generation.
- `../controllers/robot.py`: shared IRB120 robot wrapper, IK, force/torque
  helpers, and the simple position actuator controller used by HW1.
- `configs/`: sim, data, training, and domain-randomization defaults.
- `scripts/`: collect, train, and evaluate entry points.
- `models/`: starter VLA-shaped policy.
- `data/`: `.npz` dataset loader.

The old RL sandbox, phase-controller experiments, object-library loader, and
checked-in generated XML were removed so this repo can stay focused on the VLA
homework path.

## Current Commands

Run from the repo root with `PYTHONPATH=src` unless the package is installed in
editable mode.

You can also run the direct script form from `src/mujoco_irb120`, which is handy
while iterating:

```bash
python3 VLA/main.py --episodes 2 --max-sim-time 2.0 --render
```

Collect a tiny starter dataset:

```bash
PYTHONPATH=src python3 -m mujoco_irb120.VLA.main collect --episodes 2 --max-sim-time 2.0
```

Train the starter behavior-cloning policy:

```bash
PYTHONPATH=src python3 -m mujoco_irb120.VLA.main train --epochs 5
```

Evaluate a checkpoint:

```bash
PYTHONPATH=src python3 -m mujoco_irb120.VLA.main eval --checkpoint data/vla/checkpoints/vla_bc.pt --render
```

## Current Limitations

This is not yet an OpenVLA fine-tuning pipeline. Right now it is a readable
VLA-shaped simulation scaffold:

- image input from MuJoCo
- prompt string input
- robot state input
- 6D joint action output
- behavior-cloning training loop
- scripted expert demonstrations

The scripted expert is a first pass. The next important validation step is to run
rollouts, inspect video, and tune bin placement, home pose, tray approach height,
and tilt angle until the expert itself is reliable before training a policy.
