# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MuJoCo-based robot learning system where an ABB IRB120 6-DOF manipulator executes autonomous push/tip manipulation sequences on objects to estimate their physical properties (mass, center of mass, friction) from force/torque sensor data.

## Setup

```bash
pip install -e .   # install package in development mode (from repo root)
```

Key dependencies: `mujoco`, `numpy`, `scipy`, `trimesh`, `mediapy` (optional, for video recording).

## Running the Simulation

Main entry point:
```bash
cd src/mujoco_irb120/scripts/phase-state-machine/
python simulation_multiphase.py
```

Key configuration variables at the top of `simulation_multiphase.py`:
- `OBJECT`: object ID (0=box, 10=heart, 11=L-shape, 14=flashlight)
- `VIZ`: 1 to open MuJoCo viewer, 0 for headless
- `RECORD_VIDEO`: save MP4 output
- `CONTROLLER_TYPE`: `"position"` or `"velocity"`
- `START_PHASE`: skip directly to a specific phase for debugging

## Push Selection Pipeline

```bash
cd src/mujoco_irb120/push_selection/
python run_push_selection.py [--loa-epsilon 0.02] [--top-k 3] [--output-dir ../figures]
```

Programmatic usage:
```python
from push_selection_pipeline import select_push_config, visualize_ranked_pairs
ranked = select_push_config(mesh, com_2d, verbose=True)
visualize_ranked_pairs(mesh, ranked, com_2d, top_n=3, save_png_path="output.png")
```

## Architecture

### Data Flow

```
simulation_multiphase.py
  → load_environment(object_id)   # generates gen_main.xml dynamically from templates
  → Robot.__init__()              # FK/IK setup, F/T sensor biasing
  → StateMachine loop:
      SCAN → APPROACH_PUSH → PUSH → RETREAT → DESCEND → SQUASH → PULL_TIP → RETURN_PRE_SQUASH → DONE
  → pc.record()                   # logs t, wrench, quat, phase to history arrays
  → mujoco.mj_step()              # advances physics
  → pc.save("simulation_data_multiphase.npz")  # final data dump
```

### Component Responsibilities

**`controllers/robot.py`** — Core `Robot` class wrapping the MuJoCo model:
- Forward/inverse kinematics (damped least-squares IK, 3 variants)
- Jacobian computations for Cartesian control
- F/T sensor biasing and gravity compensation
- Contact detection and topple detection
- Payload pose estimation

**`controllers/state_machine.py`** — `StateMachine` class driving the full manipulation sequence (~140 tunable parameters for speeds, forces, tolerances). Key abort conditions: push force > 20 N, tip angle > 60°.

**`controllers/phase_controllers.py`** — Two implementations sharing the same phase interface:
- `PositionPhaseController`: accumulated position targets, KP=200/100
- `VelocityPhaseController`: direct velocity commands with gentler gains

**`push_selection/push_selection_pipeline.py`** — Geometry-driven push point optimizer:
- Extracts support polygon, tip edges, and push faces from mesh
- Pairs edges to faces with parallel normals (opposite sides of object)
- Scores candidates: `s_orth + w_tip·s_tip + w_lev·s_lev + w_edge·s_edge + w_loa·s_loa`

**`formulation/com_estimation.py`** — Wrench transformation using Adjoint frame transforms; maps F/T sensor readings to object-frame applied forces for inverse-dynamics CoM/mass estimation.

**`util/load_obj_in_env.py`** — Generates `gen_main.xml` at runtime by compositing robot + object XML snippets. Supports 15 configurable object poses and both actuator types.

**`util/helper_fns.py`** — Modern Robotics library wrappers plus custom helpers: quaternion continuity enforcement, screw-theory rotation conversions, Adjoint matrix computations.

### Asset Configuration

**`assets/object_params.json`** — Ground truth for the 4 main experiment objects:
```json
{ "name": "box", "mass_gt": 0.635, "com_gt_onshape": [0.05, 0, 0.15],
  "init_xyz": [0.4, 0, 0.4], "theta_star": 18.435 }
```

**Scene XMLs:**
- `assets/main.xml` — Template scene (robot + object includes)
- `assets/common_modified.xml` — Shared geometry/friction definitions
- `assets/gen_main.xml` — Auto-generated at runtime; do not edit manually

Object mesh files live in `assets/my_objects/` (6 main experiment objects) and `assets/object_sim/` (50+ ShapeNet-derived models).
