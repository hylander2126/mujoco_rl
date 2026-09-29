# First-milestone results

Executed with MuJoCo 3.14.0, NumPy 2.5.3, Python 3.12; complete installed
versions are in `outputs/contact_selection/environment.txt`. These are small
diagnostic sweeps, not trained-model evaluation or generalization evidence.

## Unmodified scene pivots

Command:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m contact_selection.generate \
  --output outputs/contact_selection/pilot_v1 --objects 0 10 14 --candidates 9
```

Box: 0/9 feasible. Heart and flashlight: no candidates; their `site:obj_frame`
pivots are at Z=-0.1 m while the bottom collision surface is at Z=0.05 m.
Those are scene-compatibility rejections, not negative contact labels.

The box has location-dependent failure modes. Near-edge points lose contact;
some off-centre points rotate by 5–13° mostly outside the intended axis. The
controller's existing total-angle `tipped` check can call those tips, but the
new label rejects them. Intended world -Y rotation stays below 0.01°.

## Explicit geometric pivot overrides

Command (the checked-in config reproduces the saved resolved configuration):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m contact_selection.generate \
  --config contact_selection/config/pilot_geometry_pivots.json \
  --output outputs/contact_selection/pilot_geometry_pivots
MPLCONFIGDIR=/tmp/contact-selection-mpl .venv/bin/python \
  -m contact_selection.visualize outputs/contact_selection/pilot_geometry_pivots
```

Heart and L use explicit near-X bottom-hull XZ coordinates via the existing
`arc_center_xz` controller option. Box uses the original site. No assets or
control laws changed, and these approximated mesh pivots are recorded in full.

| Object | Candidate rollouts | Feasible | Centre heuristic succeeds |
|-------|--:|--:|----|
| Box   | 9 | 0 | No |
| Heart | 9 | 0 | No |
| L     | 9 | 1 | No |

Across these three scenes, oracle candidate availability is 1/3, exact expected
uniform-random selection success is 1/27, and centre-heuristic success is zero.
These are tiny-sample diagnostics, not probability estimates with useful
statistical confidence. The box's nine rollouts repeat the first sweep and
are not nine additional independent observations.

L candidate 5, world point `(0.625046, -0.009500, 0.199954)` m, is feasible:

* intended tip: 2.157° (threshold 2° inherited from controller);
* ARC ball contact: 100%;
* maximum pivot drift: 2.115 mm;
* maximum off-axis rotation: 1.151°;
* maximum ball contact force: 5.841 N;
* no measured joint, collision, numerical, or controller failure.

L candidate 2 also reaches about 2°, but drifts 129 mm, rotates 14.1° off axis,
and has an unintended collision. It is infeasible. This demonstrates why a
single rotation-magnitude reward would be misleading.

Spatial plots are under each scene's `contacts.png`. In particular:

![L feasibility and execution diagnostics](../outputs/contact_selection/pilot_geometry_pivots/L_0_2810129793/contacts.png)

## Verification and gate decision

18 unit tests pass, covering candidate generation, IK placement, invalid scene
rejection, explicit pivot overrides, features, conjunctive labeling, NaN
rejection, serialization, full-state round-trip and scene-level aggregation.

All nine repeated box trajectories match exactly in the native time, wrench,
object-pose and phase arrays. Independent MJB/state replay of box candidate 0
and successful L candidate 5 also exactly reproduces ball poses and the other
native channels; the L contact remains feasible. Deterministic repeatability is
verified, **not** robustness to physical or sensing perturbations.

There is a location-dependent positive on L, but it is only 0.157° above the
rotation cutoff; raising that provisional outcome requirement to 2.5° removes
the only positive. A nine-point sweep does not establish a robust successful
region. Mass/force perturbation tests and denser local sampling are still needed.

The configured training split contains 18 negatives and **zero positives**.
The sole positive belongs to the held-out validation geometry. There are no
test-object rollouts in this small pilot. The generated summary explicitly
reports `blocked_single_class_training_data`. Moving that held-out candidate
into training would defeat the object-level split.

Therefore logistic regression, MLP training, learned ranking and secondary
information-quality ranking are **not implemented at this milestone**. A
meaningful classifier comparison needs successful interactions on training
objects, repeatable spatial regions across perturbations, and held-out coverage.
No estimator accuracy or information-gain claims are made. The next experiment
should address the nominal controller/scene feasibility and collect both classes
on the training geometries before activating the learning stage.
