# Contact tests with the known-good box setup

The contact-test pipeline now reuses `prepare_box()` from the working demo.
It does not maintain a second copy of its physical or controller settings.
The default generator configuration is `config/box_grip.json`; legacy experiment
configs and existing datasets remain available unchanged.

## Completed sweep

Saved experiment: `outputs/contact_selection/box_grip_25`.

| Measurement | Result |
|---|---:|
| Geometrically valid contacts tested | 25 |
| Feasible interactions | 25 / 25 |
| Intended tipping angle | 10.53–11.24° |
| ARC fingertip contact | 100% for every contact |
| Maximum pivot displacement across trials | 0.632 mm |
| Maximum off-axis rotation across trials | 0.270° |
| Maximum fingertip contact force across trials | 6.111 N |
| Known-good reference contact | Passed |
| Centre-contact heuristic | Passed |

The reference contact is candidate 0, 6 mm inside the near tipping edge.
Its time, wrench, object-pose, ball-pose, and controller-phase arrays exactly
match the standalone successful demo. Every candidate has the same saved
initial-state hash and uses the same physical parameters and controller
configuration, except for the selected contact offset. Feasibility thresholds
were not relaxed.

![Contact outcomes](../outputs/contact_selection/box_grip_25/box_0_481830384/contacts.png)

The blue diamond identifies the demo reference. All points are green because
all tested points actually passed. The other panels show continuous execution
metrics rather than manufacturing binary failures.

## Reproduce

Choose a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/generate_contact_dataset.py \
  --output outputs/contact_selection/my_box_sweep --candidates 25
MPLCONFIGDIR=/tmp/contact-selection-mpl .venv/bin/python \
  scripts/visualize_contact_selection.py outputs/contact_selection/my_box_sweep
```

The named config can also be provided explicitly:
`--config contact_selection/config/box_grip.json`.

Replay the reference using its saved model, state, and controller settings:

```bash
.venv/bin/python scripts/visualize_contact_selection.py \
  outputs/contact_selection/box_grip_25 \
  --replay-scene box_0_481830384 --candidate 0 --show-viewer
```

Scene manifests and rollout records identify the resolved preset, reference
contact, friction, contact priority, solver settings, and controller parameters.
The compiled MJB snapshot preserves these settings for replay. The reference
point is subjected to the same geometry and reachability filters as other points.

36 tests passed, including preset/demo parity, reference filtering, shared reset
state, metadata serialization, and existing controller/evaluator tests.

## Interpretation

This verifies integration and stable interactions for the sampled contacts on
one nominal box under the explicitly grippy simulation preset. It is not proof
that every possible contact succeeds, nor calibration to hardware.

This sweep has only positive labels. Random selection, the centre heuristic,
and the candidate-set oracle all succeed in this single scene, so there is no
observed selection advantage to learn here. The report correctly retains the
single-class training gate; no classifier or information ranking was trained.
Future experiments can vary physically justified object or interaction conditions
to investigate feasibility boundaries, while keeping this reference as a control.
