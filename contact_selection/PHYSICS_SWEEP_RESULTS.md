# Box contact sweeps across table friction

Starting from the validated `box_grip` setup, these controlled runs changed
only the table's sliding friction. The controller, geometry filters, feasibility
thresholds, seed, and 25 candidate coordinates stayed the same. The nominal
preset uses table friction 0.5 and finger friction 2.0.

| Table friction | Feasible contacts | Random success | Centre heuristic | Reference |
|---:|---:|---:|---|---|
| 0.5 (nominal) | 25 / 25 | 100% | pass | pass |
| 0.2 | 20 / 25 | 80% | pass | pass |
| 0.15 | 10 / 25 | 40% | fail | pass |

At table friction 0.2, all five failures lie on the far X edge of the top
surface (`x = 0.624 m`). All five have `unintended_collision`; two also have
`unintended_pivot_or_sliding`. Replaying candidate 2 reproduced its failure:
the robot's `ft_and_adapter_link` geom contacted the payload at 8.776 s,
during ARC. The other four collision pairs have not been inspected individually.

At table friction 0.15, the 10 successful contacts all have `x <= 0.561 m`.
Every contact at `x >= 0.574 m` fails. The near-edge reference (`x = 0.536 m`)
passes, while the centre contact (`x = 0.580 m`) slides substantially and fails
several force, joint, and controller checks. This is the first saved scene where
contact selection can beat the existing centre heuristic. Failures can have
multiple reasons; see each JSONL record for their metrics and full phase trace.

The reproducible configurations are
[config/box_grip_table_02.json](config/box_grip_table_02.json) and
[config/box_grip_table_015.json](config/box_grip_table_015.json). The saved
local datasets are `outputs/contact_selection/box_grip_table_02` and
`outputs/contact_selection/box_grip_table_015`. To regenerate one run in a
fresh directory:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/generate_contact_dataset.py \
  --config contact_selection/config/box_grip_table_015.json \
  --output outputs/contact_selection/my_table_015
MPLCONFIGDIR=/tmp/contact-selection-mpl .venv/bin/python \
  scripts/visualize_contact_selection.py outputs/contact_selection/my_table_015
```

A further 16-rollout probe checked contacts 0, 1, 2, and 4 at table
friction 0.14 and 0.16 with nominal mass, and at friction 0.15 with mass and
inertia scaled together to 0.9 and 1.1. In every condition, contacts 0 and 4
passed while contacts 1 and 2 failed. Metrics are saved under
`outputs/contact_selection/boundary_probe.json`. This supports a local boundary
for those sampled contacts; it does not measure stochastic repeatability.

Small three-contact probes bracketed this result. At table friction 0.1, the
reference, centre, and far corner all failed. At 0.17, the reference and
centre passed but the far corner failed. At finger friction 0.2 with table
friction 0.5, all three failed. These probes were not saved as full datasets
and are not evidence of repeatability.

These are useful spatial feasibility boundaries on one simulated box, not enough
for a geometry-general classifier. The same box contact can have different
labels as friction changes. A geometry-only predictor must represent success
probability over a stated property distribution; alternatively, measured or
estimated properties can be added as inputs. Next experiments should repeat
settings near the boundary, vary mass and other geometries, and reserve whole
objects and physical settings for validation. Do not split near-duplicate
contacts across train and test or train a predictor from this box alone.
