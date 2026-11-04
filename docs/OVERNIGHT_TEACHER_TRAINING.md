# Train on available teacher data

From the repository root in PowerShell:

```powershell
.pilot-venv/Scripts/python.exe -u scripts/train_teachers_overnight.py --hours 10 --epochs 20
```

The command immediately copies a checksum-verified snapshot of the currently
exported dataset and trains combat observation v3 and strategy policies
sequentially. It does not read campaign status or wait for generation. Missing or
invalid datasets fail visibly. Data exported later is used on the next invocation.

The ten-hour limit begins when the script starts preparing the snapshot. It is a maximum, not
a target: each policy gets 20 imitation epochs and the command exits when both
finish. A small teacher dataset should train much sooner; repeatedly fitting it
for ten hours does not add strategic knowledge. This command does not perform
RL refinement or claim an improved win rate.

Progress, checkpoint paths, mapping rejection counts and held-out imitation
agreement are recorded in `output/teachers_overnight_v3/status.json`. Each policy
also has a `training.log` and `model.report.json` beside its `model.zip`.
The terminal prints dataset counts,
per-epoch imitation loss, elapsed training time and saved checkpoint paths.
Trainer stdout/stderr is streamed live and also retained in `training.log`.
Splits remain grouped by source seed/configuration. Validation may be empty
with very few runs. Missing eligible examples fail visibly in the training log.

Each invocation trains both policies for `--epochs` additional epochs, continuing
from their last successful checkpoints even if the dataset is unchanged. Increase
`--epochs` and rerun the same command to train longer. Each pass retains its own
checkpoint and log under `combat/pass-NNNN` and `strategy/pass-NNNN`.
Use `--fresh` to initialize new policies, or a different `--output` directory for
an independent experiment. `status.json` tracks the latest checkpoints and total
completed epochs per policy (older status files begin counting from zero).
The dataset snapshot and model schema metadata remain with the output.

Leave the terminal open. Ctrl+C stops the command; creating
`output/teachers_overnight_v3/STOP` stops it before training starts. Delete that file before
restarting. If forcibly terminated, verify the old runner is gone before removing
a stale `runner.lock`. A training timeout keeps previously completed checkpoints,
but does not save the interrupted policy's in-memory updates.
