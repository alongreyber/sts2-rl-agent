# Alternating full-run training

Run from the repository root:

```powershell
.ppo-venv/Scripts/python.exe -u scripts/train_alternating.py --hours 10 --teacher-epochs 20 --cycles 3 --combat-steps 10000 --strategy-steps 5000 --eval-runs 30
```

The pipeline snapshots currently exported teacher data, warm-starts both policies,
evaluates that baseline, then alternates combat PPO and strategy PPO. Combat PPO
episodes span full runs: a frozen strategy policy navigates every noncombat
decision, and PPO sees only combat transitions. Strategy PPO freezes the updated
combat checkpoint and collects only strategy transitions. Combat policy weights
are not updated during strategy learning, or vice versa. Both use the Python
simulator; native teacher collection remains in sts2-ml. This retains the existing
MLP/strategy-attention architectures and combat observation v3.

## Hardware and throughput

PPO now uses independent subprocess simulators (up to eight by default), batched
learner inference and updates, and automatic CUDA selection when available.
Defaults are one Torch thread per worker, four learner threads, 128 rollout steps
per worker and minibatches of 256. Eight workers therefore collect 1,024 decisions
per rollout. `phase_steps` and `fps` in stdout report aggregate decision throughput,
including worker startup and policy update time. They are not primitive game FPS.
Seed allocation is shared and locked across processes; worker resets cannot
duplicate seeds or enter the validation range. Opponent policies stay on CPU in
each worker; only the learner uses CUDA. Validation remains sequential on CPU.

On this machine, `.pilot-venv` has CPU-only Torch. The separate `.ppo-venv` has
CUDA Torch and references `.pilot-venv` for the existing simulator/SB3 packages.
Both directories are required. Recreate it using:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_ppo_cuda.ps1
```

Use `--n-envs`, `--device cpu|cuda|auto`, `--torch-threads`, `--worker-threads`,
`--rollout-steps` and `--batch-size` to tune resources. Resource settings may change
on resume. Restart an existing trainer to load these changes; checkpoints remain
compatible. CPU-only use still works with `.pilot-venv` and `--device cpu`.
GPU utilization may stay modest because Python simulation and small-batch rollout
inference dominate. Compare decision throughput, not utilization alone. More
workers/threads can hurt performance through contention; eight workers target
the eight physical cores on the development machine. Changing vectorization also
changes PPO's rollout size and stochastic training sequence.

A short local benchmark with capped 100-step runs measured 121 → 185 combat
decisions/sec and 16 → 28 strategy decisions/sec, comparing one CPU worker with
eight workers plus CUDA. Both started from the same checkpoints; rollout batching
and stochastic trajectories differed. These figures include process startup and
updates, and do not predict long-run win rates or guarantee the same speedup on
uncapped runs. Raw measurements are in `output/resource_benchmark/report.json`.

## Corrected reward (version 2)

Both PPO stages now use `gamma=1.0` and the same bounded reward ledger:

- Terminal victory +1, defeat -1.
- A once-only bonus for each newly reached floor: +0.25/60, capped at +0.25
  for the run. The constant 60 is a normalization budget, not an assertion that
  every run has exactly 60 floors. Returning to a previously seen floor pays zero.
- -0.001 per completed combat round, capped at -0.10 over the run. Playing cards
  and confirming selections do not themselves incur this cost. A lethal enemy
  turn still counts even if the round counter never increments.

The bounded terms guarantee every victory scores above every defeat. Removing
terminal-loss discounting removes the incentive to delay a loss; floor progress
then distinguishes losing runs. At the round-cost cap, additional stalling is
neutral rather than profitable. No rewards are given directly for block, healing,
card plays or damage. Strategy steps aggregate progress and round costs across
all frozen combat actions; combat steps aggregate frozen noncombat progress.

A configured game-step limit is a terminal loss (`terminated=True`,
`truncated=False`, `info['timeout']=True`), preventing PPO from bootstrapping a
continuation value. The existing combat-turn cap is also a game loss. Wall-clock
training interruption stops collection without emitting a fabricated terminal
reward. `reward_components` exposes progress, round cost and outcome separately.

Defaults are shorter phases: 10,000 combat and 5,000 strategy decisions. The
pipeline now evaluates after combat PPO as well as after strategy PPO; these
reports do not automatically promote or roll back policies. PPO discounting
still counts active-policy decisions; GAE lambda remains 0.95 and needs empirical
assessment. No later-act starting-state curriculum or teacher anchoring is added.

Corrected experiments default to `output/alternating_rewards_v2`. Reward settings
are recorded in configuration; old sparse-reward experiments cannot be resumed
into this version. To retain the exact teacher baseline from the first experiment:

```powershell
.ppo-venv/Scripts/python.exe -u scripts/train_alternating.py --hours 10 --initial-combat output/alternating/teacher_combat.zip --initial-strategy output/alternating/teacher_strategy.zip
```

These flags copy starting weights into the new output, and do not overwrite the
source models. Reruns resume the new experiment's latest weights. A new experiment
without initial checkpoints still performs imitation from current demonstrations.

Training run seeds are allocated below 1,000,000. Evaluation uses exact fixed
seeds starting at 1,000,000, deterministic policies, and the same episode limits
for baseline and subsequent cycles. It reports win rate with a 95% Wilson
interval, floors reached, acts reached, truncations, and per-run details. Keep
seeds at 2,000,000 and above unused for a later final test. Validation influences
development, so it is not an untouched final test set. Integration exceptions
stop the pipeline; they are not converted to losses.

`output/alternating_rewards_v2/pipeline.json` records the baseline, phase checkpoints,
training exposure, and each cycle's validation. No automatic promotion is made:
compare validation against the teacher baseline before deploying new policies.
All previous phase checkpoints remain in numbered cycle directories. PPO rollout
length can round requested step counts upward.

Rerunning the same command with the same output resumes the latest saved policies
and runs additional cycles. Teacher imitation and baseline evaluation are skipped
once saved. Use another `--output` directory for an independent experiment.
Character, evaluation seed count, reward version and run-step limit must match on resume.
`--hours` covers the whole invocation; the limit is checked during PPO steps and
evaluation, so an individual simulator operation may run past it. A timed-out PPO
phase saves its current policy and is marked time-limited; the next invocation
continues with the following phase. An interrupted evaluation is repeated.
An interrupted, unsaved phase restarts from the previous checkpoint.

The standalone `train_teachers_overnight.py` remains imitation-only. This pipeline
does not spend Fireworks credit or generate additional teacher demonstrations.

Smoke check (short episode caps verify integration, not strength):

```powershell
.pilot-venv/Scripts/python.exe -u scripts/train_alternating.py --output output/alternating_rewards_v2_smoke --teacher-epochs 1 --cycles 1 --combat-steps 32 --strategy-steps 16 --rollout-steps 16 --eval-runs 1 --max-run-steps 100 --hours 0.05
```
