# Fixed-strategy combat experiments

```powershell
.ppo-venv/Scripts/python.exe -u scripts/experiment_combat.py --steps 10000 --repeats 2 --eval-runs 30 --n-envs 8 --hours 2
```

All arms start from the same original teacher combat checkpoint. Strategy stays
frozen at the original teacher checkpoint throughout training and evaluation.
Training uses two RNG seeds, with the same initial run-seed allocation across
arms within each repeat. Eight independent workers collect complete runs.
Requested 10,000 decisions round up to 10,240. Training seeds start at 50,000 and
70,000; development evaluation uses the existing fixed 30 seeds at 1,000,000.
These seeds have influenced development; this is not a final untouched test.

| Arm | PPO settings | Teacher anchor | Combat quality reward |
|---|---|---|---|
| control | LR .0003, 10 epochs, no KL stop | None | None |
| retention | LR .00003, 3 epochs, target KL .01 | Coefficient .1 | None |
| quality | Same as control | None | Enabled |
| combined | Same as retention | Coefficient .1 | Enabled |

Gamma remains 1, GAE lambda .95 and entropy coefficient 0. Conservative settings
and anchoring are bundled in this pilot; their individual effects are not isolated.
The teacher term is KL(teacher || student) on current masked rollout states.
The reference policy is frozen, excluded from checkpoint serialization, and
receives no gradients. Checkpoints still load through ordinary MaskablePPO.
The adapted PPO train method retains upstream clipping/early stopping; tests
check identical updates with no anchor and frozen teacher gradients with it.

The optional quality reward is awarded once at actual combat completion, before
frozen noncombat decisions can heal the player: +.01 for victory plus .01 times
HP retained relative to entry HP, clipped to [0,1]. Total extra quality credit is
capped at +.20 per run. No reward is paid for individual attacks, blocks, heals,
or killing summoned enemies. There is no extra quality credit for defeats.
Combined with reward v2, every victory still scores above every defeat. A limited
incentive to heal before ending combat can remain; monitor rounds and HP metrics.

All evaluation uses the original reward-v2 environment without the extra quality
bonus, deterministic CPU inference and identical run limits. Evaluate gameplay
metrics rather than comparing raw training reward across reward variants.

`output/combat_experiments/results.json` contains per-seed evaluation results,
true combat-boundary HP/round metrics, training episode results, reward components,
PPO KL/entropy/value diagnostics, and the strategy-rollout profile summary.
Each arm retains its checkpoint, training report and per-worker error logs.
Rerunning skips fully completed arms; interrupted arms restart from the teacher.
Configuration changes require a separate output directory. A lock prevents
concurrent writers; verify the old process is gone before removing a stale lock.

The profile covers ten single-process CPU strategy rollouts and saves a pstats
file and top cumulative functions. Exclusive-time category grouping is approximate
and cProfile adds overhead; it does not measure parallel-worker IPC overhead.
Simulator/encoder code is held unchanged across the comparison. The default
quality option remains off in the normal alternating pipeline.

## Shared-encoder comparison

`StructuredCombatPolicy` uses categorical card embeddings, shared card/relic/
creature encoders, pooled context and shared card-target/potion-target scorers.
It consumes the same observation v3 and action masks. Tests check hand/enemy
permutation equivariance, checkpoint loading and PPO updates. It has 204,198
parameters versus the flat policy's 2,751,348. The prototype is tested on Silent;
allied-target behavior and other characters need broader validation.

```powershell
.ppo-venv/Scripts/python.exe scripts/pretrain_demonstrations.py --data output/alternating/teacher_snapshot --role combat --architecture structured --seed 42 --epochs 20 --output output/new_structured/combat
.ppo-venv/Scripts/python.exe scripts/evaluate_combat_checkpoint.py --combat output/new_structured/combat.zip --output output/new_structured/evaluation.json --data output/alternating/teacher_snapshot
.ppo-venv/Scripts/python.exe scripts/experiment_combat.py --combat output/new_structured/combat.zip --output output/new_structured_ppo --arms control --skip-profile
```

The architecture pilot compared 20 imitation epochs at two initializations,
100 total epochs at one initialization, and standard PPO from the 20-epoch seed-42
checkpoint using the same two PPO RNG seeds and settings as the flat control.
`--arms` supports subsets for subsequent comparisons. Results and caveats are in
[COMBAT_EXPERIMENT_RESULTS.md](COMBAT_EXPERIMENT_RESULTS.md).
