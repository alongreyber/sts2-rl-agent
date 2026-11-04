# Fixed-strategy experiment results

All planned runs finished. Original checkpoints and experiments were preserved;
no candidate was promoted or deployed. Strategy was frozen at the teacher policy.
Each PPO recipe used two training RNG seeds, 10,240 decisions per fit and the
same 30 development-evaluation run seeds. Ten PPO fits collected 102,400 decisions.

## Full-run validation

| Policy/recipe | Mean floor | Rounds/combat | HP lost per won fight |
|---|---:|---:|---:|
| Original teacher baseline | 6.47 | 4.60 | 12.42 |
| Flat PPO control | 6.25 | 6.54 | 12.02 |
| Flat conservative PPO + teacher KL | 6.47 | 4.66 | 12.52 |
| Flat quality reward | 6.32 | 5.62 | 12.06 |
| Flat conservative + KL + quality | 6.47 | 4.67 | 12.61 |
| Structured encoder + standard PPO | 6.87 | 5.23 | 11.74 |

Every evaluated policy had zero victories and zero Act 2 reaches. Combat quality
is measured at actual fight completion before automatic strategy actions/healing.
The HP/round comparisons include different reached encounters, so they are not
pure matched-state causal effects. The quality term was active and nonzero in
training; reward bounds and frozen-teacher gradients passed tests.

Conservative settings preserved behavior but did not improve progression. At this
budget their teacher KL stayed tiny (about 0.0004–0.0008 in the last training
minibatch), so these settings may simply move the actor too little. The conservative
recipe also changes value-learning speed; its separate ingredients are not isolated.
The tested quality bonus did not establish an improvement and should not become
the default based on this pilot.

The structured model outperformed the flat PPO control in both training repeats:
6.90 vs 6.23 and 6.83 vs 6.27. Average paired difference is +0.62 floors; an
exploratory paired-bootstrap 95% interval over development run seeds is +0.27 to
+1.08. Against the original teacher baseline the difference is only +0.40, with
interval -0.03 to +0.90. Thus this is a promising representation, not established
better-than-teacher full-run play. Intervals condition on these two PPO fits and
do not represent all training uncertainty. Both fits within an architecture start
from the same warm-start checkpoint. Development seeds have been reused and
must not be treated as an untouched final test set.

## Imitation architecture ablation

| Model | Imitation epochs | Exact teacher agreement | Mean validation floor |
|---|---:|---:|---:|
| Flat original | 20 | 39.6% | 6.47 |
| Flat additional-imitation diagnostic | 100 total | 46.8% | 6.37 |
| Structured, initialization 42 | 20 | 62.1% | 6.63 |
| Structured, initialization 43 | 20 | 61.2% | 6.77 |
| Structured, initialization 42 | 100 total | 73.7% | 6.57 |

Agreement is measured on the 1,934 training examples, not held-out teacher data.
Exact labels distinguish interchangeable card copies. Structured models share
action scoring across slots; that constraint was tested explicitly. The original
flat initialization was not reproducibly seeded, so warm-start comparisons do not
isolate architecture from all initialization effects. The structured model has
204,198 parameters versus 2,751,348 for the flat model. Better data fit and faster
fights have not yet produced meaningful full-run wins.

## Throughput profile

Ten fixed CPU strategy runs with the original policies took 6.21 seconds under
coarse, nonoverlapping section timers:

| Section | Seconds | Share |
|---|---:|---:|
| Combat observation encoding | 2.95 | 47% |
| Map generation | 2.18 | 35% |
| Frozen combat inference | 0.47 | 8% |
| Strategy inference | 0.11 | 2% |
| Other simulator/routing work | 0.51 | 8% |

cProfile separately identifies duplicate-path pruning and repeated empty-card
encoding as hot functions. Its overhead is substantial, so the section timers
are preferable for approximate shares. These are CPU rollouts of the baseline,
not parallel-worker or post-PPO timings; no IPC or GPU-update overhead is measured.
Encoding and map pruning are concrete optimization targets. No simulator or
encoder implementation was changed during these experiments.

## Next steps

1. Keep the structured representation as the leading candidate. Confirm it on
   fresh development seeds before any promotion; preserve seeds >=2,000,000 for
   a final test. Neither more imitation nor the tested reward tweak alone solved
   progression.
2. Test structured-policy PPO with separate critic warm-up or faster critic/
   slower actor updates. The actor was stabilized by the conservative recipe,
   but that recipe also slowed the initially untrained critic. Isolate this
   change rather than combining more knobs at once.
3. If longer training is justified by fresh validation, keep strategy frozen and
   evaluate in short blocks. Continue act-exposure and combat-boundary diagnostics.
4. For throughput, optimize empty observation slots and repeated map-segment work,
   requiring bit-identical observations and deterministic map/RNG parity tests.

Raw artifacts:

- `output/combat_experiments/results.json` and `summary.json`: eight flat-policy
  fits and baseline, per-run results, rewards and PPO diagnostics.
- `output/structured_ppo_experiments/results.json`: two structured PPO fits.
- `output/combat_experiments/structured_*/evaluation.json`: imitation comparisons.
- `output/combat_experiments/architecture_comparison.json`: paired intervals.
- `output/combat_experiments/strategy_profile.*` and `section_timings.json`:
  profiling evidence.
