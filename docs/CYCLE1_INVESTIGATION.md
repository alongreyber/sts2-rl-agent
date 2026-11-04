# Cycle 1 regression investigation

Evaluated all policy combinations on the original 30 validation seeds, with the
same episode limits and deterministic actions. No new training was performed on
the original checkpoints. Short cycle-2 reproduction used a separate output.

| Combat | Strategy | Mean floor | Wins | Act 2 reached |
|---|---|---:|---:|---:|
| Teacher | Teacher | 6.47 | 0/30 | 0/30 |
| PPO | Teacher | 2.70 | 0/30 | 0/30 |
| Teacher | PPO | 6.63 | 0/30 | 0/30 |
| PPO | PPO | 2.50 | 0/30 | 0/30 |

Combat is the principal regression. The strategy-only difference is small and
does not demonstrate improvement in win rate. All checkpoint parameters are
finite; there is no evidence of NaN corruption.

## Reward incentive

The implemented full-run combat objective gives only terminal +1/-1 and uses
gamma=0.99. All training runs lost. On an all-losing trajectory, terminal return
from the first decision is `-gamma^(N-1)`, which becomes less negative as the
number of decisions increases. Thus the objective rewards delaying death,
including spending more decisions within the same encounter. It need not reward
reaching higher floors.

With teacher strategy fixed, validation shows:

| Combat | Mean combat decisions | Mean discounted terminal return | Mean floor |
|---|---:|---:|---:|
| Teacher | 106.6 | -0.356 | 6.47 |
| PPO | 352.5 | -0.084 | 2.70 |

These are exact Monte Carlo returns for the configured terminal objective on
these deterministic validation trajectories, not estimates of the learned value
function. The worse gameplay scores better under that objective. This establishes
an undesirable incentive and is consistent with the observed behavior; it does
not isolate the causal effect of each PPO hyperparameter.

## Behavior drift

Across ten validation seeds with frozen teacher strategy, attack cards comprised
61.0% of teacher-policy card plays versus 11.1% after PPO (including played status
cards in the denominator, excluding selection/end-turn actions). The PPO policy
used Defend and Survivor much more frequently. These are different trajectories,
so the percentages describe behavior rather than a matched-state causal test.
Opening-state probes also show reduced action entropy. This is not an immediate
end-turn collapse: mean action-zero probability on teacher observations decreased.

On the original 1,934 eligible combat teacher examples, deterministic agreement
dropped 39.6% → 24.1%; mean teacher-action negative log likelihood rose 1.47 → 2.91.
These are training-set diagnostics, not independent validation scores. Strategy
agreement changed 37.2% → 33.9% on its 339 examples.

PPO used learning rate 0.0003, gamma 0.99, GAE lambda 0.95, entropy coefficient 0,
and no target KL. There is no teacher-rehearsal/teacher-KL term. Drift from the warm
start is therefore unconstrained beyond ordinary PPO clipping. Sparse losses,
short-horizon GAE credit and these update settings are additional concerns, but
their individual contributions have not been experimentally separated.

## Cycle-2 failure

The original state contains `status=failed` and an empty error string, with no
durable traceback. Its cause cannot be recovered reliably from that artifact.
The same checkpoints and starting training seed successfully completed short
eight-worker CUDA combat/strategy phases and validation in a separate diagnostic
run. A short reproduction does not exclude an intermittent or later failure.

The pipeline now records exception class, full traceback and `last-error.log`.
Each subprocess writes a separate stderr/fault log in its cycle directory. No
speculative simulator fix or long training restart was applied.

## Before another long training run

1. Start from the teacher checkpoints, retaining cycle 1 as evidence.
2. Correct the stalling incentive. Undiscounted finite-run returns remove the
   benefit of merely delaying a loss, but all losses then provide the same total
   return. Add bounded, non-repeatable progress feedback and combat-quality
   feedback; check for new exploits. Pure potential shaping with the old gamma
   preserves the old discounted objective and alone does not remove this flaw.
3. Use shorter combat phases with intermediate held-out evaluation, and retain
   prior checkpoints when they regress. Do not wait until strategy training to
   evaluate a combat update.
4. Measure teacher drift and consider teacher replay or a teacher-policy KL
   anchor alongside PPO. Smaller updates/KL stopping may help but do not fix the
   reward incentive by themselves.
5. Track later-act exposure; neither the baseline nor cycle 1 reached Act 2 here.

Raw diagnostic scripts/results are in `output/cycle1_investigation`: ablations,
opening-action probes, behavior counts, and the separate reproduction. The
original `output/alternating` checkpoints and training state were preserved.
