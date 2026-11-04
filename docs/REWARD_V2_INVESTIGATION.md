# Reward-v2 experiment investigation

The run completed three cycles successfully. Each cycle collected 10,240 combat
and 5,120 strategy PPO decisions. The original experiment and checkpoints remain
unchanged; diagnostic work is in `output/reward_v2_investigation`.

## Held-out simulator results

All evaluations use the original 30 fixed seeds and the same run limits.

| Policy pair | Mean floor | Highest floor | Wins | Reached Act 2 |
|---|---:|---:|---:|---:|
| Teacher baseline | 6.47 | 13 | 0/30 | 0/30 |
| Cycle 1 after combat | 5.90 | 12 | 0/30 | 0/30 |
| Cycle 1 after strategy | 6.57 | 13 | 0/30 | 0/30 |
| Cycle 2 after combat | 6.03 | 11 | 0/30 | 0/30 |
| Cycle 2 after strategy | 6.20 | 11 | 0/30 | 0/30 |
| Cycle 3 after combat | 6.53 | 11 | 0/30 | 0/30 |
| Cycle 3 after strategy | 6.93 | 11 | 0/30 | 0/30 |

After-combat evaluations use that cycle's frozen opponent, not always the original
teacher strategy. They are not a factorial ablation of both policies.

Cycle 3 versus baseline improved on 11 seeds, tied on nine, and regressed on ten.
Mean paired floor difference is +0.47, with an exploratory paired-bootstrap 95%
interval of -0.37 to +1.37 (10,000 resamples). This does not establish improvement.
Zero wins in 30 trials cannot distinguish low win rates reliably.

Training completed 188 combat-policy run episodes and 1,069 strategy-policy run
episodes, with no victories and no Act 2 reaches. Every combat PPO action was in
Act 1. Later-act teacher rows exist, but later-act PPO experience remains absent.

## Combat quality and drift

Paired first-encounter diagnostics freeze the original teacher strategy and start
from the same 30 validation seeds. Both combat checkpoints won 30/30 opening
fights. The teacher took 4.07 rounds on average; cycle 3 took 10.77. Reported HP
after the environment's automatic postfight strategy advancement averaged 60.27
versus 54.33. This HP measurement can include postfight effects, so it is not a
pure damage-received measurement. Opponent strategy is identical and the fights
start from matched pre-combat conditions.

Attack-card share of played cards in those first encounters fell from 65.9% to
30.2% (including played statuses, excluding selections and end turns). Defend-like
skills and selections increased markedly. This demonstrates less efficient
opening combat, even though first-fight victory rate has already saturated.

Teacher combat agreement on 1,934 training examples:

| Checkpoint | Exact agreement | Allowing interchangeable hand-card copies |
|---|---:|---:|
| Teacher warm start | 39.6% | 44.3% |
| Cycle 1 | 32.4% | 38.7% |
| Cycle 2 | 23.7% | 27.6% |
| Cycle 3 | 21.9% | 26.2% |

The equivalence diagnostic compares observed identity/cost/modifier features and
target; it is not a proof of complete semantic equivalence. These are training-set
diagnostics, not independent validation. Strategy agreement similarly fell from
37.2% to 29.2% on 339 examples. Drift need not be bad by itself, but the opening
combat results show that it has not consistently produced better behavior.

## Extra imitation ablation

An independent copy of the original combat teacher checkpoint received 80 more
imitation epochs (approximately 30 seconds of training), with original strategy
frozen for evaluation. Exact teacher agreement rose to 46.8%, but mean validation
floor was 6.37, with zero wins. Merely extending imitation did not demonstrate
better play. The original policy was not overwritten or promoted.

## Likely remaining limitations

- Correcting discounted terminal loss removed its explicit delay incentive, but
  did not automatically make combat efficient. Progress feedback is only 0.25/60
  per floor and round cost 0.001; HP preservation has no immediate reward.
- Full-run victory feedback is absent in training. GAE lambda remains 0.95;
  distant consequences of early HP loss receive weak direct credit through GAE.
  Gamma=1 alone does not give long-range low-variance credit assignment.
- PPO still uses learning rate 0.0003, default ten optimizer epochs, no target KL,
  no entropy bonus and no teacher replay/KL anchor. Useful warm-start behavior is
  not explicitly protected. The relative effect of each setting is unmeasured.
- The warm start is weak even on its own examples. The flat 21,370-input MLP has
  a 64-unit bottleneck and no shared card/action encoder. Sparse demonstrations,
  slot dependence and observation differences between native teacher collection
  and Python rollout remain plausible contributors, not proven sole causes.
- Updating strategy changes the distribution seen by combat. Both policies are
  weak, making simultaneous development harder to diagnose.

## Recommended next experiments

1. Freeze the original strategy. Compare short combat updates from the same
   teacher checkpoint on identical training/validation seeds. Log combat win
   rate, HP loss at the combat boundary, rounds, PPO KL/entropy/value loss, reward
   components and teacher drift. Do not use overall floor alone as a diagnostic.
2. Test conservative PPO plus teacher retention: smaller learning rate (e.g.
   0.00003), fewer optimizer epochs, target KL (e.g. 0.01), and a teacher replay or
   teacher-policy KL term. These are candidate settings, not validated fixes.
3. Separately test bounded, once-per-combat quality feedback for actual victory
   and HP preservation, with terminal win/loss remaining dominant. Avoid per-card
   block/damage rewards and healing loops. Compare reward change independently
   of teacher anchoring before combining them.
4. Prototype the shared card/relic/enemy encoder and action scoring discussed
   earlier. Compare both imitation and simulator play; more raw features or more
   epochs are not evidence of useful representation learning.
5. Resume strategy alternation once combat improves on the fixed-opponent panel.
   Continue reporting act exposure. If natural progression still prevents later
   acts, revisit start-state curricula instead of assuming more cycles suffice.

The existing 30 seeds have now informed several development decisions. Keep a
separate untouched final seed set for confirming the eventual selected model.
