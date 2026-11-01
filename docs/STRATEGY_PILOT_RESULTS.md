# Strategy pilot baseline (2026-11-01)

No demonstration trajectories or LLM calls. Silent, A0, three simulator acts,
default map start without Neow. Evaluation seeds 10000-10099 were held out from
training. Learned policies were evaluated deterministically.

| Condition | Full-run wins | Mean floor | Timeouts | Exceptions |
|---|---:|---:|---:|---:|
| Random combat + random strategy | 0/100 | 3.19 | 0 | 0 |
| Trained combat + random strategy | 0/100 | 8.01 | 7 | 0 |
| Trained combat + trained strategy | 0/100 | 9.99 | 11 | 0 |

All attempts, including timeouts, remain in the win-rate denominator.
The Wilson 95% interval for each 0/100 result is approximately 0-3.7%.
The floor increase is descriptive; a single training seed does not establish
that the architecture or training method improves win rate.

Combat: 200,192 PPO steps on Silent starter-deck Act 1 encounters, 70 HP.
Strategy: 20,224 PPO decisions with the combat checkpoint frozen, using sparse
terminal rewards. CPU only. The final strategy training/evaluation run took
564 seconds, excluding the preceding combat-training run.
The saved combat checkpoint separately won 74/100 starter-deck combats on
evaluation seeds 20000-20099, with a 2,000-step cap per combat.

Results and both checkpoints are saved under `output/strategy_baseline_final/`:
`summary.json`, per-condition per-seed JSON, `combat_eval.json`,
`environment.json` (package versions and source hashes), `combat.zip`, and
`strategy.zip`. Outputs are intentionally gitignored.

Reproduce with the commands in [STRATEGY_POLICY.md](STRATEGY_POLICY.md).
Use `--combat-path` to reuse the saved combat checkpoint when repeating strategy
experiments, and keep the evaluation seed set reserved for final comparisons.

Validation: 109 targeted regression tests passed; the latest seven strategy
tests also passed after adding map-parity coverage. C# compilation passed.
Full mod packaging could not complete because the configured Godot executable
is absent; no live-game evaluation or end-to-end bridge validation was run.
Special reward/event metadata and optional multi-card selection counts still
need live parity checks. Combat retains the legacy compact observation.

The next experiments should address combat coverage across acquired decks and
later acts, investigate timeouts, and compare a curriculum or teacher warm-start
against this sparse-reward baseline. This is not a matched AgenticSTS comparison.
