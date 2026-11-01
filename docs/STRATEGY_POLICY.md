# Separate strategy policy pilot

`StrategyEnv` exposes only noncombat choices. A frozen combat policy executes
the intervening combats. Both policies are MaskablePPO checkpoints, with separate
observations and action spaces. This is phase routing, not a learned manager.

Strategy schema v1 uses a 128-feature signed hash of persistent inventory and
numeric run state, plus up to 128 individually encoded candidates. Candidate
attributes include IDs, upgrades, prices, event descriptions and map paths three
edges ahead. Hash collisions are possible; this compact pilot representation is
not a lossless state encoding. A small attention layer processes context and
candidates before separate actor/value heads. The combat observation remains the
legacy observation so existing combat checkpoints remain compatible.

Run from the repository root with Python 3.12 and `.[train,dev]` installed:

```powershell
python scripts/pilot_strategy.py --combat-steps 200000 --strategy-steps 20000 --episodes 100 --hours 2
```

The pilot trains a Silent starter-deck Act 1 combat model, freezes it, then trains
strategy across three acts at A0 with sparse terminal rewards, starting at the
map (the default simulator start, without the opening Neow event). No trajectories
or teacher calls are used. Combat training does not yet cover acquired decks,
relics or later acts, which limits interpretation of full-run results. This is
not a matched reproduction of AgenticSTS's live-game benchmark.

The three conditions share held-out evaluation seeds 10000–10099: random both,
trained combat/random strategy, and trained combat/trained strategy. Evaluation
is deterministic for learned policies. JSON files retain individual outcomes,
floors, timeouts and exception tracebacks. Errors remain in the attempted-run
denominator and are reported separately. The training deadline excludes
evaluation and can overshoot by one PPO rollout/update. Use the actual timestep
counts recorded in summary.json, not just the requested counts.

## Real-game bridge

```powershell
python -m sts2_env.bridge.agent_runner --model-path output/strategy_pilot/combat.zip --strategy-model-path output/strategy_pilot/strategy.zip
```

Rebuild the mod first. It now attaches deck, relic, potion, HP and gold context
to decision messages, and route lookahead to map candidates. Python routes
noncombat phases to the strategy checkpoint and translates candidate positions
back to original wire indexes. Missing context fails explicitly. Multi-card
screens rank distinct cards until the minimum required count is satisfied;
optional extra picks are not optimized yet. Without a strategy checkpoint, existing heuristics
remain available.

Simulator and bridge share the hashing encoder and semantic candidate fields.
Shop/card/map normalization is covered by adapter tests, but native event prose
and special reward screens may differ. Live end-to-end parity,
optional multi-card counts, and learned combat observation completeness need further
work before treating real-game win rates as validated. Simulator results are
not real-game results.
