# Starter combat Expert Iteration

Run from the repository root using the Python environment containing the project dependencies.

```powershell
python -m scripts.expert_iteration --character Ironclad --pool both --rounds 2 --train-seeds 4 --eval-seeds 10 --samples 2 --output output/experts/ironclad.npz
python -m scripts.expert_iteration --character Ironclad --pool both --checkpoint output/experts/ironclad.npz --eval-start 100 --eval-seeds 100 --samples 2
```

Supported characters are Ironclad, Silent, Defect, Regent and Necrobinder.
Each fixture starts at full HP with its character's starter deck and relic,
an empty potion inventory, and default ascension. Both pools comprise the eight
weak encounters across Overgrowth and Underdocks.

The teacher evaluates every legal action with the same sampled hidden draw
orders, rolls out to termination, and scores victory, remaining HP and turns.
The apprentice learns a masked distribution over those scores. Later iterations
use the apprentice for rollouts and aggregate examples from earlier iterations.
This implementation uses a small NumPy MLP and flat Monte Carlo search.

Search forks reconstruct the environment from its seed and action history.
Deep copying is unsafe because monster callbacks close over their original
creatures. The regression tests check root isolation, replay equivalence and
hidden draw-order invariance.

## Initial evaluation

Two iterations, four training seeds per iteration per encounter, two rollouts
per action, and ten separate evaluation seeds per encounter produced the
following results. Each character therefore has 80 evaluation fights.

| Character | Heuristic | Initial search | Apprentice | Apprentice search |
|---|---:|---:|---:|---:|
| Ironclad | 80/80 | 80/80 | 80/80 | 80/80 |
| Silent | 80/80 | 80/80 | 80/80 | 80/80 |
| Defect | 80/80 | 80/80 | 80/80 | 80/80 |
| Regent | 80/80 | 80/80 | 80/80 | 80/80 |
| Necrobinder | 80/80 | 80/80 | 80/80 | 80/80 |

These small per-encounter samples establish integration and preliminary coverage.
They do not establish a near-perfect underlying win rate for every pairing.
The heuristic also wins this set, so the win counts alone do not establish an
advantage from iteration. A separate 24-fight Ironclad check on seeds 100–102
showed higher remaining HP with search. The CLI reports HP and timeouts to
measure improvements even when win rates saturate.

## Current limits

The apprentice now uses a rich public combat observation with registered powers,
absolute HP, unordered pile composition, choice options, orbs, relic state,
modifiers, history and allies. The compact encoder used for the initial
evaluation above has been replaced. The MLP architecture remains unchanged.
Checkpoints carry an observation layout fingerprint and incompatible older
checkpoints are rejected. NPZ training
data contains encoded features and targets, rather than re-encodable public
snapshots. Search samples the current hidden draw order but retains seeded
future RNG and enemy AI state. Its results are therefore simulator benchmarks;
a public-information belief model and live parity checks remain necessary.

Training artifacts include model weights, examples, character, schema version
and training seed boundary. Checkpoint evaluation requires an explicit held-out
seed range. Replay forks favor correctness over speed and currently support
single-process search only because card instance allocation is process-global.

Previous PPO, strategy, demonstration and v2/v3 implementations are preserved in
commit `fbcaeca` on `codex/archive-ppo-strategy`.
