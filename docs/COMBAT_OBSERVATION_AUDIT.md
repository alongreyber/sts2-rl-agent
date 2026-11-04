# Combat observation audit

## Current schema: v3

New combat training and teacher import default to **21,370 floats**. Versions 1
(217) and 2 (7,019) remain available and unchanged. The environment's direct
constructor still defaults to legacy v1 for existing clients; request
`STS2CombatEnv(observation_version=3)` explicitly. The live adapter and frozen
combat policy inside strategy rollouts select a schema by checkpoint shape.
Existing weights cannot be loaded into a different-sized schema.

V3 appends these features to the complete v2 prefix:

- Per-relic display counter and its availability, stack, melted state, visible
  status, and explicitly whitelisted usage flags with availability bits.
- Per-hand-card and per-selection-option enchantment/affliction identity and
  amount; current base dynamic variables, exhaust-on-next-play and keyword flags.
  The fixed vocabularies contain 23 enchantments and seven afflictions, plus
  unknown buckets. Damage/block are explicit; other numeric variables use 16
  stable hash buckets. These are original parameters, not final hook-adjusted
  damage predictions.
- Unordered modifier-aware pile signatures (256 hash buckets per pile).
- Thirteen combat-history counts with individual availability flags: plays and
  attack/skill/power counts this turn and combat, draws/discards/exhausts and
  energy spent this turn, and previous-turn play count.
- Four allied creature slots, including Osty: presence, HP/max HP, block, alive
  state, Osty identity and registered power amounts.

The native teacher exporter already saved display counters, card modifiers and
base variables. Older examples can be re-encoded for these features immediately.
They do not contain the newly added history/status/usage fields; those remain
explicitly unavailable. Card-selection snapshots previously omitted some modifier
metadata, so it cannot be reconstructed retroactively.

The native C# exporter now includes history, status/usage and choice-card modifier
metadata. Its updated runtime was built separately in
`sts2-ml/.local/observation-v3-runtime`; new Python simulator processes prefer it.
An already-running collector retains its original imported runtime paths through
its replay check. Its next episode process picks up the exporter update. Raw
snapshots remain the interchange format; generation stays in sts2-ml.

The live bridge serializes the same canonical fields. It compiles but still needs
packaging and in-game validation; full live selection routing is not implemented.

### Limits and verification

Python public-counter parity is currently audited for Pen Nib, Nunchaku and Happy
Flower. Other Python relic counters are marked unavailable rather than guessed;
native/live exporters provide every relic's public display counter. Usage export
is deliberately restricted to IsUsedUp and known public used/triggered properties,
not arbitrary private state. Further per-relic/per-card temporal state remains.
Allied creatures retain the existing combat action mapping; observing an ally does
not add new target-action support. Removed/dead pets may not be present in native
creature lists. Fixed slot limits, unknown registry entries and hash collisions
remain limitations. None of these additions exposes RNG or hidden draw order.

74 targeted tests, a native exporter/importer exercise, 64 PPO rollout steps and
one imitation epoch have passed. Both C# projects compile. The imitation smoke
used 1,139 combat examples, with no mapping rejections and no held-out validation
examples; this is an integration check, not evidence of improved win rate.

The overnight runner now trains v3 into `output/teachers_overnight_v3` so it cannot
accidentally resume an older v2 checkpoint. An already-running Python waiter must
be restarted to load the updated runner code.

## Previous schema: v2


The previous combat-training default was a **7,019-float v2 observation**. The original
217-float encoder remains unchanged for existing checkpoints and the legacy
full-run environment. `STS2CombatEnv(observation_version=2)` and the demonstration
importer use v2; `--observation-version 1` selects legacy training explicitly.
Frozen v2 combat policies also work inside `StrategyEnv`.

## Findings and changes

| Gap | Change |
|---|---|
| Enemy poison and most player scaling powers invisible | Dense amounts for all 271 registered powers, player plus five enemy slots |
| Percentage HP hides actual damage-to-kill | Absolute player HP/max HP and enemy HP/max HP |
| Different monster behaviors indistinguishable | Stable two-part hash of public monster model identity |
| Upgrades, star cost, X cost, retain, exhaust missing | Per-hand-card extra features |
| Piles represented only by counts | Unordered normal/upgraded composition across 583 registered card IDs |
| Discard/retain/removal choices indistinguishable | Choice presence, bounds, purpose, selected indices, ordered option card features |
| Potion actions lack potion identity | Slot identities and belt capacity, preserving empty slots |
| Relics/orbs absent | Relic identity counts; visible orb order, type, capacity, passive/evoke values |
| Live adapter omitted stars and shifted all later features | Correct five-feature player prefix |
| Live adapter discarded secondary intents and heal | Preserve all three existing intent slots |
| Live power suffixes failed matching | Normalize `_POWER` suffixes |
| Live hand payload omitted damage/block | Serialize original base dynamic values |
| Native negative status cost confused with X cost | Explicit reference-derived X-card identities, with native flags taking precedence |

The expanded tail uses the same canonical packet encoder for Python simulation,
native exported demonstrations and live bridge payloads. Its layout fingerprint
includes ordered registry vocabularies. Training checkpoints receive an
`.observation.json` sidecar, and imitation reports record the version, size and
fingerprint. Dimensions must match; an existing 217-input checkpoint cannot be
loaded directly into a v2 environment. No weights were silently padded/migrated.

No RNG, future monster move state or hidden draw order is encoded. Draw-pile
features are multisets: reversing the hidden order leaves the observation equal.
The bridge's pre-existing ordered pile IDs remain available for parity tooling;
they are not read by the policy encoder. Model identity means visible monster
type, not an instance ID or future AI branch.

## Usage

```powershell
.pilot-venv/Scripts/python.exe scripts/train_combat.py --observation-version 2 --total-timesteps 500000
.pilot-venv/Scripts/python.exe scripts/pretrain_demonstrations.py --data ../sts2-ml/runs/fireworks-full-generation-20261101/dataset --role combat --observation-version 2 --output output/combat_teacher_v2
```

The live runner chooses v1/v2 by checkpoint observation dimensions. Rebuild and
package the bridge mod before using v2 live. Its serializer now provides stars,
card flags/base values, all intents, unordered pile cards, relics, potion capacity
and orb values. Only compilation was checked here: a live game and .pck deployment
were not performed. Selection screens still need additional live routing/payload
work to achieve the same coverage as native demonstration selection snapshots.
Older bridge payloads expose missing-pile/resource coverage flags rather than
being equivalent to complete v2 observations.

## Validation and limits

- 58 targeted observation, bridge, environment and interchange tests passed.
- Exact simulator/native/live-packet equality tested on the same public state.
- Behavioral distinction tests cover poison/scaling, absolute enemy HP, potion
  identity, visible orb ordering, choice purpose and selected cards.
- Hidden draw-order invariance, legacy dimensions and v2 frozen-combat strategy
  rollout tested.
- Short PPO and demonstration imitation smoke training run locally. These check
  tensor/mask/training integration, not gameplay strength.
- Updated bridge compilation: zero errors, 116 existing nullable warnings. No
  mod installation or external generator source changes were made.

Remaining gaps include per-instance relic counters, card enchantment/affliction
details, some per-turn/history counters, allies/Osty and full live selection
parity. Unknown registry powers/cards and capacities beyond fixed hand/enemy/menu
limits also need broader coverage work. Monster IDs use hashes rather than a
semantic embedding; rare hash collisions are possible. Retaining the full raw
public snapshots permits future re-encoding without paying to regenerate data.

The larger schema is intentionally dense and is not proof of optimal feature
design or improved win rate. Compare newly trained checkpoints on held-out
encounters and full runs before promoting them.
