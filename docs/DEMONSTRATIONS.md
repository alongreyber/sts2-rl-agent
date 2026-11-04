# Demonstration interchange and imitation warm start

Generation stays in `sts2-ml/scripts/engine/generate_fireworks.py`. The training
repository consumes `sts2-demonstrations/v1`: `manifest.json` plus checksummed
gzip JSONL episode shards. Each row retains the native public observation,
ordered legal candidates, chosen index, combat/run role, episode/source-group
identity, actor attribution, game fingerprint, status/outcome, and execution
verification. Raw prompts and provider responses remain in generator run folders.
The producer's existing internal format 4 is not the portable schema version.

`sts2_env/demonstrations.py` validates the schema/checksums and owns native-state
encoding. Combat defaults to the expanded observation v3 (see the audit for its size) and the
existing legal action mapping. Use `--observation-version 1` for legacy
217-feature checkpoints, or version 2 for 7,019-feature checkpoints; dimensions must match. See
[the combat observation audit](COMBAT_OBSERVATION_AUDIT.md) for fields and limits.
Strategy translates native candidates to the existing candidate
encoder. Unknown/unrepresentable mappings are reported as rejected examples;
they are not replaced by random actions. Candidate ordering defines legality;
the consumer constructs the current network's mask. Game metadata is provenance,
not an input feature; seed/RNG/future monster moves are excluded by the producer.

Automatic/unattributed actions are excluded by default. `llm` and `plan` execution
is eligible, including planned action sequences; a recorded primitive state is
not necessarily a separate LLM judgment. Skipping non-card reward collection is
not used as a strategy label. The producer records decisions after execution;
replay verification certifies execution, not optimality. Interrupted prefixes are
usable for imitation but must not supply fabricated defeat/value targets.

```powershell
.pilot-venv/Scripts/python.exe scripts/pretrain_demonstrations.py --data ../sts2-ml/runs/my-campaign/dataset --role combat --output output/imitation/combat --validate-only
.pilot-venv/Scripts/python.exe scripts/pretrain_demonstrations.py --data ../sts2-ml/runs/my-campaign/dataset --role combat --output output/imitation/combat --epochs 5
.pilot-venv/Scripts/python.exe scripts/pretrain_demonstrations.py --data ../sts2-ml/runs/my-campaign/dataset --role strategy --output output/imitation/strategy --epochs 5
```

`--initial` accepts an existing PPO checkpoint. The masked negative log-likelihood
loss trains actor behavior; it supplies no teacher value targets. Outputs are
ordinary MaskablePPO checkpoints usable for subsequent RL. The model report
records eligible train/validation counts, mapping rejections, and held-out teacher
agreement. Agreement measures imitation, not strength. Splits are deterministic
by source seed/configuration group, keeping neighboring decisions together.
Tiny pilots may contain no validation episodes; reports state this explicitly.

This importer is a first native-simulator adapter, not proof of parity between
both game engines. Expanded combat observations still have documented coverage gaps.
Strategy candidate semantics and route features should receive broader live
parity validation before interpreting a warm start as a demonstrated win-rate gain.
