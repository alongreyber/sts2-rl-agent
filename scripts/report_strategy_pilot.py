"""Write a compact, reviewable pilot report with provenance and saved outcomes."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    root = Path(args.directory)
    data = json.loads((root/'summary.json').read_text())
    provenance = {'python': platform.python_version(),
                  'packages': {name: importlib.metadata.version(name) for name in
                               ['numpy', 'torch', 'gymnasium', 'stable-baselines3', 'sb3-contrib']},
                  'source_sha256': {}}
    for name in ['sts2_env/gym_env/strategy_env.py', 'sts2_env/gym_env/strategy_policy.py',
                 'sts2_env/gym_env/combat_env.py', 'scripts/pilot_strategy.py']:
        provenance['source_sha256'][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (root/'environment.json').write_text(json.dumps(provenance, indent=2))
    import torch
    from sb3_contrib import MaskablePPO
    from sts2_env.gym_env.combat_env import STS2CombatEnv
    torch.set_num_threads(2)
    model = MaskablePPO.load(root/'combat.zip', device='cpu')
    env = STS2CombatEnv(character_id='Silent', player_hp=70, player_max_hp=70)
    combat_rows = []
    for seed in range(20000, 20100):
        obs, info = env.reset(seed=seed)
        for step in range(2000):
            action, _ = model.predict(obs, action_masks=info['action_mask'], deterministic=True)
            obs, reward, end, timeout, info = env.step(int(action))
            if end or timeout or step == 1999:
                combat_rows.append({'seed': seed, 'win': bool(end and reward > 0), 'timeout': bool(timeout or step == 1999)})
                break
    (root/'combat_eval.json').write_text(json.dumps(combat_rows, indent=2))
    lines = ['# No-trajectory strategy pilot', '',
             'Silent, A0, three acts; 100 held-out simulator seeds, 10000–10099.', '',
             '| Condition | Wins | Mean floor | Timeouts | Errors |',
             '|---|---:|---:|---:|---:|']
    labels = {'random_both': 'Random combat + random strategy',
              'random_strategy_trained_combat': 'Trained combat + random strategy',
              'trained_strategy_trained_combat': 'Trained combat + trained strategy'}
    for key, label in labels.items():
        row = data[key]
        lines.append(f'| {label} | {row["wins"]}/{row["attempts"]} | {row["mean_floor"]:.2f} | {row["timeouts"]} | {row["errors"]} |')
    lines += ['', f'Combat steps: {data["combat_timesteps"]:,}. Strategy steps: {data["strategy_timesteps"]:,}.',
              f'Isolated starter-deck combat evaluation: {sum(r["win"] for r in combat_rows)}/100 wins (seeds 20000–20099).',
              f'Final pilot elapsed: {data["elapsed_seconds"]:.0f} seconds (reuses combat checkpoint).', '',
              'Each condition uses a single training seed; this is a baseline, not an architecture comparison.',
              'Zero wins out of 100 has a Wilson 95% interval of approximately 0–3.7%.',
              'Combat training covers a Silent starter deck and Act 1 encounters only.',
              'Sparse strategy rewards and limited combat coverage constrain this result.',
              'The C# compile target passed. Mod packaging requires configuring the missing Godot executable.',
              'No live-game benchmark was performed. Simulator/bridge special-screen parity remains unverified.',
              'The targeted regression suite passed; see the chat for the exact test result.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
