"""Reproducible no-demonstration pilot; outputs per-seed outcomes and checkpoints."""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.strategy_env import StrategyEnv, SCHEMA_VERSION
from sts2_env.gym_env.strategy_policy import StrategyExtractor


class Deadline(BaseCallback):
    def __init__(self, seconds):
        super().__init__()
        self.until = time.monotonic() + seconds

    def _on_step(self):
        return time.monotonic() < self.until


def evaluate(combat, strategy, seeds, output):
    rows = []
    env = StrategyEnv(combat)
    for seed in seeds:
        try:
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed + 500000)
            for _ in range(10000):
                if strategy is None:
                    action = rng.choice(np.flatnonzero(env.action_masks()))
                else:
                    action, _ = strategy.predict(obs, action_masks=env.action_masks(), deterministic=True)
                obs, _, end, timeout, info = env.step(int(action))
                if end or timeout:
                    rows.append({'seed': seed, **info})
                    break
        except Exception:
            rows.append({'seed': seed, 'error': traceback.format_exc()})
        output.write_text(json.dumps(rows, indent=2))
    env.close()
    valid = [r for r in rows if 'error' not in r]
    wins = sum(r['win'] for r in valid)
    n = len(rows)
    p = wins / n if n else 0
    z = 1.96
    center = (p + z*z/(2*n))/(1+z*z/n) if n else 0
    margin = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n) if n else 0
    return {'attempts': n, 'errors': n-len(valid), 'wins': wins,
            'attempt_win_rate': p, 'wilson95': [max(0, center-margin), min(1, center+margin)],
            'timeouts': sum(r['timeout'] for r in valid),
            'mean_floor': float(np.mean([r['floor'] for r in valid])) if valid else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/strategy_pilot')
    parser.add_argument('--combat-steps', type=int, default=200000)
    parser.add_argument('--strategy-steps', type=int, default=20000)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--hours', type=float, default=2)
    parser.add_argument('--combat-path', default=None, help='Reuse an already trained combat checkpoint')
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {'config': vars(args), 'schema_version': SCHEMA_VERSION,
               'character': 'Silent', 'ascension': 0, 'acts': 3,
               'training_seed': 42, 'eval_seeds': [10000, 10000+args.episodes-1],
               'combat_training': 'Silent starter deck, 70 HP, Act 1 encounters; limited deck coverage',
               'torch': torch.__version__, 'cuda': torch.cuda.is_available()}
    seeds = range(10000, 10000 + args.episodes)
    summary['random_both'] = evaluate(None, None, seeds, out/'random_both.json')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    if args.combat_path:
        combat = MaskablePPO.load(args.combat_path, device='cpu')
    else:
        combat = MaskablePPO('MlpPolicy', STS2CombatEnv(character_id='Silent', player_hp=70, player_max_hp=70), seed=42, device='cpu',
                             n_steps=512, batch_size=128, verbose=0)
        combat.learn(args.combat_steps, callback=Deadline(args.hours*3600*0.3))
    combat.save(out/'combat')
    summary['combat_timesteps'] = combat.num_timesteps
    summary['random_strategy_trained_combat'] = evaluate(combat, None, seeds, out/'random_strategy.json')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    strategy = MaskablePPO('MultiInputPolicy', StrategyEnv(combat), seed=42, device='cpu',
                           n_steps=256, batch_size=64, gamma=0.99, verbose=1,
                           policy_kwargs={'features_extractor_class': StrategyExtractor,
                                          'net_arch': {'pi': [128], 'vf': [128]}})
    strategy.learn(args.strategy_steps, callback=Deadline(max(1, args.hours*3600-(time.monotonic()-started))))
    strategy.save(out/'strategy')
    summary['strategy_timesteps'] = strategy.num_timesteps
    summary['trained_strategy_trained_combat'] = evaluate(combat, strategy, seeds, out/'trained_strategy.json')
    summary['elapsed_seconds'] = time.monotonic()-started
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
