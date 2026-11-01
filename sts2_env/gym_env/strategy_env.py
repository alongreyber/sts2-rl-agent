"""Strategy-only environment. Combat advances under a frozen masked policy.

Versioned observations contain named state features and candidate features;
actions are indexes into the current candidate list, never legacy action slices.
"""
from __future__ import annotations

import hashlib
import re
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.run.run_manager import RunManager

MAX_CHOICES = 128
FEATURES = 128
SCHEMA_VERSION = 1


def features(value):
    """Stable signed feature hashing, including numeric values and field names."""
    out = np.zeros(FEATURES, np.float32)
    def add(key, amount=1.0):
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        out[int.from_bytes(digest[:4], 'little') % FEATURES] += amount * (1 if digest[4] & 1 else -1)
    def walk(obj, path=''):
        if isinstance(obj, dict):
            for key, val in sorted(obj.items()):
                if key not in {'index', 'action_index', 'instance_id', 'session_id', 'seed', 'last_description'}:
                    walk(val, path + '/' + key)
        elif isinstance(obj, (list, tuple)):
            for val in obj:
                walk(val, path)
        elif isinstance(obj, (int, float)):
            number = float(obj)
            add(path, np.sign(number) * np.log1p(abs(number)))
        else:
            text = str(obj.name if isinstance(obj, Enum) else obj)
            text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', text)
            for token in re.findall(r'[a-z]+|\d+', text.lower()):
                add(path + ':' + token)
    walk(value)
    return np.clip(out, -20, 20)


def encode_strategy(context, choices):
    if len(choices) > MAX_CHOICES:
        raise ValueError(f'{len(choices)} choices exceeds schema capacity {MAX_CHOICES}')
    candidates = np.zeros((MAX_CHOICES, FEATURES), np.float32)
    valid = np.zeros(MAX_CHOICES, np.float32)
    for i, choice in enumerate(choices):
        candidates[i] = features(canonical_choice(choice))
        valid[i] = 1
    return {'context': features(context), 'candidates': candidates, 'valid': valid}


def canonical_choice(choice):
    """Only expose semantic fields shared with the game bridge, not wire indexes."""
    kind = choice.get('action', '')
    keys = {
        'move': ('point_type', 'coord', 'routes'),
        'pick_card': ('card_id', 'upgraded'),
        'choose': ('card_id',),
        'pick_relic': ('relic_id',),
        'buy_card': ('card_id', 'price'),
        'buy_relic': ('relic_id', 'price'),
        'buy_potion': ('potion_id', 'price'),
        'remove_card': ('price',),
        'rest_option': ('option_id',),
        'event_choice': ('label', 'description'),
    }.get(kind, ('card_ids', 'potion_id', 'relic_id', 'label', 'description'))
    result = {'action': kind, **{key: choice[key] for key in keys if key in choice}}
    if kind == 'pick_card':
        result.setdefault('upgraded', False)
    return result


def simulator_snapshot(mgr):
    actions = mgr.get_available_actions()
    if mgr.phase == RunManager.PHASE_MAP_CHOICE:
        from sts2_env.map.map_point import MapCoord
        def paths(point, depth=3):
            if depth == 0 or not point.children:
                return [point.point_type.name]
            return [point.point_type.name + '>' + tail
                    for child in point.children for tail in paths(child, depth-1)]
        actions = [dict(a) for a in actions]
        for action in actions:
            point = mgr.run_state.map.get_point(MapCoord(*action['coord']))
            action['routes'] = paths(point) if point else []
    player = mgr.run_state.player
    context = {'phase': mgr.phase, 'act': mgr.run_state.current_act_index + 1,
               'floor': mgr.run_state.total_floor, 'hp': player.current_hp,
               'max_hp': player.max_hp, 'gold': player.gold,
               'deck': [{'id': c.card_id.name, 'upgraded': c.upgraded} for c in player.deck],
               'relics': list(player.relics),
               'potions': [p.potion_id for p in player.potions if p is not None]}
    # Keep raw candidate attributes: IDs, upgrades, prices, routes, and descriptions.
    return context, actions, actions


class StrategyEnv(gym.Env):
    def __init__(self, combat_model=None, character='Silent', max_steps=10000):
        self.run = STS2RunEnv(character_id=character, max_steps=max_steps)
        self.combat_model = combat_model
        self.action_space = spaces.Discrete(MAX_CHOICES)
        self.observation_space = spaces.Dict({
            'context': spaces.Box(-20, 20, (FEATURES,), np.float32),
            'candidates': spaces.Box(-20, 20, (MAX_CHOICES, FEATURES), np.float32),
            'valid': spaces.Box(0, 1, (MAX_CHOICES,), np.float32),
        })
        self.steps = 0
        self.ended = False

    def _obs(self):
        context, choices, self.choices = simulator_snapshot(self.run._mgr)
        obs = encode_strategy(context, choices)
        if not self.choices:
            obs['valid'][0] = 1
        self.obs = obs
        return obs

    def action_masks(self):
        return self.obs['valid'].astype(bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.ended = False
        self.run.reset(seed=seed)
        return self._obs(), {}

    def step(self, action):
        if self.ended:
            raise gym.error.ClosedEnvironmentError('Call reset after a terminal strategy step')
        if not self.action_masks()[int(action)]:
            raise ValueError('Illegal strategy choice')
        mgr = self.run._mgr
        mgr.take_action(self.choices[int(action)])  # Exceptions propagate: never score bugs as defeats.
        self.steps += 1
        truncated = False
        while not mgr.is_over and mgr.phase == RunManager.PHASE_COMBAT:
            if self.steps >= self.run.max_steps:
                truncated = True
                break
            obs = self.run._encode_obs()[:self.run.observation_space.shape[0] - 20]
            mask = self.run.action_masks()
            from sts2_env.core.constants import ACTION_SPACE_SIZE
            mask = mask[:ACTION_SPACE_SIZE]
            if self.combat_model is None:
                chosen = int(self.np_random.choice(np.flatnonzero(mask)))
            else:
                chosen, _ = self.combat_model.predict(obs, action_masks=mask, deterministic=True)
            # Direct dispatch avoids run_env's exception-to-loss handler.
            self.run._step_combat(int(chosen))
            self.steps += 1
        truncated = truncated or (not mgr.is_over and self.steps >= self.run.max_steps)
        terminated = mgr.is_over
        self.ended = terminated or truncated
        info = {'win': bool(mgr.player_won), 'floor': mgr.run_state.total_floor,
                'act': mgr.run_state.current_act_index, 'timeout': truncated, 'primitive_steps': self.steps}
        return self._obs(), (1.0 if mgr.player_won else -1.0) if terminated or truncated else 0.0, terminated, truncated, info

    def close(self):
        self.run.close()
