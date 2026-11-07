"""Reward calculation."""

from __future__ import annotations

from sts2_env.core.combat import CombatState


def compute_reward(combat: CombatState, prev_hp: int, current_hp: int, prev_creatures: List, creatures: List) -> float:
    """Compute step reward.

    Sparse reward: +1 for win, -1 for loss, 0 otherwise.
    """
    total_reward = 0
    for i in range(0, len(prev_creatures)):
        diff = prev_creatures[i].current_hp - creatures[i].current_hp
        total_reward += diff

    total_reward += current_hp - prev_hp
     
    if combat.is_over:
        if combat.player_won:
            total_reward += 100
        else:
            total_reward -= 100
        return total_reward
    return 0.0
