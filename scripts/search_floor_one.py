"""Run a seeded Ironclad starter fight with flat Monte Carlo search."""

import argparse
import json
from random import Random

from sts2_env.cards import base as card_ids
from sts2_env.encounters.act1 import WEAK_ENCOUNTERS
from sts2_env.gym_env.action_space import action_to_card_and_target
from sts2_env.gym_env.combat_env import STS2CombatEnv


class ReplayCombat(STS2CombatEnv):
    """Rebuild enemy callbacks along with the rest of the combat state."""

    def reset(self, *, seed, options=None):
        self.seed = seed
        self.history = []
        return super().reset(seed=seed, options=options)

    def step(self, action):
        result = super().step(action)
        self.history.append(action)
        return result

    def fork(self):
        branch = ReplayCombat(encounter_pool=self.encounter_pool,
                              player_hp=self.player_hp, player_max_hp=self.player_max_hp,
                              max_turns=self.max_turns, relics=self.relics,
                              character_id=self.character_id)
        branch.reset(seed=self.seed)
        for action in self.history:
            branch.step(action)
        return branch


def heuristic(env, mask):
    combat = env.combat
    if combat.pending_choice is not None:
        return int(mask.nonzero()[0][0])
    actions = [int(action) for action in mask.nonzero()[0] if action]
    if not actions:
        return 0
    incoming = sum(
        intent.damage * intent.hits
        for enemy in combat.alive_enemies
        for intent in combat.enemy_ais[enemy.combat_id].current_move.intents
        if intent.damage is not None
    )

    def priority(action):
        slot, _ = action_to_card_and_target(action)
        card = combat.hand[slot]
        if incoming > combat.player.block and card.base_block:
            return card.base_block
        return card.base_damage or 0

    return max(actions, key=priority)


def search_scores(env, mask, rng, samples, rollout=heuristic):
    counter = card_ids._next_instance_id
    try:
        return search_branches(env, mask, rng, samples, rollout)
    finally:
        card_ids._next_instance_id = counter


def search_branches(env, mask, rng, samples, rollout):
    actions = [int(action) for action in mask.nonzero()[0]]
    if len(actions) == 1:
        return {actions[0]: 0.0}
    scores = dict.fromkeys(actions, 0.0)
    for _ in range(samples):
        scenario_seed = rng.getrandbits(64)
        for action in actions:
            branch = env.fork()
            Random(scenario_seed).shuffle(branch.combat.draw_pile)
            _, _, ended, timed_out, info = branch.step(action)
            while not (ended or timed_out):
                action_next = rollout(branch, info["action_mask"])
                _, _, ended, timed_out, info = branch.step(action_next)
            if ended and branch.combat.player_won:
                scores[action] += 2 + branch.combat.player.current_hp / 100
            scores[action] -= 0.02 * branch.combat.turn_count
    return {action: score / samples for action, score in scores.items()}


def search(env, mask, rng, samples):
    scores = search_scores(env, mask, rng, samples)
    fallback = heuristic(env, mask)
    return max(scores, key=lambda action: (scores[action], action == fallback))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--heuristic", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    rng = Random(args.seed)
    env = ReplayCombat(encounter_pool=WEAK_ENCOUNTERS, max_turns=30,
                        relics=("BURNING_BLOOD",))
    _, info = env.reset(seed=args.seed)
    ended = timed_out = False
    while not (ended or timed_out):
        mask = info["action_mask"]
        action = heuristic(env, mask) if args.heuristic else search(env, mask, rng, args.samples)
        _, _, ended, timed_out, info = env.step(action)
    print(json.dumps(dict(seed=args.seed, won=env.combat.player_won,
                          hp=env.combat.player.current_hp, timed_out=timed_out)))


if __name__ == "__main__":
    main()
