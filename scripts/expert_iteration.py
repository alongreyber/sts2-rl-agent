"""Train and evaluate a masked apprentice on opening-act weak encounters."""

import argparse
import json
from pathlib import Path
from random import Random

import numpy as np

from scripts.search_floor_one import ReplayCombat, heuristic, search_scores
from sts2_env.characters.all import get_character
from sts2_env.encounters.act1 import WEAK_ENCOUNTERS
from sts2_env.encounters.act4 import WEAK_ENCOUNTERS as UNDERDOCKS_WEAK
from sts2_env.gym_env.combat_observation import encode, SCHEMA_SHA256


def observation(combat):
    return encode(combat)


def probabilities(logits, mask):
    masked = np.where(mask, logits, -np.inf)
    weights = np.exp(masked - masked.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


class Apprentice:
    def __init__(self, width, actions, rng):
        self.hidden = rng.normal(0, 0.08, (width + 1, 64))
        self.output = rng.normal(0, 0.08, (65, actions))

    def __call__(self, env, mask):
        hidden = np.tanh(np.append(observation(env.combat), 1) @ self.hidden)
        logits = np.append(hidden, 1) @ self.output
        return int(np.argmax(np.where(mask, logits, -np.inf)))

    def fit(self, rows, rng):
        features, masks, targets = (np.stack(column) for column in zip(*rows))
        features = np.column_stack((features, np.ones(len(rows))))
        for _ in range(24):
            for batch in np.array_split(rng.permutation(len(rows)), max(1, len(rows) // 64)):
                x = features[batch]
                hidden = np.tanh(x @ self.hidden)
                h = np.column_stack((hidden, np.ones(len(batch))))
                error = (probabilities(h @ self.output, masks[batch]) - targets[batch]) / len(batch)
                gradient = (error @ self.output[:-1].T) * (1 - hidden**2)
                self.output -= 0.15 * (h.T @ error)
                self.hidden -= 0.15 * (x.T @ gradient)


def fight(encounter, seed, policy, samples=0, rows=None, *, character="Ironclad"):
    config = get_character(character)
    env = ReplayCombat(encounter_pool=[encounter], max_turns=30,
                       relics=(config.starting_relic.name,), character_id=character,
                       player_hp=config.starting_hp, player_max_hp=config.starting_hp)
    _, info = env.reset(seed=seed)
    rng = Random(seed)
    while True:
        mask = info["action_mask"]
        if samples:
            scores = search_scores(env, mask, rng, samples, rollout=policy)
            prior = policy(env, mask)
            action = max(scores, key=lambda candidate: (scores[candidate], candidate == prior))
            if rows is not None and len(scores) > 1:
                logits = np.zeros(len(mask))
                for candidate, score in scores.items():
                    logits[candidate] = 8 * score + 0.01 * (candidate == prior)
                rows.append((observation(env.combat), mask, probabilities(logits, mask)))
        else:
            action = policy(env, mask)
        _, _, done, timeout, info = env.step(action)
        if done or timeout:
            return bool(done and env.combat.player_won), env.combat.player.current_hp, timeout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--train-seeds", type=int, default=8)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--eval-start", type=int, help="First held-out seed.")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--character", default="Ironclad",
                        choices=("Ironclad", "Silent", "Defect", "Regent", "Necrobinder"))
    parser.add_argument("--pool", choices=("overgrowth", "underdocks", "both"), default="both")
    parser.add_argument("--output", type=Path, help="Save trained weights and search examples as NPZ.")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate saved weights without training.")
    args = parser.parse_args()
    if min(args.rounds, args.train_seeds, args.eval_seeds, args.samples) < 1:
        parser.error("all counts must be positive")
    training_end = args.rounds * args.train_seeds
    eval_start = training_end if args.eval_start is None else args.eval_start
    if not args.checkpoint and eval_start < training_end:
        parser.error("--eval-start must follow the training seeds")
    rng = np.random.default_rng(0)
    rows = []
    apprentice = None
    encounters = {"overgrowth": WEAK_ENCOUNTERS, "underdocks": UNDERDOCKS_WEAK,
                  "both": WEAK_ENCOUNTERS + UNDERDOCKS_WEAK}[args.pool]
    if args.checkpoint:
        if args.eval_start is None:
            parser.error("--checkpoint requires an explicit held-out --eval-start")
        if args.output:
            parser.error("--output is for training runs")
        with np.load(args.checkpoint) as checkpoint:
            if checkpoint["schema_version"] != 2 or checkpoint["character"] != args.character:
                parser.error("checkpoint schema or character does not match")
            if eval_start < checkpoint["training_end"]:
                parser.error("--eval-start must follow the checkpoint's training seeds")
            if str(checkpoint["schema_fingerprint"]) != SCHEMA_SHA256:
                parser.error("checkpoint observation layout does not match")
            hidden, output = checkpoint["hidden"], checkpoint["output"]
            apprentice = Apprentice(hidden.shape[0] - 1, output.shape[1], rng)
            apprentice.hidden, apprentice.output = hidden, output

    for iteration in range(0 if args.checkpoint else args.rounds):
        rollout = heuristic if apprentice is None else apprentice
        for encounter in encounters:
            for seed in range(iteration * args.train_seeds, (iteration + 1) * args.train_seeds):
                fight(encounter, seed, rollout, args.samples, rows, character=args.character)
        if apprentice is None:
            features, mask, _ = rows[0]
            apprentice = Apprentice(len(features), len(mask), rng)
        apprentice.fit(rows, rng)
        print(json.dumps({"iteration": iteration + 1, "examples": len(rows)}), flush=True)

    if args.output:
        features, masks, targets = (np.stack(column) for column in zip(*rows))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, hidden=apprentice.hidden, output=apprentice.output,
                            features=features, masks=masks, targets=targets,
                            character=args.character, schema_version=2, schema_fingerprint=SCHEMA_SHA256, training_end=training_end)

    for encounter in encounters:
        seeds = range(eval_start, eval_start + args.eval_seeds)
        results = {}
        for name, policy, samples in (
            ("heuristic", heuristic, 0), ("initial_search", heuristic, args.samples),
            ("apprentice", apprentice, 0), ("guided_search", apprentice, args.samples),
        ):
            fights = [fight(encounter, seed, policy, samples, character=args.character) for seed in seeds]
            winning_hp = [hp for won, hp, _ in fights if won]
            results[name] = {
                "wins": len(winning_hp),
                "mean_hp_on_win": round(sum(winning_hp) / len(winning_hp), 2) if winning_hp else None,
                "timeouts": sum(timeout for _, _, timeout in fights),
            }
        print(json.dumps(dict(encounter=encounter.__name__, seeds=len(seeds), results=results)), flush=True)


if __name__ == "__main__":
    main()
