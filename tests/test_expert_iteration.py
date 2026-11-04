from random import Random

import numpy as np

from scripts.expert_iteration import observation
from scripts.search_floor_one import ReplayCombat, search_scores
from sts2_env.encounters.act1 import setup_fuzzy_wurm_crawler_weak


def test_rollouts_leave_live_enemy_and_draw_order_unchanged():
    env = ReplayCombat(encounter_pool=[setup_fuzzy_wurm_crawler_weak])
    _, info = env.reset(seed=7)
    before = observation(env.combat)
    order = [card.instance_id for card in env.combat.draw_pile]

    scores = search_scores(env, info["action_mask"], Random(7), samples=2)

    assert set(scores) == set(info["action_mask"].nonzero()[0])
    np.testing.assert_array_equal(observation(env.combat), before)
    assert [card.instance_id for card in env.combat.draw_pile] == order
    assert env.history == []


def test_replay_fork_recreates_the_same_transition():
    env = ReplayCombat(encounter_pool=[setup_fuzzy_wurm_crawler_weak])
    env.reset(seed=7)
    env.step(0)
    branch = env.fork()
    assert branch.combat.enemies[0] is not env.combat.enemies[0]
    np.testing.assert_array_equal(observation(branch.combat), observation(env.combat))
    branch_obs, _, _, _, branch_info = branch.step(0)
    live_obs, _, _, _, live_info = env.step(0)
    np.testing.assert_array_equal(branch_obs, live_obs)
    np.testing.assert_array_equal(branch_info["action_mask"], live_info["action_mask"])


def test_apprentice_observation_hides_draw_order():
    env = ReplayCombat(encounter_pool=[setup_fuzzy_wurm_crawler_weak])
    env.reset(seed=7)
    before = observation(env.combat)
    env.combat.draw_pile.reverse()
    np.testing.assert_array_equal(observation(env.combat), before)
