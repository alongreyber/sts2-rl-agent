import numpy as np
import pytest

from scripts.expert_iteration import observation
from scripts.search_floor_one import ReplayCombat
from sts2_env.characters.all import get_character
from sts2_env.encounters.act1 import WEAK_ENCOUNTERS
from sts2_env.gym_env.combat_observation import OBS_SIZE, extras


@pytest.mark.parametrize("character", ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"])
def test_observation_shape_and_hidden_order(character):
    config = get_character(character)
    env = ReplayCombat(character_id=character, encounter_pool=WEAK_ENCOUNTERS,
                       relics=(config.starting_relic.name,))
    env.reset(seed=0)
    before = observation(env.combat)
    assert before.shape == (OBS_SIZE,)
    assert np.isfinite(before).all()
    env.combat.draw_pile.reverse()
    np.testing.assert_array_equal(observation(env.combat), before)
    env.combat.hand[0].enchantments["Sharp"] = 3
    assert not np.array_equal(observation(env.combat), before)


def test_ally_and_history_are_visible():
    packet = {"allies": [{"id": "OSTY", "hp": 20}]}
    before = extras(packet)
    packet["allies"][0]["hp"] = 5
    assert not np.array_equal(extras(packet), before)
    assert not np.array_equal(extras({}), extras({"history": {"plays_turn": 0}}))
