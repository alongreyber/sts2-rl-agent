import copy
import numpy as np
from sts2_env.gym_env import combat_observation_v2 as v2, combat_observation_v3 as v3
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.observation_versions import version_for_shape
from sts2_env.relics.base import RelicId
from sts2_env.relics.uncommon import PenNib


def fixture():
    env = STS2CombatEnv(observation_version=3)
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    return env.combat


def test_relic_counter_distinguishes_previously_identical_states():
    c = fixture(); r = PenNib(RelicId.PEN_NIB); c.relics.append(r)
    before = v3.encode(c); legacy = v2.encode(c)
    r._attacks_played = 9
    np.testing.assert_array_equal(v2.encode(c), legacy)
    assert not np.array_equal(v3.encode(c), before)


def test_card_modifiers_in_hand_and_hidden_piles_are_observed_without_order():
    c = fixture(); before = v3.encode(c)
    c.hand[0].enchantments['Sharp'] = 3
    assert not np.array_equal(v3.encode(c), before)
    before = v3.encode(c)
    c.draw_pile[0].afflict('Bound')
    assert not np.array_equal(v3.encode(c), before)
    before = v3.encode(c); c.draw_pile.reverse()
    np.testing.assert_array_equal(before, v3.encode(c))


def test_history_counts_and_missing_data_are_distinct():
    c = fixture(); before = v3.encode(c)
    card = c.hand[0]; card.owner = c.player
    c._played_cards_this_turn.append(card)
    assert not np.array_equal(before, v3.encode(c))
    empty = {'player': {}, 'hand': []}
    assert not np.array_equal(v3.extras(empty), v3.extras(dict(empty, history={'plays_turn':0})))


def test_native_existing_snapshot_fields_are_retained():
    raw = {'Id':1, 'Model':'STRIKE_SILENT', 'Cost':1, 'Upgrade':0,
           'Enchantment':'SHARP:3','Affliction':'BOUND:1',
           'Details':{'Variables':{'Damage':9}}, 'Keywords':[], 'ExhaustOnNextPlay':True}
    creature = {'Id':0,'Model':'SILENT','Hp':70,'MaxHp':70,'Block':0,'Powers':[]}
    state = {'Decision':'Monster','Options':[], 'RelicStates':[{'Model':'PEN_NIB','Counter':9,'Stack':1,'IsMelted':False}],
             'Combat':{'Round':1,'Cards':[raw], 'Creatures':[creature], 'Enemies':[], 'Piles':[{'Type':'Hand','CardIds':[1]}]}}
    packet = v3.native_packet(state)
    assert packet['relic_states'][0]['counter'] == 9
    assert packet['hand'][0]['modifiers']['enchantments']=={'SHARP':3}
    assert packet['hand'][0]['exhaust_next']
    assert packet['hand'][0]['variables']['Damage']==9
    before = v3.extras(packet); state['RelicStates'][0]['Counter']=0
    assert not np.array_equal(before,v3.extras(v3.native_packet(state)))


def test_ally_state_and_powers_are_visible():
    packet = {'allies':[{'id':'OSTY','hp':20,'max_hp':30,'block':0,'alive':True,'powers':[]}]}
    before=v3.extras(packet); packet['allies'][0]['hp']=5
    assert not np.array_equal(before,v3.extras(packet))
    before=v3.extras(packet); packet['allies'][0]['powers']=[{'id':'STRENGTH','amount':4}]
    assert not np.array_equal(before,v3.extras(packet))


def test_schema_shapes_and_legacy_prefix():
    c=fixture(); obs=v3.encode(c)
    assert obs.shape==(v3.OBS_SIZE,) and np.isfinite(obs).all()
    np.testing.assert_array_equal(obs[:v2.OBS_SIZE],v2.encode(c))
    assert [version_for_shape(s) for s in [(217,),(v2.OBS_SIZE,),(v3.OBS_SIZE,)]]==[1,2,3]


def test_usage_flags_and_dynamic_variables_are_visible():
    packet={'relic_states':[{'id':'CENTENNIAL_PUZZLE','usage':{'used_this_combat':False}}]}
    before=v3.extras(packet); packet['relic_states'][0]['usage']['used_this_combat']=True
    assert not np.array_equal(before,v3.extras(packet))
    packet={'hand':[{'variables':{'Damage':10,'Forge':3}}]}
    before=v3.extras(packet); packet['hand'][0]['variables']['Forge']=8
    assert not np.array_equal(before,v3.extras(packet))


def test_bridge_preserves_the_v3_canonical_tail():
    from sts2_env.bridge.state_adapter import StateAdapter
    packet=v3.simulator_packet(fixture())
    encoded=StateAdapter(3).encode_observation(packet)
    np.testing.assert_array_equal(encoded[v2.OBS_SIZE:],v3.extras(packet))


def test_variable_names_normalize_without_double_counting():
    np.testing.assert_array_equal(v3.card_features({'variables':{'damage':3,'Damage':9}}),
                                  v3.card_features({'variables':{'Damage':9}}))


def test_strategy_can_use_v3_frozen_combat_policy():
    from types import SimpleNamespace
    from sts2_env.gym_env.strategy_env import StrategyEnv
    class Model:
        observation_space=SimpleNamespace(shape=(v3.OBS_SIZE,))
        calls=0
        def predict(self,obs,**kwargs):
            assert obs.shape==(v3.OBS_SIZE,)
            self.calls+=1
            return 0,None
    model=Model(); env=StrategyEnv(combat_model=model,max_steps=100)
    env.reset(seed=42); env.step(0)
    assert model.calls>0
    env.close()
