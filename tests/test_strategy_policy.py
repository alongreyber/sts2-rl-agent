import numpy as np
import pytest

from sts2_env.gym_env.strategy_env import StrategyEnv, encode_strategy, MAX_CHOICES
from sts2_env.bridge.strategy_adapter import strategy_observation, send_strategy_choice


def test_inventory_and_offer_changes_are_visible():
    first = encode_strategy({'deck': [{'id': 'STRIKE', 'upgraded': False}]}, [{'action': 'pick_card', 'card_id': 'DEFEND'}])
    second = encode_strategy({'deck': [{'id': 'STRIKE', 'upgraded': True}]}, [{'action': 'pick_card', 'card_id': 'POISONED_STAB'}])
    assert not np.array_equal(first['context'], second['context'])
    assert not np.array_equal(first['candidates'], second['candidates'])
    assert first['valid'].sum() == 1
    with pytest.raises(ValueError):
        encode_strategy({}, [{}] * (MAX_CHOICES + 1))


def test_bridge_packet_matches_simulator_encoder_and_retains_wire_indexes():
    state = {'strategy_context': {'gold': 99, 'deck': []}, 'type': 'card_reward',
             'cards': [{'id': 'DEFEND', 'index': 7}], 'can_skip': True}
    obs = strategy_observation(state)
    expected = encode_strategy(state['strategy']['context'], state['strategy']['choices'])
    for key in obs:
        np.testing.assert_array_equal(obs[key], expected[key])
    class Client:
        def choose(self, index):
            self.index = index
    client = Client()
    send_strategy_choice(client, state, 1)
    assert client.index == 7
    with pytest.raises(ValueError):
        strategy_observation({'type': 'card_reward'})


def test_native_offers_match_simulator_semantics():
    context = {'deck': [{'id': 'POISONED_STAB', 'upgraded': False}], 'gold': 100}
    sim = encode_strategy({**context, 'phase': 'SHOP'},
                          [{'action': 'buy_card', 'card_id': 'POISONED_STAB',
                            'rarity': 'COMMON', 'price': 50}])
    native = strategy_observation({'type': 'shop', 'strategy_context': context,
                                   'options': [{'index': 9, 'action': 'buy_card',
                                                'id': 'buy_card', 'label': 'POISONED_STAB',
                                                'price': 50, 'description': 'Cost: 50'}]})
    for key in sim:
        np.testing.assert_array_equal(sim[key], native[key])


def test_native_map_matches_simulator_semantics():
    context = {'floor': 1, 'deck': []}
    sim = encode_strategy({**context, 'phase': 'MAP_CHOICE'},
                          [{'action': 'move', 'point_type': 'MONSTER', 'coord': (2, 3),
                            'routes': ['MONSTER>REST_SITE']}])
    native = strategy_observation({'type': 'map_select', 'strategy_context': context,
                                   'nodes': [{'type': 'MONSTER', 'row': 3, 'col': 2,
                                              'routes': ['MONSTER>REST_SITE'], 'index': 17}]})
    for key in sim:
        np.testing.assert_array_equal(sim[key], native[key])


def test_strategy_advances_to_next_decision_and_rejects_masked_actions():
    env = StrategyEnv(max_steps=100)
    obs, _ = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    with pytest.raises(ValueError):
        env.step(MAX_CHOICES - 1)
    obs, _, terminated, truncated, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert env.observation_space.contains(obs)
    assert info['primitive_steps'] >= 1
    if not terminated and not truncated:
        assert env.run._mgr.phase != 'COMBAT'


def test_attention_policy_masks_and_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    from sb3_contrib import MaskablePPO
    from sts2_env.gym_env.strategy_policy import StrategyExtractor
    torch.set_num_threads(1)
    env = StrategyEnv(max_steps=100)
    model = MaskablePPO('MultiInputPolicy', env, n_steps=8, batch_size=8,
                       policy_kwargs={'features_extractor_class': StrategyExtractor})
    obs, _ = env.reset(seed=123)
    action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
    assert env.action_masks()[int(action)]
    model.save(tmp_path/'policy')
    loaded = MaskablePPO.load(tmp_path/'policy')
    again, _ = loaded.predict(obs, action_masks=env.action_masks(), deterministic=True)
    assert int(again) == int(action)


def test_multi_card_bridge_picks_distinct_required_cards():
    from sts2_env.bridge.strategy_adapter import act_strategy
    class Model:
        def predict(self, obs, action_masks, deterministic):
            return np.flatnonzero(action_masks)[0], None
    class Client:
        def choose_many(self, indexes):
            self.indexes = indexes
    state = {'type': 'card_select', 'strategy_context': {},
             'cards': [{'id': 'STRIKE', 'index': 4}, {'id': 'DEFEND', 'index': 9}],
             'min_select': 2, 'max_select': 2}
    state['strategy_context'] = {'deck': []}
    client = Client()
    act_strategy(Model(), client, state)
    assert client.indexes == [4, 9]
