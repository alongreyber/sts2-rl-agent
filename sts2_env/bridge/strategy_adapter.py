"""Versioned strategy bridge; fails closed when inventory is missing."""
from sts2_env.gym_env.strategy_env import encode_strategy, SCHEMA_VERSION


def native_packet(state):
    """Translate native indexed choices while retaining their wire commands."""
    context = state.get('strategy_context')
    if not context:
        raise ValueError('Missing strategy_context; rebuild bridge mod before strategy inference')
    phase = {'map_select': 'MAP_CHOICE', 'card_reward': 'CARD_REWARD',
             'card_select': 'CARD_REWARD', 'rest_site': 'REST_SITE',
             'shop': 'SHOP', 'event': 'EVENT', 'treasure': 'TREASURE',
             'boss_relic': 'BOSS_RELIC', 'reward_screen': 'CARD_REWARD',
             'card_bundle': 'CARD_REWARD', 'crystal_sphere': 'EVENT'}.get(state['type'])
    context = {**context, 'phase': phase}
    choices, commands = [], []
    def append(choice, command):
        choices.append(choice)
        commands.append(command)
    if state['type'] == 'map_select':
        for i, node in enumerate(state.get('nodes', [])):
            append({'action': 'move', 'point_type': node['type'],
                    'coord': (node['col'], node['row']), 'routes': node.get('routes', [])}, {'index': node.get('index', i)})
    elif state['type'] in {'card_reward', 'card_select'}:
        if state.get('can_skip'):
            append({'action': 'skip'}, {'skip': True})
        for i, card in enumerate(state.get('cards', [])):
            append({'action': 'pick_card' if state['type'] == 'card_reward' else 'choose',
                    'card_id': card.get('id'), 'upgraded': card.get('upgraded', False)},
                   {'indexes': [card.get('index', i)]} if state['type'] == 'card_select' else {'index': card.get('index', i)})
    else:
        for i, option in enumerate(state.get('options', state.get('bundles', []))):
            if option.get('enabled', True):
                choice = {k: v for k, v in option.items() if k != 'index'}
                if state['type'] == 'rest_site':
                    choice.update(action='rest_option', option_id=option.get('id'))
                if choice.get('action') in {'buy_card', 'buy_relic', 'buy_potion'}:
                    field = {'buy_card': 'card_id', 'buy_relic': 'relic_id', 'buy_potion': 'potion_id'}[choice['action']]
                    choice[field] = option.get('label')
                if choice.get('action') == 'pick_relic':
                    choice['relic_id'] = option.get('id')
                append(choice, {'index': option.get('index', i)})
    if not choices:
        raise ValueError(f'No supported strategy candidates for {state["type"]}')
    return {'schema_version': SCHEMA_VERSION, 'context': context,
            'choices': choices, 'commands': commands}


def act_strategy(model, client, state, deterministic=True):
    obs = strategy_observation(state)
    if state.get('type') == 'card_select':
        # The game expects one atomic list, while simulator choices are sequential.
        count = max(0, int(state.get('min_select', 1)))
        count = min(count, int(state.get('max_select', count)))
        selected = []
        for _ in range(count):
            if not obs['valid'].any():
                raise ValueError('Not enough legal cards to satisfy required selection')
            action, _ = model.predict(obs, action_masks=obs['valid'].astype(bool), deterministic=deterministic)
            command = state['strategy']['commands'][int(action)]
            selected.extend(command.get('indexes', []))
            obs['valid'][int(action)] = 0
        client.choose_many(selected)
    else:
        action, _ = model.predict(obs, action_masks=obs['valid'].astype(bool), deterministic=deterministic)
        send_strategy_choice(client, state, action)


def strategy_observation(state):
    packet = state.get('strategy')
    if packet is None:
        packet = native_packet(state)
        state['strategy'] = packet
    if not packet or packet.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Strategy bridge schema missing or incompatible; rebuild the bridge mod')
    return encode_strategy(packet['context'], packet['choices'])


def send_strategy_choice(client, state, action):
    command = state['strategy']['commands'][int(action)]
    if command.get('skip'):
        client.skip()
    elif 'indexes' in command:
        client.choose_many(command['indexes'])
    else:
        client.choose(command['index'])
