"""Persistent CPU checkpoint worker; uses the training repo's exact encoder."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

CARD_ALIASES = {'SLOTH': 'SLOTH_STATUS'}


def _encode_legacy(state):
    from sts2_env.gym_env.observation import encode_observation
    from sts2_env.core.enums import CardId, IntentType, PowerId
    c = state['Combat']
    def creature(v):
        powers = {p['Model'].removesuffix('_POWER'): p['Amount'] for p in v['Powers']}
        return NS(current_hp=v['Hp'], max_hp=v['MaxHp'], block=v['Block'],
                  is_alive=v['Hp'] > 0, combat_id=v['Id'],
                  get_power_amount=lambda p: powers.get(p.name, 0))
    cards = {v['Id']: v for v in c['Cards']}
    piles = {p['Type']: p['CardIds'] for p in c['Piles']}
    hand = []
    for i in piles.get('Hand', []):
        card = cards[i]; d = card['Details']; v = d['Variables']
        model_id=CARD_ALIASES.get(card['Model'],card['Model'])
        if model_id not in CardId.__members__ and model_id+'_CARD' in CardId.__members__:
            model_id += '_CARD'
        if model_id not in CardId.__members__:
            raise ValueError('Unknown checkpoint card: '+card['Model'])
        hand.append(NS(card_id=CardId[model_id], cost=card['Cost'],
                       base_damage=v.get('Damage', 0), base_block=v.get('Block', 0), is_attack=d['Type']=='Attack'))
    ais = {}; enemies = []
    for e in c['Enemies']:
        ec = creature(e['Creature']); enemies.append(ec); intents = []
        for d in e.get('IntentDetails', []):
            kind = d['Type'].removesuffix('Intent')
            kind = {'SingleAttack':'ATTACK','MultiAttack':'MULTI_ATTACK','DebuffStrong':'DEBUFF'}.get(kind, kind.upper())
            intents.append(NS(intent_type=IntentType.__members__.get(kind), damage=d.get('Damage') or 0, hits=d.get('Hits') or 1))
        ais[ec.combat_id] = NS(current_move=NS(intents=intents))
    return encode_observation(NS(player=creature(c['Creatures'][0]), energy=c['Energy'],
        max_energy=state['MaxEnergy'], stars=c.get('Stars',0), hand=hand,
        draw_pile=piles.get('Draw',[]), discard_pile=piles.get('Discard',[]), exhaust_pile=piles.get('Exhaust',[]),
        enemies=enemies, enemy_ais=ais))


def bindings(state):
    from sts2_env.core.constants import MAX_HAND_SIZE, MAX_ENEMIES, ACTION_SPACE_SIZE, POTION_ACTION_START, POTION_TARGET_OPTIONS
    c=state['Combat']; hand=next(p['CardIds'] for p in c['Piles'] if p['Type']=='Hand')
    enemies=[e['Creature']['Id'] for e in c['Enemies']]
    result={}
    for option in state['Options']:
        a=option['Action']; kind=a['Kind']; idx=None
        if kind==2:idx=0
        elif kind==1:
            h=hand.index(a['CardId']); target=a['TargetId']
            if h>=MAX_HAND_SIZE:continue
            if target is None:idx=1+h
            elif target in enemies and enemies.index(target)<MAX_ENEMIES:
                idx=1+MAX_HAND_SIZE+h*MAX_ENEMIES+enemies.index(target)
            else:continue
        elif kind==12:
            target=a['TargetId']; offset=0
            if target not in (None,c['Creatures'][0]['Id']):
                if target not in enemies or enemies.index(target)>=MAX_ENEMIES:continue
                offset=1+enemies.index(target)
            idx=POTION_ACTION_START+a['Index']*POTION_TARGET_OPTIONS+offset
        elif state['Decision'].startswith('Cards:'):
            if kind==11:idx=0
            elif kind==10 and not option['Label'].startswith('selected '):idx=1+a['Index']
        if idx is not None and 0<=idx<ACTION_SPACE_SIZE:
            if idx in result and result[idx]!=a:raise ValueError('Ambiguous checkpoint action mapping')
            result[idx]=a
    if not result:raise ValueError('No representable legal combat actions: '+state['Decision'])
    return result



def encode(state,version=1):
    legacy=_encode_legacy(state)
    if version==1:return legacy
    if version not in (2,3):raise ValueError('Unknown observation version')
    import numpy as np
    from sts2_env.gym_env import combat_observation_v2 as v2
    result=np.concatenate([legacy,v2.extras(v2.native_packet(state))])
    if version==3:
        from sts2_env.gym_env import combat_observation_v3 as v3
        result=np.concatenate([result,v3.extras(v3.native_packet(state))])
    return result
