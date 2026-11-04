"""Public state extension; v1/v2 remain frozen for checkpoint compatibility."""
import hashlib
import json
from pathlib import Path
import numpy as np
from . import combat_observation_v2 as v2

VERSION = 3
vocab = json.loads(Path(__file__).with_name('combat_modifiers.json').read_text())
ENCHANTMENTS = tuple(v2.identifier(s) for s in vocab['enchantments'])
AFFLICTIONS = tuple(v2.identifier(s) for s in vocab['afflictions'])
HISTORY = ('plays_turn', 'attacks_turn', 'skills_turn', 'powers_turn', 'plays_combat',
           'attacks_combat', 'skills_combat', 'powers_combat', 'draws_turn',
           'discards_turn', 'exhausts_turn', 'energy_spent_turn', 'plays_last_turn')
USAGE = ('used_up', 'used_this_combat', 'used_this_turn', 'triggered_this_turn')
CARD_SIZE = 27 + len(ENCHANTMENTS) + 1 + len(AFFLICTIONS) + 1
RELIC_SIZE = 9 + 2*len(USAGE)
ALLY_SIZE = 7 + len(v2.POWER_NAMES)
EXTRA_SIZE = (len(v2.RELIC_NAMES)+1)*RELIC_SIZE + 125*CARD_SIZE + 3*257 + 2*len(HISTORY) + 1+4*ALLY_SIZE
OBS_SIZE = v2.OBS_SIZE + EXTRA_SIZE
SCHEMA_SHA256 = hashlib.sha256(json.dumps([VERSION, v2.SCHEMA_SHA256, ENCHANTMENTS,
    AFFLICTIONS, HISTORY, USAGE, CARD_SIZE, RELIC_SIZE, ALLY_SIZE, EXTRA_SIZE]).encode()).hexdigest()


def card_features(card):
    variables = {v2.identifier(k):v for k,v in card.get('variables', {}).items()}
    out = np.zeros(CARD_SIZE, np.float32)
    out[:11] = [float('modifiers' in card), float('variables' in card),
        variables.get('DAMAGE',0)/100,
        variables.get('BLOCK',0)/100,
        float(card.get('exhaust_next', False)), float(card.get('ethereal', False)),
        float(card.get('unplayable', False)), float(card.get('sly', False)),
        float(card.get('star_x', False)), float(card.get('bound', False)),
        card.get('replay_count', 0)/10]
    index = 11
    for key, names in (('enchantments', ENCHANTMENTS), ('afflictions', AFFLICTIONS)):
        for name, amount in card.get('modifiers', {}).get(key, {}).items():
            name = v2.identifier(name)
            out[index + (names.index(name) if name in names else len(names))] += amount/10
        index += len(names)+1
    for name, value in variables.items():
        digest = hashlib.blake2b(v2.identifier(name).encode(), digest_size=4).digest()
        out[index+int.from_bytes(digest,'little')%16] += float(value)/100
    return out


def extras(packet):
    out = np.zeros(EXTRA_SIZE, np.float32); index = 0
    # Presence differs from a known zero counter or a known absent modifier.
    for relic in packet.get('relic_states', []):
        name = v2.identifier(relic['id'])
        slot = v2.RELIC_NAMES.index(name) if name in v2.RELIC_NAMES else len(v2.RELIC_NAMES)
        counter = relic.get('counter'); status = relic.get('status')
        offset = slot*RELIC_SIZE
        out[offset:offset+RELIC_SIZE] += [1, float(counter is not None), (counter or 0)/20,
            relic.get('stack', 0)/10, float(relic.get('melted', False)), float(status is not None),
            float(status == 'Normal'), float(status == 'Active'), float(status == 'Disabled')] + [
                value for key in USAGE for value in (float(key in relic.get('usage',{})),float(relic.get('usage',{}).get(key,False)))]
    index += (len(v2.RELIC_NAMES)+1)*RELIC_SIZE
    for cards, capacity in ((packet.get('hand', []),10), ((packet.get('choice') or {}).get('cards', []),115)):
        for card in (cards[:capacity]+[{}]*capacity)[:capacity]:
            out[index:index+CARD_SIZE] = card_features(card); index += CARD_SIZE
    # Unordered full card+modifier signatures distinguish differently modified pile
    # copies without exposing draw order. Fixed buckets have possible collisions.
    for pile in ('draw','discard','exhaust'):
        cards = packet.get('piles', {}).get(pile)
        out[index] = float(cards is not None); index += 1
        for card in cards or []:
            features = list(map(float, card_features(card))) + v2.card_features(card)
            digest = hashlib.blake2b(json.dumps([v2.card_name(card.get('id','')),features]).encode(),digest_size=8).digest()
            out[index+int.from_bytes(digest,'little')%256] += 1/20
        index += 256
    history = packet.get('history', {})
    for name in HISTORY:
        out[index:index+2] = [float(name in history), history.get(name,0)/20]; index += 2
    allies = packet.get('allies')
    out[index] = float(allies is not None); index += 1
    for ally in ((allies or [])[:4]+[{}]*4)[:4]:
        name = v2.identifier(ally.get('id',''))
        out[index:index+7] = [float(bool(ally)), ally.get('hp',0)/100, ally.get('max_hp',0)/100,
            ally.get('block',0)/100, float(ally.get('alive',False)), float(name=='OSTY'), float(bool(name) and name!='OSTY')]
        index += 7
        powers = {v2.identifier(p['id']):p['amount'] for p in ally.get('powers',[])}
        out[index:index+len(v2.POWER_NAMES)] = [powers.get(p,0)/20 for p in v2.POWER_NAMES]
        index += len(v2.POWER_NAMES)
    assert index == EXTRA_SIZE
    return out


def _modifier(value):
    if not value: return {}
    name, _, amount = value.rpartition(':')
    return {name:float(amount)}


def native_packet(state):
    packet = v2.native_packet(state); combat = state['Combat']
    def enrich(card, raw):
        if 'Enchantment' in raw or 'Affliction' in raw:
            card['modifiers'] = {'enchantments':_modifier(raw.get('Enchantment')), 'afflictions':_modifier(raw.get('Affliction'))}
        details=raw.get('Details') or raw
        if details.get('Variables') is not None:
            card['variables'] = details['Variables']
        keywords = raw.get('Keywords', [])
        card.update(exhaust_next=raw.get('ExhaustOnNextPlay',False), ethereal='Ethereal' in keywords,
                    unplayable='Unplayable' in keywords, sly='Sly' in keywords,
                    star_x=raw.get('StarCostsX',False),replay_count=raw.get('ReplayCount',0),
                    bound=v2.identifier((raw.get('Affliction') or '').partition(':')[0])=='BOUND')
    by_id = {c['Id']:c for c in combat['Cards']}
    for pile in combat['Piles']:
        encoded = packet['hand'] if pile['Type']=='Hand' else packet['piles'].get(pile['Type'].lower(),[])
        for card, ident in zip(encoded,pile['CardIds']): enrich(card,by_id[ident])
    if packet.get('choice'):
        for option in state['Options']:
            if option['Action']['Kind']==10 and option.get('Card'):
                enrich(packet['choice']['cards'][option['Action']['Index']],option['Card'])
    if 'RelicStates' in state:
        packet['relic_states'] = [dict(id=r['Model'],counter=r.get('Counter'),stack=r['Stack'],melted=r['IsMelted'],status=r.get('Status'),usage=r.get('Usage') or {}) for r in state['RelicStates']]
    enemies = {e['Creature']['Id'] for e in combat['Enemies']}
    packet['allies'] = [dict(id=c['Model'],hp=c['Hp'],max_hp=c['MaxHp'],block=c['Block'],alive=c['Hp']>0,
        powers=[{'id':p['Model'],'amount':p['Amount']} for p in c['Powers']]) for c in combat['Creatures'][1:] if c['Id'] not in enemies]
    if 'PublicHistory' in combat: packet['history'] = combat['PublicHistory']
    return packet


def simulator_packet(combat):
    packet = v2.simulator_packet(combat)
    def enrich(card, raw):
        card.update(modifiers={'enchantments':dict(raw.enchantments),'afflictions':{raw.affliction:raw.affliction_amount} if raw.affliction else {}},
            variables=dict(raw.effect_vars,Damage=raw.base_damage if raw.base_damage is not None else raw.effect_vars.get('damage',0),Block=raw.base_block or raw.effect_vars.get('block',0)),
            ethereal='ethereal' in raw.keywords,unplayable='unplayable' in raw.keywords,
            sly=raw.is_sly,star_x=raw.has_star_cost_x,bound=raw.bound,replay_count=raw.base_replay_count)
    for card, raw in zip(packet['hand'],combat.hand): enrich(card,raw)
    for name in ('draw','discard','exhaust'):
        for card, raw in zip(packet['piles'][name],getattr(combat,name+'_pile')): enrich(card,raw)
    if combat.pending_choice:
        for card, option in zip(packet['choice']['cards'],combat.pending_choice.options): enrich(card,option.card)
    # Only audited equivalents of public relic display counters are read.
    counters = {'PEN_NIB':('_attacks_played',10),'NUNCHAKU':('_attacks_played',10),'HAPPY_FLOWER':('_turns_seen',3)}
    packet['relic_states'] = []
    for relic in combat.relics:
        name = relic.relic_id.name; counter = None
        if name in counters:
            field, modulus = counters[name]; counter = getattr(relic,field)%modulus
        packet['relic_states'].append(dict(id=name,counter=counter,stack=1,melted=relic.is_melted,
            status='Disabled' if relic.is_used_up else None,usage=dict(used_up=relic.is_used_up,
                **{key:getattr(relic,'_'+key) for key in USAGE[1:] if hasattr(relic,'_'+key)})))
    packet['allies'] = [dict(id='OSTY' if a.is_osty else a.monster_id or 'ALLY',hp=a.current_hp,max_hp=a.max_hp,
        block=a.block,alive=a.is_alive,powers=[{'id':p.name,'amount':v.amount} for p,v in a.powers.items()]) for a in combat.allies]
    history = {}
    for suffix,cards in (('turn',combat._played_cards_this_turn),('combat',combat._played_cards_combat)):
        cards = [c for c in cards if c.owner is combat.player]
        history['plays_'+suffix] = len(cards)
        for kind in ('attacks','skills','powers'):
            history[kind+'_'+suffix] = sum(c.card_type.name=={'attacks':'ATTACK','skills':'SKILL','powers':'POWER'}[kind] for c in cards)
    history.update(draws_turn=sum(owner is combat.player for owner,_,_ in combat._draw_events_this_turn),
        discards_turn=sum(c.owner is combat.player for c in combat._discard_events_this_turn),
        exhausts_turn=sum(c.owner is combat.player for c in combat._exhaust_events_this_turn),
        energy_spent_turn=combat._energy_spent_this_turn.get(combat.player,0),
        plays_last_turn=combat.count_cards_played_last_round(combat.player))
    packet['history'] = history
    return packet


def encode(combat):
    return np.concatenate([v2.encode(combat),extras(simulator_packet(combat))])
