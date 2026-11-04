"""Append-only public combat features. Draw order and future AI are never read."""
import hashlib,json,re
import numpy as np
from sts2_env.core.enums import PowerId,CardId
from sts2_env.relics.base import RelicId
from sts2_env.potions import all_potion_models
from sts2_env.core.enums import OrbType

VERSION=2
POWER_NAMES=tuple(p.name for p in PowerId)
CARD_NAMES=tuple(c.name for c in CardId)
CHOICE_PURPOSES=('upgrade','remove','transform','discard','exhaust','retain','copy')
RELIC_NAMES=tuple(r.name for r in RelicId)
POTION_NAMES=tuple(sorted(p.potion_id.upper() for p in all_potion_models()))
ORB_NAMES=tuple(o.name for o in OrbType)
# Exported from the repository's reference card metadata; negative status cost is not X.
X_COST_CARDS=frozenset(('CASCADE','DIRGE','ERADICATE','HEAVENLY_DRILL','MALAISE','MULTI_CAST','SKEWER','TEMPEST','VOLLEY','WHIRLWIND'))
RESOURCE_SIZE=6+len(RELIC_NAMES)+1+9*(len(POTION_NAMES)+2)+10*(len(ORB_NAMES)+4)+5*4
EXTRA_SIZE=8+6*len(POWER_NAMES)+10*8+3*2*len(CARD_NAMES)+115*4+len(CHOICE_PURPOSES)+115+RESOURCE_SIZE
OBS_SIZE=217+EXTRA_SIZE
SCHEMA_SHA256=hashlib.sha256(json.dumps([VERSION,POWER_NAMES,CARD_NAMES,RELIC_NAMES,POTION_NAMES,ORB_NAMES,CHOICE_PURPOSES,EXTRA_SIZE]).encode()).hexdigest()

def identifier(value):
    text=str(getattr(value,'name',value))
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])','_',text).upper().removesuffix('_POWER')

def card_name(value):
    value=identifier(value)
    if value=='SLOTH':return 'SLOTH_STATUS'
    return value if value in CARD_NAMES else value+'_CARD' if value+'_CARD' in CARD_NAMES else value

def card_features(card):
    key=card_name(card.get('id',''))
    return [float(key in CARD_NAMES),(CARD_NAMES.index(key)+1)/(len(CARD_NAMES)+1) if key in CARD_NAMES else 0,
            float(card.get('upgraded',False)),card.get('cost',0)/10,card.get('star_cost',0)/10,
            float(card.get('costs_x',False)),float(card.get('retain',False)),float(card.get('exhaust',False))]

def extras(packet):
    out=np.zeros(EXTRA_SIZE,np.float32);player=packet['player'];choice=packet.get('choice')
    out[:8]=[player.get('hp',0)/100,player.get('max_hp',0)/100,packet.get('round',0)/20,
             float(choice is not None), (choice or {}).get('min',0)/10,(choice or {}).get('max',0)/10,
             float('piles' in packet),float('choice' in packet)]
    index=8
    for entity in [player]+packet.get('enemies',[])[:5]+[{}]*max(0,5-len(packet.get('enemies',[]))):
        powers={identifier(p.get('id','')):p['amount'] for p in entity.get('powers',[])}
        out[index:index+len(POWER_NAMES)]=[powers.get(name,0)/20 for name in POWER_NAMES];index+=len(POWER_NAMES)
    for card in (packet.get('hand',[])[:10]+[{}]*10)[:10]:
        out[index:index+8]=card_features(card);index+=8
    for pile in ('draw','discard','exhaust'):
        for card in packet.get('piles',{}).get(pile,[]):
            name=card_name(card.get('id',''))
            if name in CARD_NAMES:out[index+int(bool(card.get('upgraded')))*len(CARD_NAMES)+CARD_NAMES.index(name)]+=1/20
        index+=2*len(CARD_NAMES)
    # Choice order is action-index order, not hand order. Unknown options have presence=0.
    for card in ((choice or {}).get('cards',[])[:115]+[{}]*115)[:115]:
        values=card_features(card);out[index:index+4]=values[:4];index+=4
    prompt=str((choice or {}).get('prompt','')).lower()
    for purpose in CHOICE_PURPOSES:
        out[index]=float(purpose in prompt);index+=1
    for selected in (choice or {}).get('selected',[]):
        if 0<=selected<115:out[index+selected]=1
    index+=115
    out[index:index+6]=[float('relics' in packet),float('potions' in packet),float('orbs' in packet),
                        packet.get('potion_capacity',0)/9,packet.get('orb_capacity',0)/10,float('prompt' in (choice or {}))]
    index+=6
    for relic in packet.get('relics',[]):
        name=identifier(relic)
        out[index+(RELIC_NAMES.index(name) if name in RELIC_NAMES else len(RELIC_NAMES))]+=1
    index+=len(RELIC_NAMES)+1
    potions={p['slot']:p for p in packet.get('potions',[]) if p}
    for slot in range(9):
        name=identifier(potions.get(slot,{}).get('id',''))
        position=0 if not name else POTION_NAMES.index(name)+1 if name in POTION_NAMES else len(POTION_NAMES)+1
        out[index+position]=1;index+=len(POTION_NAMES)+2
    for orb in (packet.get('orbs',[])[:10]+[{}]*10)[:10]:
        name=identifier(orb.get('id','')).removesuffix('_ORB')
        out[index]=float(bool(orb));index+=1
        if name in ORB_NAMES:out[index+ORB_NAMES.index(name)]=1
        out[index+len(ORB_NAMES)]=orb.get('passive',0)/20
        out[index+len(ORB_NAMES)+1]=orb.get('evoke',0)/20
        out[index+len(ORB_NAMES)+2]=float(bool(name) and name not in ORB_NAMES)
        index+=len(ORB_NAMES)+3
    for enemy in (packet.get('enemies',[])[:5]+[{}]*5)[:5]:
        name=identifier(enemy.get('model_id',enemy.get('id','')))
        digest=hashlib.blake2b(name.encode(),digest_size=8).digest() if name else bytes(8)
        out[index:index+4]=[enemy.get('hp',0)/100,enemy.get('max_hp',0)/100,
                            int.from_bytes(digest[:4],'little')/2**32,int.from_bytes(digest[4:],'little')/2**32]
        index+=4
    assert index==EXTRA_SIZE
    return out

def simulator_packet(combat):
    def creature(c):return {'hp':c.current_hp,'max_hp':c.max_hp,'model_id':c.monster_id or '', 'powers':[{'id':p.name,'amount':v.amount} for p,v in c.powers.items()]}
    def card(c):return {'id':c.card_id.name,'upgraded':c.upgraded,'cost':c.cost,'star_cost':c.star_cost,
                        'costs_x':c.has_energy_cost_x,'retain':c.single_turn_retain or 'retain' in c.keywords,'exhaust':'exhaust' in c.keywords}
    pending=combat.pending_choice
    queue=combat.orb_queue
    return {'player':creature(combat.player),'enemies':[creature(e) for e in combat.enemies],
            'relics':[r.relic_id.name for r in combat.relics],
            'potions':[{'slot':i,'id':p.potion_id} for i,p in enumerate(combat.potions) if p],
            'potion_capacity':combat.max_potion_slots,'orb_capacity':queue.capacity if queue else 0,
            'orbs':[{'id':o.orb_type.name,'passive':o.get_passive_value(combat),'evoke':o.get_evoke_value(combat)} for o in queue.orbs] if queue else [],
            'round':combat.turn_count,'hand':[card(c) for c in combat.hand],
            'piles':{name:[card(c) for c in getattr(combat,name+'_pile')] for name in ('draw','discard','exhaust')},
            'choice':None if pending is None else {'min':pending.min_choices,'max':pending.max_choices,
                 'cards':[card(o.card) for o in pending.options],'prompt':pending.prompt,'selected':sorted(pending.selected_indices)}}

def encode(combat):
    from sts2_env.gym_env.observation import encode_observation
    return np.concatenate([encode_observation(combat),extras(simulator_packet(combat))])

def native_packet(state):
    combat=state['Combat'];cards={c['Id']:c for c in combat['Cards']}
    def card(c):return {'id':c['Model'],'upgraded':bool(c.get('Upgrade',0)),'cost':c.get('Cost',0),
                        'star_cost':c.get('StarCost',0),'costs_x':c.get('CostsX',card_name(c['Model']) in X_COST_CARDS),
                        'retain':c.get('RetainThisTurn',False) or 'Retain' in c.get('Keywords',[]),
                        'exhaust':'Exhaust' in c.get('Keywords',[])}
    def creature(c):return {'hp':c['Hp'],'max_hp':c['MaxHp'],'model_id':c['Model'],'powers':[{'id':p['Model'],'amount':p['Amount']} for p in c['Powers']]}
    piles={p['Type'].lower():[card(cards[i]) for i in p['CardIds']] for p in combat['Piles']}
    packet={'player':creature(combat['Creatures'][0]),'enemies':[creature(e['Creature']) for e in combat['Enemies']],
            'round':combat['Round'],'hand':piles.get('hand',[]),'piles':piles,'choice':None}
    if 'Relics' in state:packet['relics']=state['Relics']
    if 'Potions' in state:
        packet.update(potions=[{'slot':i,'id':p} for i,p in enumerate(state['Potions']) if p],potion_capacity=len(state['Potions']))
    if 'Orbs' in combat:
        packet.update(orbs=[{'id':o['Model'],'passive':o['Passive'],'evoke':o['Evoke']} for o in combat['Orbs']],orb_capacity=combat['OrbCapacity'])
    if state['Decision'].startswith('Cards:'):
        import re
        limits=re.search(r'(\d+)\.\.(\d+)',state['Decision'])
        options=[o for o in state['Options'] if o['Action']['Kind']==10]
        ordered=[{} for _ in range(max((o['Action']['Index'] for o in options),default=-1)+1)]
        for o in options:
            if o.get('Card'):ordered[o['Action']['Index']]=card(o['Card'])
        packet['choice']={'min':int(limits[1]) if limits else 1,'max':int(limits[2]) if limits else 1,'cards':ordered,
                          'prompt':state.get('SelectionContext',''),
                          'selected':[o['Action']['Index'] for o in options if o['Label'].startswith('selected ')]}
    return packet
