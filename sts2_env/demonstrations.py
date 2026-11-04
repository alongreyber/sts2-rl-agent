"""Portable native-state demonstration import; tensor encoding belongs here."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from sts2_env.gym_env.strategy_env import encode_strategy


def strategy(state,options):
    actions=[]
    for o in options:
        a=o['Action'];kind=a['Kind'];card=o.get('Card') or {}
        choice={'action':{0:'move',3:'event_choice',4:'rest_option',5:'buy_card' if card else 'buy_relic',6:'pick_relic',8:'pick_card' if card else 'skip',9:'skip',10:'choose',11:'confirm'}.get(kind,'native'),
                'label':o['Label'],'description':o.get('Description') or (o.get('Event') or {}).get('Description','')}
        if card:choice.update(card_id=card['Model'],upgraded=bool(card.get('Upgrade',0)))
        if o.get('Price') is not None:choice['price']=o['Price']
        if kind==0:
            point=next(p for p in state['Map'] if p['Col']==a['Index'] and p['Row']==a['Option'])
            choice.update(point_type=point['Type'],coord=[point['Col'],point['Row']])
        if kind==4:choice['option_id']=o['Label']
        if kind==6:choice['relic_id']=o['Label']
        if kind==5:
            item=next((x for x in state.get('ShopInventory',[]) if x['Index']==a['Index']),None)
            if item is None:raise ValueError('Missing shop inventory mapping')
            choice['action']={'card':'buy_card','relic':'buy_relic','potion':'buy_potion','removal':'remove_card'}[item['Kind']]
            if item['Kind']!='removal':choice[item['Kind']+'_id']=item['Model']
        if kind==8 and not card and a['Option']>=0:choice['action']='reward_alternative'
        actions.append(choice)
    context={'phase':{'Map':'MAP_CHOICE','Rewards':'CARD_REWARD','RestSite':'REST_SITE','Shop':'SHOP','Event':'EVENT','Treasure':'TREASURE'}.get(state['Decision'],state['Decision']),
        'act':state['Act'],'floor':state['Floor'],'hp':state['Hp'],'max_hp':state['MaxHp'],'gold':state['Gold'],
        'deck':[{'id':c['Model'],'upgraded':bool(c['Upgrade'])} for c in state['Deck']],
        'relics':state['Relics'],'potions':[p for p in state['Potions'] if p]}
    return encode_strategy(context,actions)


def load_dataset(directory,role,split='train',include_automatic=False,observation_version=3):
    from sts2_env.native_demonstration_combat import encode,bindings
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text());rows=[];rejected=[]
    if manifest['schema']!='sts2-demonstrations/v1':raise ValueError('Unsupported dataset schema')
    for entry in manifest['files']:
        path=(directory/entry['path']).resolve()
        if not path.is_relative_to(directory.resolve()):raise ValueError('Unsafe manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Dataset checksum mismatch')
        for line in gzip.open(path,'rt',encoding='utf-8'):
            row=json.loads(line)
            if row['schema']!=manifest['schema'] or not row['execution_verified']:raise ValueError('Unverified/incompatible example')
            group=row['source_group'];validation=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%5==0
            if (split=='validation')!=validation:continue
            if row['role'] not in ({'combat'} if role=='combat' else {'run','strategy'}):continue
            if not include_automatic and row['actor'] not in ('agenticsts_llm','agenticsts_plan'):continue
            try:
                chosen=row['chosen'];options=row['legal_actions'];assert 0<=chosen<len(options)
                state=row['observation'];state=dict(state,Options=options)
                if role=='combat':
                    mapping=bindings(state);action=next(i for i,a in mapping.items() if a==options[chosen]['Action'])
                    from sts2_env.core.constants import ACTION_SPACE_SIZE
                    obs=encode(row['observation'],observation_version);mask=np.zeros(ACTION_SPACE_SIZE,dtype=bool)
                    for i in mapping:mask[i]=True
                else:
                    if options[chosen]['Action']['Kind']==8 and not options[chosen].get('Card') and options[chosen]['Action']['Option']<0:continue
                    obs=strategy(state,options);mask=obs['valid'].astype(bool);action=chosen
                rows.append((obs,mask,action))
            except (ValueError,KeyError,StopIteration,AssertionError) as error:rejected.append({'episode':row['episode_id'],'transition':row['transition_id'],'error':str(error)})
    return rows,rejected
