import numpy as np
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.combat_observation_v2 import encode,extras,simulator_packet,OBS_SIZE
from sts2_env.core.enums import PowerId
from sts2_env.gym_env.observation import encode_observation
from sts2_env.native_demonstration_combat import encode as native_encode
from sts2_env.bridge.state_adapter import StateAdapter


def fixture():
    env=STS2CombatEnv();env.reset(seed=7);return env.combat


def test_poison_and_scaling_powers_are_observable():
    c=fixture();before=encode(c);legacy=encode_observation(c)
    c.enemies[0].apply_power(PowerId.POISON,12)
    c.player.apply_power(PowerId.NOXIOUS_FUMES,3)
    np.testing.assert_array_equal(legacy,encode_observation(c))
    assert not np.array_equal(before,encode(c))


def test_unknown_draw_order_cannot_change_observation():
    c=fixture();before=encode(c);c.draw_pile.reverse()
    np.testing.assert_array_equal(before,encode(c))
    assert before.shape==(OBS_SIZE,) and np.isfinite(before).all()


def test_potion_identity_and_visible_orb_order_are_observable():
    from sts2_env.potions import create_potion,all_potion_models
    from sts2_env.orbs.base import OrbQueue
    from sts2_env.orbs.all import LightningOrb,FrostOrb
    c=fixture();before=encode(c)
    c.potions.append(create_potion(all_potion_models()[0].potion_id))
    assert not np.array_equal(before,encode(c))
    c.orb_queue=OrbQueue(3);c.orb_queue.orbs=[LightningOrb(),FrostOrb()]
    before=encode(c);c.orb_queue.orbs.reverse()
    assert not np.array_equal(before,encode(c))


def test_absolute_enemy_health_is_not_lost_in_percentage():
    c=fixture();before=encode(c);legacy=encode_observation(c)
    c.enemies[0].max_hp*=2;c.enemies[0].current_hp*=2
    np.testing.assert_array_equal(legacy,encode_observation(c))
    assert not np.array_equal(before,encode(c))


def test_native_and_simulator_v2_exact_parity():
    c=fixture();c.stars=3;c.player.apply_power(PowerId.DEXTERITY,2);c.enemies[0].apply_power(PowerId.POISON,9)
    def creature(e):return {'Id':e.combat_id,'Model':e.monster_id or '', 'Hp':e.current_hp,'MaxHp':e.max_hp,'Block':e.block,'Powers':[{'Model':p.name+'_POWER','Amount':v.amount} for p,v in e.powers.items()]}
    cards=[];piles=[]
    for pile in ('hand','draw','discard','exhaust'):
        ids=[]
        for card in getattr(c,pile if pile=='hand' else pile+'_pile'):
            index=len(cards);ids.append(index)
            cards.append({'Id':index,'Model':card.card_id.name,'Cost':card.cost,'Upgrade':int(card.upgraded),'StarCost':card.star_cost,'Keywords':list(card.keywords),
                          'Details':{'Type':'Attack' if card.is_attack else 'Skill','Variables':{'Damage':card.base_damage or 0,'Block':card.base_block or 0}}})
        piles.append({'Type':pile.title(),'CardIds':ids})
    enemies=[]
    for e in c.enemies:
        intents=c.enemy_ais[e.combat_id].current_move.intents
        enemies.append({'Creature':creature(e),'IntentDetails':[{'Type':{'ATTACK':'SingleAttackIntent','MULTI_ATTACK':'MultiAttackIntent'}.get(i.intent_type.name,i.intent_type.name.title()+'Intent'),'Damage':i.damage,'Hits':i.hits} for i in intents]})
    state={'Decision':'Monster','MaxEnergy':c.max_energy,'Options':[],'Combat':{'Round':c.turn_count,'Energy':c.energy,'Stars':c.stars,'Creatures':[creature(c.player)],'Cards':cards,'Piles':piles,'Enemies':enemies}}
    state.update(Relics=[r.relic_id.name for r in c.relics],Potions=([p.potion_id if p else None for p in c.potions]+[None]*c.max_potion_slots)[:c.max_potion_slots])
    state['Combat'].update(Orbs=[],OrbCapacity=0)
    np.testing.assert_array_equal(native_encode(state,2),encode(c))
    bridge=simulator_packet(c)
    bridge['player'].update(block=c.player.block,energy=c.energy,max_energy=c.max_energy,stars=c.stars)
    for entry,card in zip(bridge['hand'],c.hand):entry.update(base_damage=card.base_damage or 0,base_block=card.base_block or 0,type='Attack' if card.is_attack else 'Skill')
    for entry,enemy,native in zip(bridge['enemies'],c.enemies,enemies):entry.update(block=enemy.block,is_alive=enemy.is_alive,intents=[{'type':i['Type'],'damage':i['Damage'],'hits':i['Hits']} for i in native['IntentDetails']])
    bridge.update(draw_pile_count=len(c.draw_pile),discard_pile_count=len(c.discard_pile),exhaust_pile_count=len(c.exhaust_pile))
    np.testing.assert_array_equal(StateAdapter(observation_version=2).encode_observation(bridge),encode(c))


def test_bridge_legacy_stars_do_not_shift_powers():
    packet={'player':{'hp':70,'max_hp':70,'energy':3,'stars':4,'powers':[{'id':'STRENGTH','amount':6}]},'hand':[],'enemies':[]}
    obs=StateAdapter().encode_observation(packet)
    assert obs[4]==np.float32(.8) and obs[5]==np.float32(.3)


def test_expanded_environment_is_opt_in_and_space_matches():
    env=STS2CombatEnv(observation_version=2);obs,_=env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert obs.shape==(OBS_SIZE,)
    assert STS2CombatEnv().observation_space.shape==(217,)


def test_selection_purpose_and_selected_cards_are_visible():
    from sts2_env.core.selection import PendingCardChoice,CardChoiceOption
    c=fixture()
    c.pending_choice=PendingCardChoice('Discard a card',[CardChoiceOption(c.hand[0],'hand')],lambda cards:None)
    first=encode(c);c.pending_choice.prompt='Retain a card';second=encode(c)
    assert not np.array_equal(first,second)
    c.pending_choice.selected_indices.add(0)
    assert not np.array_equal(second,encode(c))


def test_strategy_rollout_accepts_expanded_frozen_combat_policy():
    from types import SimpleNamespace
    from sts2_env.gym_env.strategy_env import StrategyEnv
    class Model:
        observation_space=SimpleNamespace(shape=(OBS_SIZE,))
        calls=0
        def predict(self,obs,**kwargs):
            assert obs.shape==(OBS_SIZE,)
            self.calls+=1
            return 0,None
    model=Model();env=StrategyEnv(combat_model=model,max_steps=100)
    env.reset(seed=42);env.step(0)
    assert model.calls>0
    env.close()
