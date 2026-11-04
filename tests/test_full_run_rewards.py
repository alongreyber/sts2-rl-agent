from types import SimpleNamespace
import pytest
from sts2_env.gym_env.full_run_rewards import FullRunRewards, completed_rounds


def total(floors,rounds,win):
    tracker=FullRunRewards()
    return tracker.advance(floors,rounds,terminal=True,win=win)[0]


def test_revisiting_floor_and_repeated_actions_cannot_farm_progress():
    r=FullRunRewards()
    first,_=r.advance(3)
    assert first==pytest.approx(.25*3/60)
    for floor in (3,2,1,3): assert r.advance(floor)[0]==0
    assert r.advance(4)[0]==pytest.approx(.25/60)


def test_longer_same_floor_loss_is_never_better():
    assert total(5,20,False)<total(5,5,False)
    assert total(5,1000,False)==pytest.approx(total(5,100,False))
    assert total(8,10,False)>total(3,10,False)


def test_bounded_shaping_keeps_every_win_above_every_loss():
    lowest_win=total(0,100000,True)
    highest_loss=total(100000,0,False)
    assert lowest_win==pytest.approx(.9)
    assert highest_loss==pytest.approx(-.75)
    assert lowest_win>highest_loss


def test_split_macro_steps_and_primitive_steps_have_same_return():
    r=FullRunRewards();value=0
    for floor,rounds,terminal in [(1,2,False),(1,3,False),(2,0,False),(5,4,True)]:
        value+=r.advance(floor,rounds,terminal,False)[0]
    assert value==pytest.approx(total(5,9,False))
    with pytest.raises(RuntimeError,match='terminal'):r.advance(5,terminal=True)


def test_round_cost_does_not_charge_selection_confirmation_or_card_plays():
    c=SimpleNamespace(turn_count=2)
    assert completed_rounds(c,2,False)==0
    assert completed_rounds(c,2,True)==1  # Died during end turn; counter never advanced.
    c.turn_count=3
    assert completed_rounds(c,2,True)==1


def test_game_limit_is_terminal_loss_without_time_limit_bootstrap():
    from sts2_env.gym_env.full_run_combat_env import FullRunCombatEnv
    import numpy as np
    class Strategy:
        def predict(self,obs,action_masks,**kwargs):
            return int(np.flatnonzero(action_masks)[0]),None
    env=FullRunCombatEnv(Strategy(),max_steps=1)
    env.reset(seed=42,options={'run_seed':42})
    _,_,terminated,truncated,info=env.step(0)
    assert terminated and not truncated and info['timeout']
    assert info['reward_components']['outcome']==-1
    env.close()


def test_strategy_game_limit_also_terminates_without_bootstrap():
    from sts2_env.gym_env.strategy_env import StrategyEnv
    import numpy as np
    env=StrategyEnv(max_steps=1)
    env.reset(seed=42,options={'run_seed':42})
    _,_,terminated,truncated,info=env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert terminated and not truncated and info['timeout']
    assert info['reward_components']['outcome']==-1
    env.close()


@pytest.mark.parametrize('starting_credit',[0.,.195,.20])
def test_optional_quality_reward_is_awarded_at_actual_combat_boundary(starting_credit):
    import numpy as np
    from sts2_env.gym_env.full_run_combat_env import FullRunCombatEnv
    from sts2_env.gym_env.action_space import action_to_card_and_target,is_potion_action
    class Strategy:
        def predict(self,obs,action_masks,**kwargs):return int(np.flatnonzero(action_masks)[0]),None
    env=FullRunCombatEnv(Strategy(),combat_quality=True)
    env.reset(seed=42,options={'run_seed':42});combat=env.run._mgr.get_combat_state()
    env.quality_credit=starting_credit
    for enemy in combat.enemies:enemy.current_hp=1
    for _ in range(50):
        mask=env.action_masks();action=0
        if combat.pending_choice:
            action=int(np.flatnonzero(mask)[0])
        else:
            for candidate in np.flatnonzero(mask):
                if candidate==0 or is_potion_action(candidate):continue
                hand,_=action_to_card_and_target(int(candidate))
                if combat.hand[hand].is_attack:action=int(candidate);break
        _,_,terminated,truncated,info=env.step(action)
        if 'combat_boundary' in info:
            boundary=info['combat_boundary']
            assert boundary['win']
            expected=min(.20-starting_credit,.01+.01*boundary['retained_fraction'])
            assert info['reward_components']['combat_quality']==pytest.approx(expected)
            assert env.quality_credit==pytest.approx(starting_credit+expected)
            break
        assert not terminated and not truncated
    else:raise AssertionError('Controlled encounter never finished')
    env.close()
