import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sts2_env.gym_env.full_run_combat_env import FullRunCombatEnv
from sts2_env.run.run_manager import RunManager


class Strategy:
    calls=0
    def predict(self,obs,action_masks,**kwargs):
        self.calls+=1
        return int(np.flatnonzero(action_masks)[0]),None


def test_combat_environment_advances_strategy_and_finishes_whole_run():
    strategy=Strategy();env=FullRunCombatEnv(strategy,max_steps=250)
    obs,_=env.reset(seed=42,options={'run_seed':42})
    assert strategy.calls>0
    assert env.run._mgr.phase==RunManager.PHASE_COMBAT
    assert env.observation_space.contains(obs)
    for _ in range(251):
        legal=np.flatnonzero(env.action_masks())
        action=int(next((i for i in legal if i!=0),0))
        obs,reward,terminated,truncated,info=env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:break
    assert terminated or truncated
    assert info['reward_components']['outcome'] in (-1,1)
    assert sum(info['combat_decisions_by_act'].values())>0
    assert info['primitive_steps']>=sum(info['combat_decisions_by_act'].values())
    with pytest.raises(Exception,match='reset'):env.step(0)
    env.close()


def test_training_and_validation_seed_allocation_are_disjoint():
    import sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from train_alternating import TrainingSeeds,VALIDATION_START
    env=FullRunCombatEnv(Strategy(),max_steps=50)
    state={'next_training_seed':VALIDATION_START-1}
    wrapper=TrainingSeeds(env,state)
    wrapper.reset()
    assert state['next_training_seed']==VALIDATION_START
    with pytest.raises(ValueError,match='exhausted'):wrapper.reset()
    wrapper.close()


def test_parallel_seed_allocator_never_reuses_validation_or_training_seeds():
    import sys
    import threading
    from types import SimpleNamespace
    from concurrent.futures import ThreadPoolExecutor
    import gymnasium as gym
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from train_alternating import SharedTrainingSeeds,VALIDATION_START
    class SeedEnv(gym.Env):
        def reset(self,seed=None,options=None): return options['run_seed'],{}
    counter=SimpleNamespace(value=VALIDATION_START-64)
    lock=threading.Lock()
    workers=[SharedTrainingSeeds(SeedEnv(),counter,lock) for _ in range(8)]
    with ThreadPoolExecutor(8) as pool:
        seeds=list(pool.map(lambda i:workers[i%8].reset()[0],range(64)))
    assert len(set(seeds))==64
    assert max(seeds)<VALIDATION_START
    with pytest.raises(ValueError,match='exhausted'):workers[0].reset()
