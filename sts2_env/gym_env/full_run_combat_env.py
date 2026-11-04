"""Combat decisions across whole runs, with a frozen strategy policy between fights."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.gym_env.strategy_env import simulator_snapshot, encode_strategy
from sts2_env.gym_env.observation_versions import schema
from sts2_env.run.run_manager import RunManager
from sts2_env.gym_env.full_run_rewards import FullRunRewards, completed_rounds


class FullRunCombatEnv(gym.Env):
    def __init__(self, strategy_model, character='Silent', max_steps=10000, observation_version=3, combat_quality=False):
        self.run=STS2RunEnv(character_id=character,max_steps=max_steps)
        self.strategy_model=strategy_model
        self.encoder=schema(observation_version)
        self.observation_space=spaces.Box(-np.inf,np.inf,(self.encoder.OBS_SIZE,),np.float32)
        self.action_space=spaces.Discrete(ACTION_SPACE_SIZE)
        self.steps=0
        self.ended=False
        self.act_decisions={1:0,2:0,3:0}
        self.combat_quality=combat_quality

    def _advance_strategy(self):
        mgr=self.run._mgr
        while not mgr.is_over and mgr.phase!=RunManager.PHASE_COMBAT and self.steps<self.run.max_steps:
            context,choices,actions=simulator_snapshot(mgr)
            if not actions: raise RuntimeError(f'No strategy actions in {mgr.phase}')
            obs=encode_strategy(context,choices)
            action,_=self.strategy_model.predict(obs,action_masks=obs['valid'].astype(bool),deterministic=True)
            action=int(action)
            if not 0<=action<len(actions): raise ValueError('Frozen strategy returned an illegal action')
            mgr.take_action(actions[action]); self.steps+=1

    def _obs(self):
        combat=self.run._mgr.get_combat_state()
        if combat is None or self.run._mgr.phase!=RunManager.PHASE_COMBAT:
            return np.zeros(self.observation_space.shape,np.float32)
        if combat is not self.tracked_combat:
            self.tracked_combat=combat
            self.entry_hp=combat.player.current_hp
            self.combat_rounds=0
            self.combat_reported=False
        return self.encoder.encode(combat)

    def action_masks(self):
        if self.run._mgr.is_over or self.steps>=self.run.max_steps:
            mask=np.zeros(ACTION_SPACE_SIZE,bool); mask[0]=True; return mask
        return self.run.action_masks()[:ACTION_SPACE_SIZE].astype(bool)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.steps=0; self.ended=False; self.act_decisions={1:0,2:0,3:0}
        self.tracked_combat=None;self.quality_credit=0.0
        self.run.reset(seed=seed,options=options)
        self.rewards=FullRunRewards(self.run._mgr.run_state.total_floor)
        self._advance_strategy()
        return self._obs(),{}

    def step(self,action):
        if self.ended: raise gym.error.ClosedEnvironmentError('Call reset after terminal combat step')
        mgr=self.run._mgr
        action_act=None
        rounds=0
        boundary=None
        if not mgr.is_over and self.steps<self.run.max_steps:
            if not self.action_masks()[int(action)]: raise ValueError('Illegal combat action')
            act=mgr.run_state.current_act_index+1
            action_act=act
            self.act_decisions[act]=self.act_decisions.get(act,0)+1
            combat=mgr.get_combat_state()
            previous_round=combat.turn_count
            ended_turn=int(action)==0 and combat.pending_choice is None
            self.run._step_combat(int(action)); self.steps+=1
            rounds=completed_rounds(combat,previous_round,ended_turn)
            self.combat_rounds+=rounds
            if combat.is_over and not self.combat_reported:
                self.combat_reported=True
                remaining_hp=combat.player.current_hp
                boundary={'win':bool(combat.player.is_alive),'entry_hp':self.entry_hp,
                          'remaining_hp':remaining_hp,'hp_loss':max(0,self.entry_hp-remaining_hp),
                          'retained_fraction':min(1.,max(0.,remaining_hp/max(1,self.entry_hp))),
                          'rounds':self.combat_rounds,'act':act,'floor':mgr.run_state.total_floor}
            self._advance_strategy()
        game_limit=not mgr.is_over and self.steps>=self.run.max_steps
        terminated=mgr.is_over or game_limit
        truncated=False  # A configured game-limit loss must not bootstrap a continuation value.
        self.ended=terminated
        info={'win':bool(mgr.player_won),'floor':mgr.run_state.total_floor,
              'act':mgr.run_state.current_act_index+1,'primitive_steps':self.steps,
              'timeout':game_limit,'combat_decisions_by_act':dict(self.act_decisions)}
        info['combat_action_act']=action_act
        reward,info['reward_components']=self.rewards.advance(mgr.run_state.total_floor,rounds,self.ended,bool(mgr.player_won))
        if boundary is not None:
            info['combat_boundary']=boundary
        quality=0.0
        if self.combat_quality and boundary is not None and boundary['win']:
            quality=min(max(0.,.20-self.quality_credit),.01+.01*boundary['retained_fraction'])
            self.quality_credit+=quality
        reward+=quality
        info['reward_components']['combat_quality']=quality
        return self._obs(),reward,terminated,truncated,info

    def close(self): self.run.close()
