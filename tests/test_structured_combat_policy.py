import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.structured_combat_policy import StructuredCombatPolicy


def test_hand_slot_permutation_permutes_action_scores():
    torch.set_num_threads(1)
    env=STS2CombatEnv(observation_version=3);obs,_=env.reset(seed=42)
    m=MaskablePPO(StructuredCombatPolicy,env,device='cpu',seed=42)
    def logits(o):
        t,_=m.policy.obs_to_tensor(o)
        with torch.no_grad():return m.policy.get_distribution(t).distribution.logits[0]
    before=logits(obs)
    other=next(i for i,c in enumerate(env.combat.hand) if c.card_id!=env.combat.hand[0].card_id)
    env.combat.hand[0],env.combat.hand[other]=env.combat.hand[other],env.combat.hand[0]
    after=logits(env._encode(env.combat))
    permutation=list(range(115));permutation[1],permutation[1+other]=1+other,1
    start=11+5*other
    permutation[11:16],permutation[start:start+5]=permutation[start:start+5],permutation[11:16]
    torch.testing.assert_close(after,before[permutation],rtol=1e-5,atol=1e-6)
    env.close()


def test_enemy_permutation_permutes_only_targeted_action_scores():
    torch.set_num_threads(1)
    env=STS2CombatEnv(observation_version=3)
    for seed in range(30):
        obs,_=env.reset(seed=seed)
        if len(env.combat.enemies)>=2:break
    assert len(env.combat.enemies)>=2
    m=MaskablePPO(StructuredCombatPolicy,env,device='cpu',seed=42)
    def logits(o):
        t,_=m.policy.obs_to_tensor(o)
        with torch.no_grad():return m.policy.get_distribution(t).distribution.logits[0]
    before=logits(obs)
    env.combat.enemies[0],env.combat.enemies[1]=env.combat.enemies[1],env.combat.enemies[0]
    after=logits(env._encode(env.combat));permutation=list(range(115))
    for hand in range(10):
        a=11+hand*5;permutation[a],permutation[a+1]=a+1,a
    for potion in range(9):
        a=61+potion*6+1;permutation[a],permutation[a+1]=a+1,a
    torch.testing.assert_close(after,before[permutation],rtol=1e-5,atol=1e-6)
    env.close()


def test_structured_policy_supports_masked_ppo_and_checkpoint_loading(tmp_path):
    torch.set_num_threads(1)
    env=ActionMasker(STS2CombatEnv(observation_version=3),lambda e:e.action_masks())
    m=MaskablePPO(StructuredCombatPolicy,env,n_steps=8,batch_size=4,n_epochs=1,device='cpu',seed=42)
    m.learn(16);m.save(tmp_path/'structured')
    loaded=MaskablePPO.load(tmp_path/'structured',device='cpu')
    obs,_=env.reset(seed=42);mask=env.action_masks()
    a,_=loaded.predict(obs,action_masks=mask,deterministic=True)
    assert mask[int(a)] and isinstance(loaded.policy,StructuredCombatPolicy)
    assert all(torch.isfinite(p).all() for p in loaded.policy.parameters())
    for k,v in m.policy.state_dict().items():torch.testing.assert_close(v,loaded.policy.state_dict()[k])
    env.close()
