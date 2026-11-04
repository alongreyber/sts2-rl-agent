import copy
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sts2_env.gym_env.teacher_ppo import TeacherAnchoredPPO


class MaskedCart(gym.Wrapper):
    def __init__(self):super().__init__(gym.make('CartPole-v1'))
    def action_masks(self):return np.ones(2,bool)


def test_no_anchor_matches_upstream_ppo_updates():
    torch.set_num_threads(1)
    results=[]
    for cls in (MaskablePPO,TeacherAnchoredPPO):
        env=MaskedCart();m=cls('MlpPolicy',env,n_steps=8,batch_size=4,n_epochs=1,seed=42)
        m.learn(16);results.append(copy.deepcopy(m.policy.state_dict()));env.close()
    for name in results[0]:torch.testing.assert_close(results[0][name],results[1][name],rtol=0,atol=0)


def test_anchor_gradient_only_updates_student_and_save_is_standard(tmp_path):
    env=MaskedCart();m=TeacherAnchoredPPO('MlpPolicy',env,n_steps=8,batch_size=4,n_epochs=1,seed=42)
    teacher=copy.deepcopy(m.policy);original=copy.deepcopy(teacher.state_dict())
    m.attach_teacher(teacher,.1)
    with torch.no_grad():m.policy.action_net.bias[0]+=.5
    loss=m.teacher_kl(torch.zeros((2,4),device=m.device),np.ones((2,2),bool))
    assert loss.item()>0
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert m.policy.action_net.bias.grad.abs().sum()>0
    m.learn(16)
    for name in original:torch.testing.assert_close(original[name],teacher.state_dict()[name],rtol=0,atol=0)
    m.save(tmp_path/'model');loaded=MaskablePPO.load(tmp_path/'model')
    for name,p in m.policy.state_dict().items():torch.testing.assert_close(p,loaded.policy.state_dict()[name])
    env.close()
