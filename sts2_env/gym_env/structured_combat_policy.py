"""Shared entity encoders and card/target action scores over combat observation v3."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from . import combat_observation_v2 as v2,combat_observation_v3 as v3

W=32
HAND=217+8+6*len(v2.POWER_NAMES)
PILES=HAND+80
CHOICES=PILES+6*len(v2.CARD_NAMES)
PURPOSE=CHOICES+115*4
RESOURCE=PURPOSE+len(v2.CHOICE_PURPOSES)+115
POTIONS=RESOURCE+6+len(v2.RELIC_NAMES)+1
POTION_WIDTH=len(v2.POTION_NAMES)+2
ORBS=POTIONS+9*POTION_WIDTH
ORB_WIDTH=len(v2.ORB_NAMES)+4
RELIC_MOD=v2.OBS_SIZE
HAND_MOD=RELIC_MOD+(len(v2.RELIC_NAMES)+1)*v3.RELIC_SIZE
CHOICE_MOD=HAND_MOD+10*v3.CARD_SIZE
PILE_MOD=HAND_MOD+125*v3.CARD_SIZE
HISTORY=PILE_MOD+3*257
ALLIES=HISTORY+2*len(v3.HISTORY)+1


def network(inputs,outputs):return nn.Sequential(nn.Linear(inputs,64),nn.Tanh(),nn.Linear(64,outputs),nn.Tanh())
def pool(tokens,valid):
    return (tokens*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)


class StructuredCombatExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        if observation_space.shape!=(v3.OBS_SIZE,):raise ValueError('Structured policy requires observation v3')
        super().__init__(observation_space,features_dim=W*(1+10+5+114+9)+1)
        self.card_ids=nn.Embedding(len(v2.CARD_NAMES)+1,16)
        self.card=network(7+v3.CARD_SIZE+16+1,W)
        self.enemy=network(30+4+len(v2.POWER_NAMES),W)
        self.relic_ids=nn.Embedding(len(v2.RELIC_NAMES)+1,16)
        self.relic=network(v3.RELIC_SIZE+16,W)
        self.potion=network(POTION_WIDTH,W)
        self.ally=network(v3.ALLY_SIZE,W)
        self.register_buffer('relic_indices',torch.arange(len(v2.RELIC_NAMES)+1))
        context_width=(11+len(v2.POWER_NAMES)+8+len(v2.CHOICE_PURPOSES)+6+
                       10*ORB_WIDTH+2*len(v3.HISTORY)+3*257+6*16+6*W+1)
        self.context=network(context_width,W)

    def encode_cards(self,raw,modifiers,selected):
        ids=(raw[...,1]*(len(v2.CARD_NAMES)+1)).round().long().clamp(0,len(v2.CARD_NAMES))
        # Identity is categorical; remove the old artificial ordinal ID distance.
        values=torch.cat([raw[...,:1],raw[...,2:],modifiers,self.card_ids(ids),selected.unsqueeze(-1)],dim=-1)
        return self.card(values)

    def forward(self,o):
        n=o.shape[0]
        hand=o[:,HAND:PILES].reshape(n,10,8)
        hand_tokens=self.encode_cards(hand,o[:,HAND_MOD:CHOICE_MOD].reshape(n,10,v3.CARD_SIZE),o.new_zeros(n,10))
        choice=o[:,CHOICES:PURPOSE].reshape(n,115,4)
        choice_raw=torch.cat([choice,o.new_zeros(n,115,4)],dim=-1)
        selected=o[:,PURPOSE+len(v2.CHOICE_PURPOSES):RESOURCE]
        choice_tokens=self.encode_cards(choice_raw,o[:,CHOICE_MOD:PILE_MOD].reshape(n,115,v3.CARD_SIZE),selected)
        enemy_raw=o[:,67:217].reshape(n,5,30)
        powers=o[:,217+8:HAND].reshape(n,6,len(v2.POWER_NAMES))
        enemy_tokens=self.enemy(torch.cat([enemy_raw,o[:,v2.OBS_SIZE-20:v2.OBS_SIZE].reshape(n,5,4),powers[:,1:]],dim=-1))
        relic_raw=o[:,RELIC_MOD:HAND_MOD].reshape(n,len(v2.RELIC_NAMES)+1,v3.RELIC_SIZE)
        relic_ids=self.relic_ids(self.relic_indices).unsqueeze(0).expand(n,-1,-1)
        relic_tokens=self.relic(torch.cat([relic_raw,relic_ids],dim=-1))
        relic_present=torch.maximum(relic_raw[...,0],o[:,RESOURCE+6:POTIONS])
        potions=o[:,POTIONS:ORBS].reshape(n,9,POTION_WIDTH)
        potion_tokens=self.potion(potions)
        allies=o[:,ALLIES:].reshape(n,4,v3.ALLY_SIZE)
        ally_tokens=self.ally(allies)
        counts=o[:,PILES:CHOICES].reshape(n,6,len(v2.CARD_NAMES))
        pile_embeddings=torch.matmul(counts,self.card_ids.weight[1:])/counts.sum(-1,keepdim=True).clamp_min(1e-6)
        context=torch.cat([o[:,:11],powers[:,0],o[:,217:225],
            o[:,PURPOSE:PURPOSE+len(v2.CHOICE_PURPOSES)],o[:,RESOURCE:RESOURCE+6],
            o[:,ORBS:ORBS+10*ORB_WIDTH],o[:,HISTORY:ALLIES-1],o[:,PILE_MOD:HISTORY],
            pile_embeddings.flatten(1),pool(hand_tokens,hand[...,0]),pool(choice_tokens,choice[...,0]),
            pool(enemy_tokens,enemy_raw[...,0]),pool(relic_tokens,relic_present),
            pool(potion_tokens,1-potions[...,0]),pool(ally_tokens,allies[...,0]),o[:,ALLIES-1:ALLIES]],dim=-1)
        global_token=self.context(context)
        return torch.cat([global_token,hand_tokens.flatten(1),enemy_tokens.flatten(1),
                          choice_tokens[:,:114].flatten(1),potion_tokens.flatten(1),o[:,220:221]],dim=-1)


class SharedActionHeads(nn.Module):
    latent_dim_pi=115
    latent_dim_vf=64
    def __init__(self):
        super().__init__()
        self.end=nn.Sequential(nn.Linear(W,64),nn.Tanh(),nn.Linear(64,1))
        self.self_card=nn.Sequential(nn.Linear(2*W,64),nn.Tanh(),nn.Linear(64,1))
        self.target_card=nn.Sequential(nn.Linear(3*W,64),nn.Tanh(),nn.Linear(64,1))
        self.selection=nn.Sequential(nn.Linear(2*W,64),nn.Tanh(),nn.Linear(64,1))
        self.potion=nn.Sequential(nn.Linear(3*W,64),nn.Tanh(),nn.Linear(64,1))
        self.critic=network(3*W,64)

    def parts(self,features):
        n=features.shape[0];tokens=features[:,:-1].reshape(n,139,W)
        return tokens[:,0],tokens[:,1:11],tokens[:,11:16],tokens[:,16:130],tokens[:,130:139]

    def forward_actor(self,features):
        g,hand,enemies,choices,potions=self.parts(features);n=len(g)
        self_scores=self.self_card(torch.cat([hand,g[:,None,:].expand(-1,10,-1)],-1)).squeeze(-1)
        targeted=self.target_card(torch.cat([hand[:,:,None,:].expand(-1,-1,5,-1),enemies[:,None,:,:].expand(-1,10,-1,-1),g[:,None,None,:].expand(-1,10,5,-1)],-1)).reshape(n,50)
        targets=torch.cat([g[:,None,:],enemies],1)
        potion_scores=self.potion(torch.cat([potions[:,:,None,:].expand(-1,-1,6,-1),targets[:,None,:,:].expand(-1,9,-1,-1),g[:,None,None,:].expand(-1,9,6,-1)],-1)).reshape(n,54)
        ordinary=torch.cat([self.end(g),self_scores,targeted,potion_scores],-1)
        select_scores=self.selection(torch.cat([choices,g[:,None,:].expand(-1,114,-1)],-1)).squeeze(-1)
        selection=torch.cat([self.end(g),select_scores],-1)
        return torch.where(features[:,-1:].bool(),selection,ordinary)

    def forward_critic(self,features):
        g,hand,enemies,_,_=self.parts(features)
        return self.critic(torch.cat([g,hand.mean(1),enemies.mean(1)],-1))

    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class StructuredCombatPolicy(MaskableActorCriticPolicy):
    def __init__(self,*args,**kwargs):
        kwargs['features_extractor_class']=StructuredCombatExtractor
        super().__init__(*args,**kwargs)
    def _build_mlp_extractor(self):self.mlp_extractor=SharedActionHeads()
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.action_net=nn.Identity()
        for head in (self.mlp_extractor.end,self.mlp_extractor.self_card,self.mlp_extractor.target_card,self.mlp_extractor.selection,self.mlp_extractor.potion):
            nn.init.orthogonal_(head[-1].weight,gain=.01);nn.init.zeros_(head[-1].bias)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
