"""Evaluate one combat checkpoint with a fixed strategy and optional imitation diagnostics."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiment_combat import evaluate
from sts2_env.demonstrations import load_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--combat',type=Path,required=True)
    p.add_argument('--strategy',type=Path,default=Path('output/alternating_rewards_v2/teacher_strategy.zip'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--eval-runs',type=int,default=30)
    p.add_argument('--seed-start',type=int,default=1000000)
    a=p.parse_args()
    if a.eval_runs<1:p.error('eval-runs must be positive')
    torch.set_num_threads(1)
    model=MaskablePPO.load(a.combat,device='cpu');strategy=MaskablePPO.load(a.strategy,device='cpu')
    result=evaluate(model,strategy,range(a.seed_start,a.seed_start+a.eval_runs),time.monotonic()+7200)
    result['parameters']=sum(x.numel() for x in model.policy.parameters())
    result['combat_sha256']=hashlib.sha256(a.combat.read_bytes()).hexdigest()
    result['strategy_sha256']=hashlib.sha256(a.strategy.read_bytes()).hexdigest()
    if a.data:
        data,rejected=load_dataset(a.data,'combat');correct=0;loss=0.
        with torch.no_grad():
            for start in range(0,len(data),64):
                batch=data[start:start+64];obs=np.stack([x[0] for x in batch]);tensor,_=model.policy.obs_to_tensor(obs)
                labels=torch.tensor([x[2] for x in batch]);dist=model.policy.get_distribution(tensor,action_masks=np.stack([x[1] for x in batch]))
                correct+=int((dist.distribution.probs.argmax(1)==labels).sum());loss+=float(-dist.log_prob(labels).sum())
        result['imitation_agreement']=correct/len(data) if data else None
        result['imitation_nll']=loss/len(data) if data else None
        result['imitation_examples']=len(data);result['rejected']=rejected
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print('SUMMARY',json.dumps({k:v for k,v in result.items() if k not in ('runs','rejected')}),flush=True)


if __name__=='__main__':main()
