"""Masked behavior-cloning warm start; saves a standard PPO checkpoint."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sts2_env.demonstrations import load_dataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--role',choices=('combat','strategy'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--include-automatic',action='store_true');p.add_argument('--initial',type=Path);p.add_argument('--observation-version',type=int,choices=(1,2,3),default=3);p.add_argument('--architecture',choices=('mlp','structured'),default='mlp');p.add_argument('--seed',type=int,default=42);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    if a.architecture=='structured' and (a.role!='combat' or a.observation_version!=3):p.error('Structured architecture requires combat observation v3')
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    print(f'[{a.role}] Loading teacher examples from {a.data}',flush=True)
    train,rejected=load_dataset(a.data,a.role,include_automatic=a.include_automatic,observation_version=a.observation_version);validation,vr=load_dataset(a.data,a.role,'validation',a.include_automatic,a.observation_version)
    print(f'[{a.role}] train={len(train)} validation={len(validation)} rejected={len(rejected)+len(vr)}',flush=True)
    report={'train_examples':len(train),'validation_examples':len(validation),'rejected':rejected+vr,'role':a.role,'observation_version':a.observation_version,'architecture':a.architecture,'seed':a.seed}
    if a.role=='combat' and a.observation_version>=2:
        from sts2_env.gym_env.observation_versions import schema
        SCHEMA_SHA256=schema(a.observation_version).SCHEMA_SHA256;OBS_SIZE=schema(a.observation_version).OBS_SIZE
        report.update(observation_schema_sha256=SCHEMA_SHA256,observation_size=OBS_SIZE)
    if not a.validate_only:
        if not train:raise ValueError('No eligible training examples')
        if a.role=='combat':
            from sts2_env.gym_env.combat_env import STS2CombatEnv
            env=STS2CombatEnv(observation_version=a.observation_version);policy='MlpPolicy';kwargs={}
            if a.architecture=='structured':
                from sts2_env.gym_env.structured_combat_policy import StructuredCombatPolicy
                policy=StructuredCombatPolicy
        else:
            from sts2_env.gym_env.strategy_env import StrategyEnv
            from sts2_env.gym_env.strategy_policy import StrategyExtractor
            env=StrategyEnv();policy='MultiInputPolicy';kwargs={'features_extractor_class':StrategyExtractor}
        model=MaskablePPO.load(a.initial,env=env,device='cpu') if a.initial else MaskablePPO(policy,env,policy_kwargs=kwargs,device='cpu')
        if a.role=='combat':
            from sts2_env.gym_env.structured_combat_policy import StructuredCombatPolicy
            actual='structured' if isinstance(model.policy,StructuredCombatPolicy) else 'mlp'
            if a.architecture=='structured' and actual!='structured':raise ValueError('Initial checkpoint has a different architecture')
            report['architecture']=actual
        torch.set_num_threads(1);torch.manual_seed(a.seed);optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-4)
        print(f'[{a.role}] Starting {a.epochs} epochs on CPU',flush=True)
        started=time.monotonic();last_progress=started
        for epoch in range(a.epochs):
            total_loss=0.;seen=0
            for start in range(0,len(train),32):
                batch=train[start:start+32];observations=[r[0] for r in batch]
                obs={k:np.stack([o[k] for o in observations]) for k in observations[0]} if isinstance(observations[0],dict) else np.stack(observations)
                tensor,_=model.policy.obs_to_tensor(obs);distribution=model.policy.get_distribution(tensor,action_masks=np.stack([r[1] for r in batch]))
                loss=-distribution.log_prob(torch.tensor([r[2] for r in batch],device=model.device)).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);optimizer.step()
                total_loss+=loss.detach().item()*len(batch);seen+=len(batch)
                if time.monotonic()-last_progress>=10:
                    print(f'[{a.role}] epoch {epoch+1}/{a.epochs} examples {seen}/{len(train)} loss={total_loss/seen:.5f}',flush=True)
                    last_progress=time.monotonic()
            print(f'[{a.role}] epoch {epoch+1}/{a.epochs} loss={total_loss/seen:.5f} elapsed={time.monotonic()-started:.1f}s',flush=True)
        print(f'[{a.role}] Saving checkpoint and evaluating held-out teacher agreement',flush=True)
        a.output.parent.mkdir(parents=True,exist_ok=True);model.save(a.output)
        if a.role=='combat':a.output.with_suffix('.observation.json').write_text(json.dumps({'version':a.observation_version,
            'size':model.observation_space.shape[0],'schema_sha256':report.get('observation_schema_sha256')},indent=2))
        report['validation_agreement']=sum(int(model.predict(o,action_masks=m,deterministic=True)[0])==label for o,m,label in validation)/len(validation) if validation else None
        env.close()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
