"""Controlled full-run combat PPO experiments with strategy frozen."""
import argparse,atexit,cProfile,json,multiprocessing as mp,os,pstats,sys,time,traceback
from functools import partial
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sts2_env.gym_env.full_run_combat_env import FullRunCombatEnv
from sts2_env.gym_env.strategy_env import StrategyEnv
from sts2_env.gym_env.teacher_ppo import TeacherAnchoredPPO
from train_alternating import SharedTrainingSeeds,Progress,write

ROOT=Path(__file__).resolve().parents[1]
ARMS={
    'control':dict(conservative=False,quality=False),
    'retention':dict(conservative=True,quality=False),
    'quality':dict(conservative=False,quality=True),
    'combined':dict(conservative=True,quality=True),
}


def worker(strategy_path,quality,counter,lock,log):
    import faulthandler
    sys.stderr=open(log,'a',buffering=1,encoding='utf-8');faulthandler.enable(file=sys.stderr)
    torch.set_num_threads(1)
    frozen=MaskablePPO.load(strategy_path,device='cpu')
    return SharedTrainingSeeds(FullRunCombatEnv(frozen,combat_quality=quality),counter,lock)


def evaluate(model,strategy,seeds,deadline):
    env=FullRunCombatEnv(strategy);rows=[];boundaries=[]
    try:
        for seed in seeds:
            obs,_=env.reset(seed=seed,options={'run_seed':seed})
            reward=0.;combats=[]
            while True:
                if time.monotonic()>deadline:raise TimeoutError('Experiment budget exhausted')
                a,_=model.predict(obs,action_masks=env.action_masks(),deterministic=True)
                obs,r,term,trunc,info=env.step(int(a));reward+=r
                if 'combat_boundary' in info:combats.append(info['combat_boundary'])
                if term or trunc:break
            boundaries+=combats
            rows.append(dict(seed=seed,**info,episode_reward=reward,combats=combats))
            print(f"[eval] seed={seed} floor={info['floor']} win={info['win']}",flush=True)
    finally:env.close()
    wins=[c for c in boundaries if c['win']]
    return dict(runs=rows,mean_floor=float(np.mean([r['floor'] for r in rows])),
        wins=sum(r['win'] for r in rows),act2=sum(r['act']>=2 for r in rows),
        combat_wins=len(wins),combat_count=len(boundaries),
        mean_rounds_per_combat=float(np.mean([c['rounds'] for c in boundaries])) if boundaries else None,
        mean_hp_loss_per_won_combat=float(np.mean([c['hp_loss'] for c in wins])) if wins else None)


class AuditProgress(Progress):
    def __init__(self,deadline):
        super().__init__(deadline,'combat');self.boundaries=[];self.metrics=[];self.reward_components={}
    def _on_step(self):
        for info in self.locals['infos']:
            if 'combat_boundary' in info:self.boundaries.append(info['combat_boundary'])
            for key,value in info.get('reward_components',{}).items():
                self.reward_components[key]=self.reward_components.get(key,0.)+value
        return super()._on_step()
    def _record_metrics(self):
        metrics={k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and isinstance(v,(float,int,np.number))}
        if metrics:self.metrics.append(dict(steps=self.num_timesteps,**metrics))
    def _on_rollout_start(self):self._record_metrics()
    def _on_training_end(self):self._record_metrics()


def profile_strategy(combat,strategy,folder):
    env=StrategyEnv(combat);prof=cProfile.Profile();count=0;started=time.monotonic()
    try:
        prof.enable()
        for seed in range(900000,900010):
            obs,_=env.reset(seed=seed,options={'run_seed':seed})
            while True:
                a,_=strategy.predict(obs,action_masks=env.action_masks(),deterministic=True)
                obs,_,term,trunc,_=env.step(int(a));count+=1
                if term or trunc:break
        prof.disable()
    finally:env.close()
    prof.dump_stats(str(folder/'strategy_profile.pstats'))
    with (folder/'strategy_profile.txt').open('w') as stream:pstats.Stats(prof,stream=stream).sort_stats('cumulative').print_stats(40)
    groups={}
    for (file,line,name),(cc,nc,own,cumulative,callers) in pstats.Stats(prof).stats.items():
        lower=file.replace('\\','/').lower()
        if '/torch/' in lower or 'torch' in name or '/stable_baselines3/' in lower or '/sb3_contrib/' in lower:group='policy_torch_sb3'
        elif 'observation' in lower or lower.endswith('strategy_env.py') and name in ('features','walk','add','encode_strategy','canonical_choice','simulator_snapshot'):group='observation_encoding'
        elif '/sts2_env/' in lower and '/gym_env/' not in lower:group='simulator'
        else:group='other_glue_and_builtins'
        groups[group]=groups.get(group,0.)+own
    return dict(seconds=time.monotonic()-started,strategy_decisions=count,exclusive_seconds=groups,
                note='Single-process CPU strategy rollout; cProfile overhead; exclusive-time categories approximate, no multiprocessing IPC measured.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--combat',type=Path,default=ROOT/'output/alternating_rewards_v2/teacher_combat.zip')
    p.add_argument('--strategy',type=Path,default=ROOT/'output/alternating_rewards_v2/teacher_strategy.zip')
    p.add_argument('--output',type=Path,default=ROOT/'output/combat_experiments')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--eval-runs',type=int,default=30);p.add_argument('--n-envs',type=int,default=8)
    p.add_argument('--hours',type=float,default=2)
    p.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(ARMS))
    p.add_argument('--skip-profile',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.repeats,a.eval_runs,a.n_envs)<=0 or a.hours<=0:p.error('Limits must be positive')
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/'results.json'
    state=json.loads(path.read_text()) if path.exists() else {'arms':{}}
    config=dict(recipe_version=1,steps=a.steps,repeats=a.repeats,eval_runs=a.eval_runs,n_envs=a.n_envs,combat=str(a.combat.resolve()),strategy=str(a.strategy.resolve()),arms={k:ARMS[k] for k in a.arms},
                quality={'win_bonus':.01,'hp_retention_bonus':.01,'run_cap':.2},
                conservative={'learning_rate':3e-5,'epochs':3,'target_kl':.01,'teacher_kl_coef':.1})
    if state.get('config',config)!=config:raise ValueError('Experiment configuration differs; use another output')
    lock_path=a.output/'experiment.lock'
    fd=os.open(lock_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    atexit.register(lambda:lock_path.unlink(missing_ok=True))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    deadline=time.monotonic()+a.hours*3600
    state['config']=config;state['device']=device;state['status']='running';write(path,state)
    strategy=MaskablePPO.load(a.strategy,device='cpu')
    seeds=list(range(1000000,1000000+a.eval_runs))
    try:
        if 'baseline' not in state:
            torch.set_num_threads(1)
            state['baseline']=evaluate(MaskablePPO.load(a.combat,device='cpu'),strategy,seeds,deadline)
            write(path,state);torch.set_num_threads(4)
        if 'profile' not in state and not a.skip_profile:
            torch.set_num_threads(1)
            state['profile']=profile_strategy(MaskablePPO.load(a.combat,device='cpu'),strategy,a.output)
            write(path,state);torch.set_num_threads(4)
        for repeat in range(a.repeats):
            for arm,settings in ARMS.items():
                if arm not in a.arms:continue
                key=f'{arm}-{repeat}'
                if key in state['arms']:continue
                if time.monotonic()>deadline:raise TimeoutError('Experiment budget exhausted')
                folder=a.output/key;folder.mkdir(exist_ok=True)
                torch.manual_seed(1234+repeat);np.random.seed(1234+repeat)
                manager=mp.get_context('spawn').Manager()
                counter=manager.Value('i',50000+repeat*20000);lock=manager.Lock();env=None
                callback=AuditProgress(deadline)
                print(f'EXPERIMENT {key}: {settings}',flush=True)
                try:
                    env=SubprocVecEnv([partial(worker,str(a.strategy.resolve()),settings['quality'],counter,lock,str(folder/f'worker-{i}.log')) for i in range(a.n_envs)],start_method='spawn')
                    conservative=settings['conservative']
                    model=TeacherAnchoredPPO.load(a.combat,env=env,device=device,gamma=1.,
                        learning_rate=3e-5 if conservative else 3e-4,n_epochs=3 if conservative else 10,
                        target_kl=.01 if conservative else None,n_steps=128,batch_size=256)
                    if conservative:model.attach_teacher(MaskablePPO.load(a.combat,device=device).policy,.1)
                    model.learn(total_timesteps=a.steps,callback=callback,reset_num_timesteps=True)
                    model.save(folder/'combat.zip')
                finally:
                    if env is not None:env.close()
                    next_seed=counter.value;manager.shutdown()
                training=dict(settings=settings,steps=callback.steps,episodes=callback.episodes,
                    combat_boundaries=callback.boundaries,ppo_metrics=callback.metrics,
                    reward_components=callback.reward_components,training_seed_start=50000+repeat*20000,training_seed_end=next_seed)
                write(folder/'training.json',training)
                torch.set_num_threads(1)
                result=evaluate(MaskablePPO.load(folder/'combat.zip',device='cpu'),strategy,seeds,deadline)
                result['training']=training;state['arms'][key]=result
                write(path,state);torch.set_num_threads(4)
                print(f"RESULT {key}: floor={result['mean_floor']:.2f} wins={result['wins']} rounds/combat={result['mean_rounds_per_combat']:.2f}",flush=True)
        state['status']='complete';write(path,state)
    except BaseException as error:
        state['status']='failed';state['error']=traceback.format_exc();write(path,state);raise


if __name__=='__main__':main()
