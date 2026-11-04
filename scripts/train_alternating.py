"""Teacher warm start, alternating full-run PPO, and fixed held-out evaluation."""
import argparse
import json
import math
import multiprocessing as mp
from functools import partial
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from sts2_env.gym_env.full_run_combat_env import FullRunCombatEnv
from sts2_env.gym_env.strategy_env import StrategyEnv
from train_teachers_overnight import snapshot
from sts2_env.gym_env.full_run_rewards import REWARD_CONFIG

ROOT=Path(__file__).resolve().parents[1]
VALIDATION_START=1_000_000


def write(path,data):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=2)); temp.replace(path)


class TrainingSeeds(gym.Wrapper):
    def __init__(self,env,state): super().__init__(env); self.state=state
    def reset(self,**kwargs):
        seed=self.state['next_training_seed']
        if seed>=VALIDATION_START: raise ValueError('Training seed allocation exhausted')
        self.state['next_training_seed']+=1
        return self.env.reset(seed=seed,options={'run_seed':seed})
    def action_masks(self): return self.env.action_masks()


class SharedTrainingSeeds(gym.Wrapper):
    def __init__(self,env,counter,lock):
        super().__init__(env);self.counter=counter;self.lock=lock
    def reset(self,**kwargs):
        with self.lock:
            seed=self.counter.value
            if seed>=VALIDATION_START:raise ValueError('Training seed allocation exhausted')
            self.counter.value=seed+1
        return self.env.reset(seed=seed,options={'run_seed':seed})
    def action_masks(self):return self.env.action_masks()


def make_worker(role,checkpoint,character,max_steps,counter,lock,threads,log_path=None):
    # Each worker owns its simulator and frozen policy. Avoid CPU thread pools
    # multiplying across processes, and never initialize CUDA in rollout workers.
    if log_path:
        import faulthandler
        sys.stderr=open(log_path,'a',buffering=1,encoding='utf-8')
        faulthandler.enable(file=sys.stderr,all_threads=True)
    torch.set_num_threads(threads)
    frozen=MaskablePPO.load(checkpoint,device='cpu')
    env=FullRunCombatEnv(frozen,character=character,max_steps=max_steps) if role=='combat' else StrategyEnv(frozen,character=character,max_steps=max_steps)
    return SharedTrainingSeeds(env,counter,lock)


class Progress(BaseCallback):
    def __init__(self,deadline,role):
        super().__init__(); self.deadline=deadline; self.role=role
        self.exposure={}; self.episodes=[]
        self.started=time.monotonic();self.last_print=self.started;self.steps=0
    def _on_step(self):
        self.steps+=len(self.locals['infos'])
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            if done: self.episodes.append({k:info.get(k) for k in ('win','floor','act','timeout')})
            if self.role=='combat':
                # This is the act of the action just executed, not of the next fight.
                act=info.get('combat_action_act')
                if act is not None: self.exposure[act]=self.exposure.get(act,0)+1
        if time.monotonic()-self.last_print>=15:
            self.report()
        return time.monotonic()<self.deadline
    def report(self):
        elapsed=time.monotonic()-self.started
        print(f'[{self.role}] phase_steps={self.steps} fps={self.steps/max(elapsed,.001):.1f} completed_runs={len(self.episodes)} combat_actions_by_act={self.exposure}',flush=True)
        self.last_print=time.monotonic()
    def _on_rollout_end(self):
        self.report()


def evaluate(combat,strategy,seeds,character,max_steps,deadline):
    env=FullRunCombatEnv(strategy,character=character,max_steps=max_steps)
    rows=[]
    try:
        for seed in seeds:
            obs,_=env.reset(seed=seed,options={'run_seed':seed})
            while True:
                if time.monotonic()>=deadline: raise TimeoutError('Time limit during evaluation')
                action,_=combat.predict(obs,action_masks=env.action_masks(),deterministic=True)
                obs,_,terminated,truncated,info=env.step(int(action))
                if terminated or truncated: break
            rows.append(dict(seed=seed,**info))
            print(f"[validation] seed={seed} win={info['win']} floor={info['floor']} act={info['act']} timeout={info['timeout']}",flush=True)
    finally: env.close()
    wins=sum(r['win'] for r in rows); n=len(rows); rate=wins/n
    z=1.96; d=1+z*z/n; center=(rate+z*z/(2*n))/d
    radius=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/d
    return {'runs':rows,'win_rate':rate,'win_rate_95pct_wilson':[center-radius,center+radius],
            'mean_floor':float(np.mean([r['floor'] for r in rows])),
            'reached_act2':sum(r['act']>=2 for r in rows),'reached_act3':sum(r['act']>=3 for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT.parent/'sts2-ml/runs/fireworks-full-generation-20261101/dataset')
    p.add_argument('--output',type=Path,default=ROOT/'output/alternating_rewards_v2')
    p.add_argument('--initial-combat',type=Path,help='Existing warm-start checkpoint for a new experiment')
    p.add_argument('--initial-strategy',type=Path,help='Existing warm-start checkpoint for a new experiment')
    p.add_argument('--teacher-epochs',type=int,default=20)
    p.add_argument('--cycles',type=int,default=3,help='Additional cycles this invocation')
    p.add_argument('--combat-steps',type=int,default=10000)
    p.add_argument('--strategy-steps',type=int,default=5000)
    p.add_argument('--rollout-steps',type=int,default=128,help='Rollout length PER worker')
    p.add_argument('--n-envs',type=int,default=min(8,max(1,(os.cpu_count() or 2)//2)))
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--torch-threads',type=int,default=4,help='Learner CPU threads')
    p.add_argument('--worker-threads',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--eval-runs',type=int,default=30)
    p.add_argument('--max-run-steps',type=int,default=10000)
    p.add_argument('--hours',type=float,default=10)
    p.add_argument('--character',default='Silent')
    a=p.parse_args()
    if min(a.teacher_epochs,a.cycles,a.combat_steps,a.strategy_steps,a.eval_runs,a.max_run_steps)<=0 or a.rollout_steps<2 or a.hours<=0:
        p.error('All limits must be positive; rollout-steps must be >=2')
    if min(a.n_envs,a.torch_threads,a.worker_threads,a.batch_size)<1:p.error('Resource limits must be positive')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA PyTorch is not available in this Python environment')
    device='cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]=str(a.worker_threads)
    torch.set_num_threads(a.torch_threads)
    if device=='cuda':torch.set_float32_matmul_precision('high')
    print(f'Resources: workers={a.n_envs} worker_threads={a.worker_threads} learner_threads={a.torch_threads} device={device} rollout={a.rollout_steps*a.n_envs} batch={a.batch_size}',flush=True)
    a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=True)
    lock=a.output/'pipeline.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    deadline=time.monotonic()+a.hours*3600
    path=a.output/'pipeline.json'
    state=json.loads(path.read_text()) if path.exists() else {'cycles':[],'next_training_seed':0}
    config={'character':a.character,'max_run_steps':a.max_run_steps,'validation_seeds':list(range(VALIDATION_START,VALIDATION_START+a.eval_runs)),
            'reward':REWARD_CONFIG}
    try:
        if state.get('configuration',config)!=config: raise ValueError('Character, run limit, reward version and validation seeds must match on resume; use a new output directory')
        state['configuration']=config;state['status']='running';write(path,state)
        state['resources']={'n_envs':a.n_envs,'device':device,'torch_threads':a.torch_threads,'worker_threads':a.worker_threads,'batch_size':a.batch_size,'rollout_steps_per_worker':a.rollout_steps}
        if 'teacher_signature' not in state and any(not state.get(role) and not getattr(a,'initial_'+role) for role in ('combat','strategy')):
            state['teacher_signature']=snapshot(a.data.resolve(),a.output/'teacher_snapshot')
            write(path,state)
        for role in ('combat','strategy'):
            if state.get(role):continue
            target=a.output/f'teacher_{role}'
            initial=getattr(a,'initial_'+role)
            if initial:
                shutil.copy2(initial.resolve(),target.with_suffix('.zip'))
                state[role]=str(target.with_suffix('.zip'))
                state.setdefault('initial_sources',{})[role]=str(initial.resolve())
                write(path,state)
                continue
            command=[sys.executable,'-u',str(ROOT/'scripts/pretrain_demonstrations.py'),'--data',str(a.output/'teacher_snapshot'),
                     '--role',role,'--output',str(target),'--epochs',str(a.teacher_epochs),'--observation-version','3']
            subprocess.run(command,cwd=ROOT,check=True,timeout=max(.01,deadline-time.monotonic()))
            state[role]=str(target.with_suffix('.zip'));write(path,state)
        if 'baseline' not in state:
            state['baseline']=evaluate(MaskablePPO.load(state['combat'],device='cpu'),MaskablePPO.load(state['strategy'],device='cpu'),
                config['validation_seeds'],a.character,a.max_run_steps,deadline)
            write(path,state)
        for _ in range(a.cycles):
            # Each completed phase is durable; interrupted cycles resume at the next phase.
            cycle=state.setdefault('active_cycle',{'index':len(state['cycles'])+1,'phases':{}})
            folder=a.output/f"cycle-{cycle['index']:04d}";folder.mkdir(exist_ok=True)
            for role in ('combat','strategy'):
                if role in cycle['phases']:
                    if role=='combat' and 'combat_validation' not in cycle:
                        frozen_checkpoint=cycle['phases'][role].get('frozen_checkpoint',state['strategy'])
                        cycle['combat_validation']=evaluate(MaskablePPO.load(cycle['phases'][role]['checkpoint'],device='cpu'),MaskablePPO.load(frozen_checkpoint,device='cpu'),
                            config['validation_seeds'],a.character,a.max_run_steps,deadline)
                        write(path,state)
                    continue
                if time.monotonic()>=deadline:raise TimeoutError('Training time limit reached')
                opposite='strategy' if role=='combat' else 'combat'
                frozen_checkpoint=state[opposite]
                manager=mp.get_context('spawn').Manager()
                counter=manager.Value('i',state['next_training_seed']);seed_lock=manager.Lock()
                env=None
                callback=Progress(deadline,role)
                print(f"[cycle {cycle['index']}] PPO {role}; frozen {opposite}={state[opposite]}",flush=True)
                try:
                    env=SubprocVecEnv([partial(make_worker,role,state[opposite],a.character,a.max_run_steps,counter,seed_lock,a.worker_threads,str(folder/f'{role}-worker-{i}.log')) for i in range(a.n_envs)],start_method='spawn')
                    model=MaskablePPO.load(state[role],env=env,device=device,gamma=1.0,n_steps=a.rollout_steps,batch_size=min(a.batch_size,a.rollout_steps*a.n_envs))
                    model.learn(total_timesteps=a.combat_steps if role=='combat' else a.strategy_steps,callback=callback,reset_num_timesteps=False)
                    target=folder/f'{role}.zip';model.save(target)
                finally:
                    if env is not None:env.close()
                    state['next_training_seed']=counter.value
                    manager.shutdown()
                state[role]=str(target)
                cycle['phases'][role]={'checkpoint':str(target),'timesteps':model.num_timesteps,'runs':callback.episodes,'combat_actions_by_act':callback.exposure,
                                      'gamma':model.gamma,
                                      'frozen_checkpoint':frozen_checkpoint,
                                      'phase_steps':callback.steps,'seconds':time.monotonic()-callback.started,
                                      'fps':callback.steps/max(time.monotonic()-callback.started,.001),
                                      'time_limited':time.monotonic()>=deadline}
                write(path,state)
                if role=='combat':
                    cycle['combat_validation']=evaluate(MaskablePPO.load(state['combat'],device='cpu'),MaskablePPO.load(state['strategy'],device='cpu'),
                        config['validation_seeds'],a.character,a.max_run_steps,deadline)
                    write(path,state)
            if 'validation' not in cycle:
                cycle['validation']=evaluate(MaskablePPO.load(state['combat'],device='cpu'),MaskablePPO.load(state['strategy'],device='cpu'),
                    config['validation_seeds'],a.character,a.max_run_steps,deadline)
            state['cycles'].append(cycle);del state['active_cycle'];write(path,state)
        state['status']='complete';write(path,state)
        print(f'Pipeline complete: {path}',flush=True)
    except (TimeoutError,subprocess.TimeoutExpired):
        state['status']='time_limit_reached';write(path,state);print('Time limit reached; rerun to continue saved phases.',flush=True)
    except BaseException as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        state['error']=f'{type(error).__name__}: {error}'
        state['traceback']=traceback.format_exc()
        (a.output/'last-error.log').write_text(state['traceback'])
        write(path,state);raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
