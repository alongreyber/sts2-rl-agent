"""Summarize fixed-strategy pilot results without promoting a checkpoint."""
import argparse,json
from pathlib import Path
import numpy as np


def summarize(state):
    rng=np.random.default_rng(42)
    baseline=state['baseline'];summary={'baseline':{k:v for k,v in baseline.items() if k!='runs'},'arms':{},'profile':state.get('profile')}
    baseline_by_seed={r['seed']:r['floor'] for r in baseline['runs']}
    repeats=state['config']['repeats']
    for arm in state['config']['arms']:
        results=[state['arms'][f'{arm}-{i}'] for i in range(repeats)]
        differences=np.array([[r['floor']-baseline_by_seed[r['seed']] for r in sorted(v['runs'],key=lambda x:x['seed'])] for v in results])
        # Conditional on these training replicates: resample held-out run seeds,
        # keeping all policies' results for each seed together.
        per_seed=differences.mean(0)
        interval=np.quantile(rng.choice(per_seed,(10000,len(per_seed))).mean(1),[.025,.975]).tolist()
        summary['arms'][arm]={
            'mean_floor':float(np.mean([v['mean_floor'] for v in results])),
            'floor_by_training_seed':[v['mean_floor'] for v in results],
            'wins':sum(v['wins'] for v in results),'act2':sum(v['act2'] for v in results),
            'evaluations':sum(len(v['runs']) for v in results),
            'rounds_per_combat':float(np.mean([v['mean_rounds_per_combat'] for v in results])),
            'hp_loss_per_won_combat':float(np.mean([v['mean_hp_loss_per_won_combat'] for v in results])),
            'floor_difference_vs_teacher':float(per_seed.mean()),
            'conditional_paired_bootstrap_95pct':interval,
        }
    summary['caveat']='Two training replicates are a pilot. Repeated evaluations share the same development seeds. Intervals resample run seeds conditional on those training replicates, not all training uncertainty.'
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('output/combat_experiments/results.json'));a=p.parse_args()
    state=json.loads(a.results.read_text())
    if state['status']!='complete':raise ValueError('Experiments are not complete')
    summary=summarize(state)
    a.results.with_name('summary.json').write_text(json.dumps(summary,indent=2))
    print('arm | mean floor | floor by training seed | wins | rounds/combat | HP lost/won combat')
    for arm,v in summary['arms'].items():
        print(f"{arm} | {v['mean_floor']:.2f} | {v['floor_by_training_seed']} | {v['wins']}/{v['evaluations']} | {v['rounds_per_combat']:.2f} | {v['hp_loss_per_won_combat']:.2f}")


if __name__=='__main__':main()
