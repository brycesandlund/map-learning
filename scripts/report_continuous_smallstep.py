"""Compare original half-step runs, matched extra fitting, and exact-prefix extensions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def trace(path):
    rows=[json.loads(x) for x in (path/'progress.jsonl').read_text().splitlines()]
    return np.array([x['interactions'] for x in rows]),np.array([x['coverage_k4'] for x in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path('runs');ablation=root/'continuous-ablation-20261111';follow=root/'continuous-smallstep-followup-20261111';long=root/'continuous-long-20261111'
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));audit=[]
    for ax,objective in zip(axes,['dynamics','adjacency']):
        for name,base,template,color in [('Original cadence → 200k',long,f'{objective}-seed{{seed}}','#386cb0'),
                ('+1000 early updates',follow,f'{objective}-early-seed{{seed}}','#ef8a25'),
                ('+1000 distributed updates',follow,f'{objective}-distributed-seed{{seed}}','#8458a3'),
                ('Coordinate-only diagnostic',follow,'oracle-smallstep-seed{seed}','#269378')]:
            curves=[]
            for seed in [30,31,32]:
                path=base/template.format(seed=seed);x,y=trace(path);curves.append(y*100)
            curves=np.asarray(curves)
            ax.plot(x/1000,curves.mean(0),color=color,label=name)
            ax.fill_between(x/1000,curves.min(0),curves.max(0),color=color,alpha=.12)
        ax.set(title=objective,xlabel='Executed actions (thousands)',ylabel='0.25-bin occupancy (%)',ylim=(0,103));ax.grid(alpha=.2)
        for seed in [30,31,32]:
            original=ablation/f'{objective}-smallstep-seed{seed}';extended=long/f'{objective}-seed{seed}'
            old=np.load(original/'replay.npz');new=np.load(extended/'replay.npz')
            for key in ['observations','next_observations','actions','positions','next_positions','episodes']:
                assert np.array_equal(old[key],new[key][:len(old[key])]),(seed,objective,key)
            c1=torch.load(original/'checkpoint.pt',weights_only=True);c2=torch.load(extended/'checkpoint-20.pt',weights_only=True)
            for group in ['model','dynamics']:
                assert all(torch.equal(c1[group][key],c2[group][key]) for key in c1[group]),(seed,objective,group)
            audit.append({'objective':objective,'seed':seed,'prefix_replay_exact':True,'checkpoint100k_model_and_dynamics_exact':True})
    axes[0].legend(fontsize=8);fig.suptitle('Half-sized continuous moves: mean and range over three seeds')
    fig.tight_layout();fig.savefig(str(a.out)+'.png',dpi=160);plt.close(fig)
    Path(str(a.out)+'-prefix-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Six exact-prefix continuation audits passed')

if __name__=='__main__':main()
