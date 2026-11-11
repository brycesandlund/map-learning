"""Sweep, fresh-seed confirmation, and conditional nuisance diagnostics."""
import argparse,json,os
from pathlib import Path
from collections import defaultdict
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/map-learning-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/map-learning-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from map_learning.maze import Maze
from map_learning.config import load_config


def read(root):
 rows=[]
 for j in json.load(open(root/'manifest.json'))['jobs']:
  if j['role']=='reproduction':continue
  p=Path(j['path']);r=json.load(open(p/'result.json'));pr=[json.loads(x) for x in (p/'progress.jsonl').read_text().splitlines()]
  rows.append({**j,'auc':float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000),'first_full':r['first_full'],'visited':r['visited'],'final':r['blocks'][-1]})
 return rows


def label(key):
 kind,lam=key
 return 'Mean + KL' if kind=='kl' else 'No regularizer' if lam==0 else f'L2 λ={lam:g}'


def main():
 p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True);p.add_argument('--confirm',type=Path,required=True);p.add_argument('--noise-roots',nargs='+',required=True);a=p.parse_args()
 screen=read(a.screen);confirm=read(a.confirm);noise=[r for root in a.noise_roots for r in json.load(open(Path(root)/'results.json'))];assets=Path('docs/assets');tables=[]
 for name,rows in [('Screen seeds 9/10/11',screen),('Confirmation seeds 15/16/17',confirm)]:
  g=defaultdict(list)
  for r in rows:g[r['kind'],r['latent_l2']].append(r)
  lines=[f'## {name}','', '| Arm | Mean coverage throughout 100k | Final coverage | Completion actions by seed | Noise / movement |', '|---|---:|---:|---|---:|']
  for k,rr in g.items():
   rr.sort(key=lambda r:r['seed']);lines.append(f'| {label(k)} | {100*np.mean([r["auc"] for r in rr]):.1f}% | {100*np.mean([r["visited"]/1784 for r in rr]):.1f}% | '+' / '.join(str(r['first_full']) if r['first_full'] else '>100k' for r in rr)+f' | {np.mean([r["final"]["noise_to_move_ratio"] for r in rr]):.3g} |')
  tables.append('\n'.join(lines))
 g=defaultdict(list)
 for r in screen:g[r['kind'],r['latent_l2']].append(r)
 keys=list(g);fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
 for i,k in enumerate(keys):
  rr=g[k]
  for ax,vals in zip(axes,[[100*r['auc'] for r in rr],[100*r['visited']/1784 for r in rr],[r['final']['bottleneck']['mean_latent_variance'] for r in rr]]):
   ax.bar(i,np.mean(vals),color='#0072b2' if k[0]=='kl' else '#e69f00',alpha=.7);ax.scatter(i+np.linspace(-.15,.15,len(vals)),vals,c='black',s=12);ax.set_xticks(range(len(keys)),[label(x) for x in keys],rotation=65,ha='right');ax.grid(axis='y',alpha=.2)
 axes[0].set_title('Mean coverage throughout 100k (%)');axes[1].set_title('Final coverage (%)');axes[2].set(title='Mean-latent variance',yscale='log');fig.suptitle('Latent L2 sweep: α=.03; deterministic dynamics inputs; three seeds')
 fig.savefig(assets/'l2-noise-sweep.png',dpi=150);plt.close(fig)
 cg=defaultdict(list)
 for r in confirm:cg[r['kind'],r['latent_l2']].append(r)
 colors={k:'#0072b2' if k[0]=='kl' else '#777777' if k[1]==0 else '#009e73' for k in cg}
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for k,rr in cg.items():
  for ax,r in zip(axes,sorted(rr,key=lambda r:r['seed'])):
   pr=[json.loads(x) for x in (Path(r['path'])/'progress.jsonl').read_text().splitlines()];ax.plot([x['interactions']/1000 for x in pr],[100*x['coverage'] for x in pr],label=label(k),color=colors[k]);ax.set(title=f'Seed {r["seed"]}',xlabel='Real actions (thousands)',ylabel='Coverage (%)',ylim=(0,101));ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('Fresh confirmation: selected L2 versus KL and no regularizer; mean inputs throughout');fig.savefig(assets/'l2-noise-confirm.png',dpi=150);plt.close(fig)
 kinds=['none','kl','l2'];names={'none':'No regularizer','kl':'Mean + KL','l2':'Selected L2'};modes=['raw','queue','both','zero','cap'];ml={'raw':'Stored','queue':'Redraw queue','both':'Redraw both','zero':'Nuisance removed','cap':'Radius clipped'}
 fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained',sharey=True)
 for i,prefix in enumerate([50000,75000]):
  lines=[f'## Common archive at {prefix//1000}k','', '| Model | Raw new states / 640 actions | Redraw queue | Redraw both | Nuisance removed | Radius clipped | Top-100 overlap after both redraw | Physical Jaccard | Top-100 norm percentile | Top-100 nuisance-residual percentile |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for j,kind in enumerate(kinds):
   rr=sorted([r for r in noise if r['kind']==kind and r['prefix']==prefix],key=lambda r:r['seed']);assert len(rr)==3
   yields=[]
   for x,mode in enumerate(modes):
    vals=[np.mean([v['new_states'] for v in r['live'] if v['mode']==mode]) for r in rr];yields.append(np.mean(vals));axes[i,j].bar(x,np.mean(vals),color=['#777777','#e69f00','#cc79a7','#009e73','#56b4e9'][x],alpha=.7);axes[i,j].scatter(x+np.linspace(-.15,.15,3),vals,c='black',s=14)
   axes[i,j].set(title=f'{names[kind]} · {prefix//1000}k',ylabel='New physical positions / 640 actions');axes[i,j].set_xticks(range(5),[ml[m] for m in modes],rotation=35,ha='right');axes[i,j].grid(axis='y',alpha=.2)
   overlap=np.mean([x['top100_entry_overlap'] for r in rr for x in r['ranking'] if x['mode']=='both']);jac=np.mean([x['top100_physical_jaccard'] for r in rr for x in r['ranking'] if x['mode']=='both'])
   lines.append(f'| {names[kind]} | '+' | '.join(f'{v:.1f}' for v in yields)+f' | {100*overlap:.1f}% | {jac:.3f} | {100*np.mean([r["raw"]["top100_norm_percentile"] for r in rr]):.1f} | {100*np.mean([r["raw"]["top100_noise_residual_percentile"] for r in rr]):.1f} |')
  tables.append('\n'.join(lines))
 fig.suptitle('Noise/search interventions: identical archives and fixed models\nDots = model/source seeds; average of 3 matched rollout-noise replicates. Nuisance removal is privileged evaluation.')
 fig.savefig(assets/'l2-noise-interventions.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for i,kind in enumerate(kinds):
  rr=[r for r in noise if r['kind']==kind and r['prefix']==50000]
  for j,mode in enumerate(['queue','anchors','both']):
   vals=[np.mean([x['top100_entry_overlap'] for x in r['ranking'] if x['mode']==mode]) for r in rr]
   axes[i].bar(j,100*np.mean(vals),color='#0072b2',alpha=.65);axes[i].scatter(j+np.linspace(-.15,.15,3),100*np.array(vals),color='black',s=15)
  axes[i].set(title=names[kind],xticks=[0,1,2],xticklabels=['Queue redraw','Anchor redraw','Both'],ylabel='Top-100 entries retained (%)',ylim=(0,100));axes[i].grid(axis='y',alpha=.2)
 fig.suptitle('Ranking stability at 50k: same positions, fresh independent nuisance inputs');fig.savefig(assets/'l2-noise-stability.png',dpi=150);plt.close(fig)
 # All confirmation arms and all seeds through100k.
 kk=list(cg);fig,axes=plt.subplots(3,3,figsize=(11,11),layout='constrained');frames={};artists={}
 for i,k in enumerate(kk):
  for j,r in enumerate(sorted(cg[k],key=lambda r:r['seed'])):
   d=Path(r['path']);ax=axes[i,j];Maze(load_config(d/'config.json').maze).draw(ax);frames[i,j]=np.load(d/'frames.npz');artists[i,j]=(ax.scatter([],[],s=3,c=colors[k]),ax.scatter([],[],s=20,c='red',zorder=6),ax.set_title('',fontsize=9));ax.set_xticks([]);ax.set_yticks([])
 def update(idx):
  for i,k in enumerate(kk):
   for j in range(3):
    f=frames[i,j];dots,here,title=artists[i,j];dots.set_offsets(f['points'][f['visited'][idx]]);here.set_offsets(f['position'][idx:idx+1]);title.set_text(f'{label(k)} · seed {15+j}\n{f["visited"][idx].sum()}/1784 observed')
  fig.suptitle(f'{frames[0,0]["interactions"][idx]:,} actions · fresh confirmation\nRed = simulator-restored expansion; all queues use deterministic latents')
 n=len(frames[0,0]['interactions']);ani=FuncAnimation(fig,update,frames=np.unique(np.linspace(0,n-1,100).astype(int)),interval=150);ani.save(assets/'l2-noise.gif',writer=PillowWriter(fps=7),dpi=85);update(n-1);fig.savefig(assets/'l2-noise-final.png',dpi=140);plt.close(fig)
 Path('docs/l2-noise-tables.md').write_text('# All L2 and noise/search results\n\n'+'\n\n'.join(tables)+'\n');(assets/'l2-noise-summary.json').write_text(json.dumps(dict(screen=screen,confirmation=confirm,noise=noise),indent=2)+'\n');print('\n\n'.join(tables))

if __name__=='__main__':main()
