"""Door discovery and coverage; all physical coordinates are evaluation-only."""
import argparse,json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--prefix',type=Path,required=True);a=p.parse_args()
rows=[];groups=collections.defaultdict(list)
for root in a.roots:
 for result in sorted(root.glob('*/result.json')):
  path=result.parent;s=json.loads((path/'spec.json').read_text());r=json.loads(result.read_text());d=dict(np.load(path/'trajectory.npz'))
  method=(f"Learned {s['dim']}D" if s['policy']=='learned' else 'Coordinate oracle' if s['policy']=='coordinate_oracle' else f"Walk p{s['persistence']:g}"+(' + resets' if s['reset_probability'] else ''))
  if s['policy']=='learned' and s['noise']==0:method+=' clean'
  first=[];crossings=[]
  src=d['positions'][d['parents'][1:]];dst=d['positions'][1:]
  for gate in range(1,4):
   x=s['size']*gate/4; center=s['size']*(1.73/12 if gate%2 else 1-1.73/12)
   ids=np.flatnonzero(d['positions'][:,0]>x);first.append(int(ids[0]) if len(ids) else None)
   cross=((src[:,0]<x)&(dst[:,0]>x))|((src[:,0]>x)&(dst[:,0]<x))
   y=src[cross,1]+(dst[cross,1]-src[cross,1])*(x-src[cross,0])/(dst[cross,0]-src[cross,0])
   assert np.all(abs(y-center)<=s['gate_width']/2+1e-5),(path,gate)
   crossings.append(int(cross.sum()))
  row=dict(observations_by_chamber=[int(((d['positions'][:,0]>=i*s['size']/4)&(d['positions'][:,0]<(i+1)*s['size']/4)).sum()) for i in range(4)],source_actions_by_chamber=[int(((src[:,0]>=i*s['size']/4)&(src[:,0]<(i+1)*s['size']/4)).sum()) for i in range(4)],path=str(path),method=method,width=s['gate_width'],seed=s['seed'],noise=s['noise'],first=first,crossings=crossings,**r);rows.append(row);groups[(s['gate_width'],method)].append(row)
summary=[];lines=['| Door width | Method | n | First gate | Deepest chamber | First deep entry by seed | Coverage | AUC |','|---|---|---:|---:|---:|---|---:|---:|']
for (w,m),rs in sorted(groups.items()):
 entry=dict(width=w,method=m,n=len(rs),entered=[sum(r['first'][i] is not None for r in rs) for i in range(3)],first_deep=[r['first'][2] for r in rs],seeds=[r['seed'] for r in rs],final_mean=float(np.mean([r['coverage_k4'] for r in rs])),auc_mean=float(np.mean([r['auc'] for r in rs])))
 summary.append(entry);times=', '.join(f"{r['seed']}: {r['first'][2] if r['first'][2] is not None else 'never'}" for r in rs)
 lines.append(f"| {w:g} | {m} | {len(rs)} | {entry['entered'][0]}/{len(rs)} | {entry['entered'][2]}/{len(rs)} | {times} | {100*entry['final_mean']:.1f}% | {100*entry['auc_mean']:.1f}% |")
prefixes=[]
for x in [r for r in rows if r['method'].startswith('Learned')]:
 for y in [r for r in rows if r['width']==x['width'] and r['seed']==x['seed'] and r['method'] in ['Walk p0.8','Coordinate oracle']]:
  tx=dict(np.load(Path(x['path'])/'trajectory.npz'));ty=dict(np.load(Path(y['path'])/'trajectory.npz'))
  for key in ['positions','parents','actions']:assert np.array_equal(tx[key][:5001],ty[key][:5001]),(x['path'],y['path'],key)
  ix=np.load(Path(x['path'])/'images.npy',mmap_mode='r');iy=np.load(Path(y['path'])/'images.npy',mmap_mode='r');same_images=np.array_equal(ix[:5001],iy[:5001])
  if x['noise']==y['noise']:assert same_images
  prefixes.append(dict(pair=[x['path'],y['path']],identical_images=bool(same_images)))
a.prefix.with_suffix('.json').write_text(json.dumps(dict(rows=rows,summary=summary,prefix_checks=prefixes),indent=2));a.prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n')
widths=sorted(set(r['width'] for r in rows),reverse=True)
fig,axes=plt.subplots(2,len(widths),figsize=(5*len(widths),7),squeeze=False,sharex=True)
colors={'Learned 2D clean':'#df7b19','Learned 2D':'#c33621','Learned 8D':'#b156c0','Coordinate oracle':'#222222','Walk p0':'#999999','Walk p0.8':'#2466ad','Walk p0.95':'#20a0a0','Walk p0.99':'#d3a02c','Walk p0.8 + resets':'#43a076'}
for j,w in enumerate(widths):
 for (width,m),rs in groups.items():
  if width!=w:continue
  traces=[]
  for r in rs:
   progress=[json.loads(l) for l in (Path(r['path'])/'progress.jsonl').read_text().splitlines()]
   x=np.array([0]+[v['interactions'] for v in progress]);traces.append(np.array([1/2304]+[v['coverage_k4'] for v in progress])*100)
  traces=np.array(traces);ax=axes[0,j];ax.plot(x/1000,traces.mean(0),label=m,color=colors[m]);ax.fill_between(x/1000,traces.min(0),traces.max(0),color=colors[m],alpha=.08)
  reached=np.array([[t is not None and t<=step for step in x] for t in [r['first'][2] for r in rs]])
  axes[1,j].plot(x/1000,reached.mean(0)*100,color=colors[m])
 axes[0,j].set_title(f'Door width {w:g} units');axes[1,j].set_xlabel('Actual actions (thousands)')
 for ax in axes[:,j]:ax.set_ylim(0,102);ax.grid(alpha=.2)
axes[0,0].set_ylabel('Coverage of 0.25-unit bins (%)');axes[1,0].set_ylabel('Runs reaching deepest chamber (%)')
legend={}
for ax in axes[0]:h,l=ax.get_legend_handles_labels();legend.update(zip(l,h))
fig.legend(list(legend.values()),list(legend),loc='lower center',ncol=4,fontsize=9);fig.tight_layout(rect=(0,.10,1,1));fig.savefig(a.prefix.with_suffix('.png'),dpi=140)
print('\n'.join(lines));print('Matched prefixes:',len(prefixes))
