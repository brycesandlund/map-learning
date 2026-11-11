"""Later common-archive frozen yields, all arms and seeds."""
import argparse,json
from pathlib import Path
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/map-learning-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/map-learning-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kl_mechanisms import ARMS,LABELS,COLORS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=json.loads((a.root/'results.json').read_text());arms=list(ARMS)
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained');tables=[];summary=[]
    for i,prefix in enumerate([50000,75000]):
        lines=[f'## Shared β=0 archive at{prefix:,} transitions','', '| Arm | New states /640 actions, seeds9 /10 /11 | Mean new states | Top100 productive (%) |', '|---|---|---:|---:|']
        for j,arm in enumerate(arms):
            rr=sorted([r for r in rows if r['arm']==arm and r['prefix']==prefix],key=lambda r:r['seed'])
            assert len(rr)==3
            yields=np.array([np.mean([v['rows'][-1]['new_states'] for v in r['queue']['live']]) for r in rr]);density=np.array([100*r['queue']['stats']['top100_productive_fraction'] for r in rr])
            for ax,vals in zip(axes[i],[yields,density]):
                ax.bar(j,vals.mean(),color=COLORS[arm],alpha=.7);ax.scatter(j+np.linspace(-.15,.15,3),vals,c='black',s=17,zorder=5)
                ax.set_xticks(range(len(arms)),[LABELS[x] for x in arms],rotation=45,ha='right');ax.grid(axis='y',alpha=.2)
            lines.append(f'| {LABELS[arm]} | '+' / '.join(f'{v:.1f}' for v in yields)+f' | {yields.mean():.1f} | {density.mean():.2f} |')
            summary.append({'arm':arm,'prefix':prefix,'yields':yields.tolist(),'density':density.tolist()})
        axes[i,0].set(title=f'{prefix//1000}k: frozen continuation',ylabel='New positions in 640 real actions')
        axes[i,1].set(title=f'{prefix//1000}k: productive queue entries',ylabel='Top100 productive entries (%)')
        tables.append('\n'.join(lines))
    fig.suptitle('Later shared-archive follow-up: no additional model training\nDots = model/source seed pairs; yields average 3 matched observation-noise replicates')
    fig.savefig('docs/assets/kl-mechanisms-late.png',dpi=150);plt.close(fig)
    Path('docs/kl-mechanisms-late-tables.md').write_text('# Later frozen-archive results\n\n'+'\n\n'.join(tables)+'\n')
    Path('docs/assets/kl-mechanisms-late-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('\n\n'.join(tables))

if __name__=='__main__':main()
