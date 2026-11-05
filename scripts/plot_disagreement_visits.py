"""Final learned-model diagnostics against evaluation-only state-action visit counts."""
import argparse
import json
from pathlib import Path
import numpy as np
from map_learning.evaluation import plotting


def run_statistics(path):
    path=Path(path)
    with np.load(path/'probe.npz') as probe, np.load(path/'replay.npz') as replay:
        lookup={tuple(np.round(p,5)):i for i,p in enumerate(probe['points'])}
        counts=np.zeros((len(lookup),5),dtype=int)
        for p,a in zip(replay['positions'],replay['actions']):
            counts[lookup[tuple(np.round(p,5))],int(a)]+=1
        js=probe['disagreement'].copy()
        counts=counts.reshape(-1)
        bins=[counts==0,(counts>=1)&(counts<=5),(counts>=6)&(counts<=20),(counts>=21)&(counts<=100),counts>100]
        return [{'count':int(mask.sum()),'mean_js':float(js[mask].mean()) if mask.any() else None} for mask in bins]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--long',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    rows=[r for folder in args.long for r in json.loads((Path(folder)/'summary.json').read_text())]
    stats=[{'variant':r['variant'],'seed':r['seed'],'bins':run_statistics(r['run'])} for r in rows]
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    (output/'visitation-disagreement.json').write_text(json.dumps(stats,indent=2)+'\n')
    plt=plotting()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    labels=['0','1–5','6–20','21–100','>100']
    for name in sorted({r['variant'] for r in stats}):
        values=np.array([[b['mean_js'] if b['mean_js'] is not None else np.nan for b in r['bins']] for r in stats if r['variant']==name])
        axes[0].plot(labels,np.nanmean(values,axis=0),marker='o',label=name)
    axes[0].set(xlabel='Observed transitions per physical state-action',ylabel='Final distribution disagreement (nats)',title='Does disagreement decrease with familiarity?')
    axes[0].legend()
    groups={name:[r for r in rows if r['variant']==name] for name in sorted({r['variant'] for r in rows})}
    for i,(name,items) in enumerate(groups.items()):
        values=[100*r['metrics']['encoder_saturation_fraction'] for r in items
                if r['metrics']['encoder_saturation_fraction'] is not None]
        if values:
            axes[1].scatter([i]*len(values),values,s=35)
        else:
            axes[1].text(i,50,'unbounded',ha='center',va='center',rotation=90)
    axes[1].set(xticks=np.arange(len(groups)),xticklabels=list(groups),ylabel='Latent coordinates near tanh bound (%)',title='Encoder saturation across seeds',ylim=(0,100))
    axes[1].tick_params(axis='x',rotation=20)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'long-diagnostics.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
