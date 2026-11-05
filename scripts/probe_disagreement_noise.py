"""Evaluation-only sensitivity of immediate JS action rankings to MC sample count."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from map_learning.training import Trainer, seed_all
from map_learning.probabilistic import ensemble_js


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    seed_all(123,1)
    rows=[]
    for folder in args.runs:
        path=Path(folder)
        trainer=Trainer.load(path/'checkpoint.pt')
        with np.load(path/'probe.npz') as probe:
            index=np.random.default_rng(123).choice(len(probe['latents']),size=64,replace=False)
            z=trainer.tensor(probe['latents'][index]).repeat_interleave(5,dim=0)
        actions=torch.arange(5,device=trainer.device).repeat(64)
        generator=torch.Generator(device=trainer.device).manual_seed(123)
        with torch.no_grad():
            params=trainer.dynamics.distribution(z,actions)
            ref=ensemble_js(params,256,generator).reshape(64,5)
            draws=torch.stack([ensemble_js(params,8,generator).reshape(64,5) for _ in range(20)])
            selected=draws.argmax(-1)
            chosen_ref=ref.unsqueeze(0).expand(20,-1,-1).gather(-1,selected.unsqueeze(-1)).squeeze(-1)
            gap=ref.sort(-1).values[:,-1]-ref.sort(-1).values[:,-2]
            rows.append({'run':str(path.resolve()),'states':64,'repeats':20,'reference_samples_per_member':256,
                         'search_samples_per_member':8,'action_agreement_with_reference':float((selected==ref.argmax(-1)).float().mean()),
                         'mean_mc_std':float(draws.std(0).mean()),'mean_reference_best_second_gap':float(gap.mean()),
                         'mean_immediate_reward_regret':float((ref.max(-1).values-chosen_ref).mean())})
    Path(args.out).write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
