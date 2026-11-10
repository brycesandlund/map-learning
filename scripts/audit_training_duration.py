"""Compare 10k/30k models on identical early replay paths, not different datasets."""
import argparse
import json
from pathlib import Path
import numpy as np
from map_learning.training import Trainer,seed_all
from audit_latent_rollouts import audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path);args=p.parse_args()
    seed_all(0,1)
    for run in args.runs:
        trainer=Trainer.load(run/'checkpoint.pt');seed=trainer.config.seed
        name=run.name.split('-seed')[0].removesuffix('_30k')
        if name=='random_epoch':old=Path('runs/ensemble-matched-20261109')/f'epoch_independent-seed{seed}'
        else:old=Path('runs')/('horizon-pilot-20261109' if seed==0 else 'horizon-confirm-20261109')/f'{name}-seed{seed}'
        replay=dict(np.load(run/'replay.npz'));early=dict(np.load(old/'replay.npz'))
        for k in ['actions','observations','next_observations','positions','next_positions']:
            np.testing.assert_array_equal(early[k],replay[k][:len(early[k])])
        result={'reference_run':str(old),'description':'Both models evaluated on identical first-10k replay paths and visited starts.',
                'model_10k':json.loads((old/'overnight-audit.json').read_text())['rollout'],
                'model_30k':audit(trainer,replay,len(early['actions']))}
        out=run/'early-replay-audit.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(out,flush=True)

if __name__=='__main__':main()
