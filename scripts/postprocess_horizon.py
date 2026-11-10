"""Audit completed runs while a sweep is running; resume missing analyses safely."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from map_learning.training import Trainer,seed_all
from audit_latent_rollouts import audit as rollouts
from audit_reward_ranking import audit as ranking
from evaluate_next_state_retrieval import evaluate as retrieval
from analyze_reward_stability import analyze as stability


def main():
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);args=p.parse_args()
    seed_all(0,1)
    while True:
        pending=0
        for root in args.roots:
            manifest=json.loads((root/'manifest.json').read_text())
            for variant in manifest['variants']:
                for seed in manifest['seeds']:
                    run=root/f"{variant['name']}-seed{seed}";out=run/'overnight-audit.json'
                    if out.exists():continue
                    pending+=1
                    if not (run/'metrics.json').exists():continue
                    trainer=Trainer.load(run/'checkpoint.pt');replay=dict(np.load(run/'replay.npz'))
                    result={'retrieval':retrieval(run),'stability':stability(run),
                        'rollout':rollouts(trainer,replay,len(replay['actions'])),
                        'ranking':ranking(trainer,replay)}
                    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                    print(out,flush=True)
        if not pending:break
        time.sleep(10)

if __name__=='__main__':main()
