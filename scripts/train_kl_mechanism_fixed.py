"""Train identical initial networks on a common chronological replay stream."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from map_learning.training import seed_all
from map_learning.information_bottleneck import BottleneckTrainer
from map_learning.config import load_config
from map_learning.replay import Replay
from map_learning.persistent_frontier import epoch_slices
from map_learning.maze import Maze,reachable_graph
from noisy_frontier import diagnostics
from audit_moment_prior import full_moments
from kl_mechanisms import ARMS,spec,data_source,digest_model,queue_audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--family',choices=['winner','zero'],required=True);p.add_argument('--source-seed',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--limit',type=int,default=100000);p.add_argument('--skip-queue',action='store_true');p.add_argument('--latent-l2',type=float,default=0.);a=p.parse_args()
    assert a.limit%5000==0
    a.out.mkdir(parents=True,exist_ok=False);source=data_source(a.family,a.source_seed)
    schedule=json.loads(a.schedule.read_text());cfg=load_config('configs/noisy_frontier.json',[f'seed={a.seed}','train.repulsion_weight=0.03'])
    seed_all(a.seed,1);t=BottleneckTrainer(cfg,(10,),**spec(a.arm,schedule),latent_l2=a.latent_l2)
    initial_hash=digest_model(t);t.save(a.out/'checkpoint-0.pt')
    (a.out/'config.json').write_text(json.dumps(asdict(cfg),indent=2)+'\n')
    (a.out/'runner.py').write_text(Path(__file__).read_text())
    data=np.load(source/'replay.npz');r=Replay();rng=np.random.default_rng(a.seed+61000)
    batchhash=hashlib.sha256();sample_count=0;rows=[];audits=[];start=time.monotonic()
    points,_,_=reachable_graph(Maze(cfg.maze))
    with (a.out/'training.jsonl').open('x') as log:
        for block in range(a.limit//5000):
            old=block*5000;order=rng.permutation(old)
            for j in range(10):
                end=old+(j+1)*500
                # Prefix exposure is identical across arms. Physical coordinates
                # are retained only for diagnostics and never sampled by loss.
                for key in ['observations','next_observations','actions','episodes','positions','next_positions']:
                    setattr(r,key,data[key][:end])
                r._arrays={key:getattr(r,key) for key in ['observations','next_observations','actions','episodes','positions','next_positions']}
                batches=epoch_slices(old,old+j*500,end,j,10,rng,order,old+5000,cfg.train.batch_size)
                for batch in batches:
                    batchhash.update(batch.astype('int64').tobytes())
                    metrics=t.update_probabilistic(r,head_indices=[batch]*3);sample_count+=len(batch)
                log.write(json.dumps({'interactions':end,'updates':t.updates,'loss_metrics':metrics})+'\n')
            row={'interactions':end,'updates':t.updates,'samples_per_head':sample_count,
                 **diagnostics(t,r,points),'bottleneck':t.bottleneck_diagnostics(r)}
            rows.append(row);t.save(a.out/f'checkpoint-{block+1}.pt')
            (a.out/'blocks.json').write_text(json.dumps(rows,indent=2)+'\n')
            if end in [25000,35000,50000,100000] and not a.skip_queue:
                q=queue_audit(t,data,end,a.source_seed,live=end==35000)
                q['moments']=full_moments(t,data,end);audits.append(q)
                (a.out/'queue.json').write_text(json.dumps(audits,indent=2)+'\n')
    t.save(a.out/'checkpoint.pt')
    result={'arm':a.arm,'family':a.family,'seed':a.seed,'source_seed':a.source_seed,'source':str(source),
            'initial_parameters_sha256':initial_hash,'batch_indices_sha256':batchhash.hexdigest(),
            'source_replay_sha256':hashlib.sha256((source/'replay.npz').read_bytes()).hexdigest(),
            'interactions':a.limit,'updates':t.updates,'samples_per_head':sample_count,'prior_updates':t.prior_updates,
            'spec':t.spec,'blocks':rows,'queue':audits,'seconds':time.monotonic()-start}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['arm','family','seed','updates','seconds']}),flush=True)

if __name__=='__main__':main()
