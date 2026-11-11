"""Audit completed continuous-frontier runs and summarize all seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from map_learning.config import load_config
from map_learning.maze import Maze


def analyze(root):
    manifest=json.loads((root/'manifest.json').read_text()); records=[]
    for seed in manifest['seeds']:
        hashes=[]
        for arm in manifest['arms']:
            name=f"{arm['name']}-seed{seed}";path=root/name
            status=json.loads((root/f'{name}.status.json').read_text())
            assert status['returncode']==0,name
            r=json.loads((path/'result.json').read_text());cfg=json.loads((path/'config.json').read_text())
            rows=[json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()]
            frames=np.load(path/'frames.npz');replay=np.load(path/'replay.npz')
            count=manifest['blocks']*5000
            assert r['interactions']==count==len(replay['actions'])
            assert replay['actions'].shape==(count,2)
            assert np.max(np.linalg.norm(replay['actions'],axis=1))<=1.000001
            assert np.all(np.isfinite(replay['actions']))
            assert np.array_equal(replay['actions'][4::5],np.zeros((count//5,2)))
            assert r['quarter_lattice_fraction']<.05,name
            env=Maze(load_config(path/'config.json').maze)
            probes=np.linspace(0,count-1,256,dtype=int)
            physical=np.stack([env.transition(replay['positions'][i],replay['actions'][i]) for i in probes])
            assert np.array_equal(physical,replay['next_positions'][probes]),name
            assert np.allclose(replay['observations'][:,:2],2*replay['positions']/env.config.size-1)
            assert np.allclose(replay['next_observations'][:,:2],2*replay['next_positions']/env.config.size-1)
            endpoints=np.concatenate([env.start[None],replay['next_positions']])
            for k in (1,2,4,8):
                occupied=len(np.unique(np.floor(endpoints*k).astype(int),axis=0))
                assert np.isclose(occupied/(env.config.size*k)**2,r[f'coverage_k{k}']),name
            r['action_stream_sha256']=hashlib.sha256(replay['actions'].tobytes()).hexdigest()
            if arm['objective'] in {'dynamics','hybrid','adjacency'}:
                updates=sum((i*5000+255)//256 for i in range(1,manifest['blocks']+1))
                samples=sum(i*5000 for i in range(1,manifest['blocks']+1))
                updates+=r.get('extra_updates',0)
                samples+=r.get('extra_updates',0)*cfg['train']['batch_size']
                assert r['updates']==updates and r['samples_per_head']==samples
                assert r['trained_through']==count
            for k in (1,2,4,8):
                values=np.array([row[f'coverage_k{k}'] for row in rows])
                assert np.all(np.diff(values)>=0)
                r[f'auc_k{k}']=float(np.trapezoid(np.r_[0,values],np.r_[0,[row['interactions'] for row in rows]])/count)
            assert np.isclose(frames['occupied_k4'][-1].mean(),r['coverage_k4'])
            # Only compare initial parameters for equal observation dimensions.
            hashes.append((cfg['maze']['nuisance_dims'],r['initial_parameters_sha256']))
            records.append({'name':arm['name'],'run':str(path),**r})
        for dims in {x[0] for x in hashes}:
            assert len({h for d,h in hashes if d==dims})==1,(seed,dims,'initialization mismatch')
    summaries=[]
    for arm in manifest['arms']:
        group=[r for r in records if r['name']==arm['name']]
        stats={k:{'mean':float(np.mean([r[k] for r in group])),
                  'values':[r[k] for r in group]} for k in ['coverage_k1','coverage_k2','coverage_k4','coverage_k8','auc_k4','fill_radius_0.1','blocked_fraction']}
        for target in ['k1_100','k4_90','k4_95','k4_99']:
            times=[r['milestones'].get(target) for r in group]
            stats[target]={'values':times,'successes':sum(x is not None for x in times),
                           'mean_if_all_reach':float(np.mean(times)) if all(x is not None for x in times) else None}
        summaries.append({'name':arm['name'],**stats})
    return {'root':str(root),'manifest':manifest,'summary':summaries,'runs':records}


def main():
    p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reports=[analyze(root) for root in a.roots]
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(reports,indent=2)+'\n')
    for report in reports:
        print(report['root'])
        for row in report['summary']:
            print(row['name'],'rooms',row['coverage_k1']['values'],'quarter',row['coverage_k4']['values'],
                  'fine',row['coverage_k8']['values'],'95%',row['k4_95']['values'],'allrooms',row['k1_100']['values'])

if __name__=='__main__':main()
