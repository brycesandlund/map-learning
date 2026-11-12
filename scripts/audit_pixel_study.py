"""Audit physical transitions, stored pixels, ancestry and nonreactive controls."""
import argparse,json
from pathlib import Path
import numpy as np
from map_learning.config import MazeConfig
from map_learning.pixel_exploration import PixelMaze
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
results=[]
for root in a.roots:
 for path in sorted(root.glob('*/result.json')):
    spec=json.loads((path.parent/'spec.json').read_text());r=json.loads(path.read_text())
    d=dict(np.load(path.parent/'trajectory.npz')); images=np.load(path.parent/'images.npy',mmap_mode='r');n=spec['steps']
    assert len(d['positions'])==len(images)==n+1
    assert np.all((d['parents'][1:]>=0)&(d['parents'][1:]<np.arange(1,n+1)))
    e=PixelMaze(MazeConfig(size=spec['size'],seed=spec['layout'],step_size=spec['step_size'],image_size=spec['image_size']),spec['seed']+71000,spec['family'],spec['noise'],spec.get('rgb',0),spec.get('marker_size',.85),spec.get('noise_mode','full'))
    e.reset()
    # Observation RNG advances once per actual action and never on restores.
    for i in range(1,min(n,25)+1):
        e.position=d['positions'][d['parents'][i]].copy();im,_=e.step(d['actions'][i])
        assert np.array_equal(im,images[i])
    for i in np.random.default_rng(0).choice(np.arange(1,n+1),min(n,256),replace=False):
        assert np.array_equal(e.transition(d['positions'][d['parents'][i]],d['actions'][i]),d['positions'][i])
    for k in [1,2,4,8]:
        bins=np.floor(d['positions']*k).astype(int)
        measured=len(np.unique(bins,axis=0))/(spec['size']*k)**2
        assert abs(measured-r[f'coverage_k{k}'])<1e-12
    if spec['policy']=='walk' or spec.get('warm_steps',0):
        wrng=np.random.default_rng(spec['seed']+93000);rrng=np.random.default_rng(spec['seed']+94000)
        heading=None;current=0;events=[]
        end=n if spec['policy']=='walk' else spec['warm_steps']
        for i in range(1,end+1):
            if i>1 and rrng.random()<spec['reset_probability']:
                current=int(rrng.integers(i));events.append((i-1,current))
            retain,angle,length=wrng.random(3)
            if heading is None or retain>=spec['persistence']:heading=angle
            control=(e.ACTIONS[min(3,int(heading*4))] if spec['controls']=='cardinal' else np.array([np.cos(heading*2*np.pi),np.sin(heading*2*np.pi)],np.float32))*np.float32(.5+.5*length)
            assert np.array_equal(control,d['actions'][i])
            assert current==d['parents'][i];current=i
        assert np.array_equal(np.array(events).reshape(-1,2),d['resets'])
    results.append(dict(run=str(path.parent),interactions=n,policy=spec['policy']))
a.out.write_text(json.dumps(dict(runs=len(results),interactions=sum(r['interactions'] for r in results),results=results),indent=2))
print('Audited',len(results),'runs')
