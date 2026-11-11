"""Do centroid distances encode wall detours? Fixed physical-separation probes."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def main():
    p=argparse.ArgumentParser();p.add_argument('audits',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=[];cache={}
    for file in sorted(a.audits.glob('*-at*.npz')):
        d=dict(np.load(file));pts=d['points'];adj=d['adjacency'];n=len(pts)
        key=pts.tobytes()+adj.tobytes()
        if key not in cache:
            s=np.repeat(np.arange(n),4);t=adj[:,:4].flatten();moving=s!=t
            geo=shortest_path(csr_matrix((np.ones(moving.sum()),(s[moving],t[moving])),shape=(n,n)),directed=False,unweighted=True)
            delta=pts[:,None]-pts[None];upper=np.triu(np.ones((n,n),bool),1)
            groups=[]
            for axis in [0,1]:
                same=upper&np.isclose(abs(delta[:,:,axis]),.5)&np.isclose(delta[:,:,1-axis],0)
                direct=np.argwhere(same&np.isclose(geo,2));detour=np.argwhere(same&(geo>=8))
                groups.append((direct,detour,geo[detour[:,0],detour[:,1]]))
            cache[key]=groups
        groups=cache[key];z=d['centroids'];row={'audit':str(file),'axes':[]}
        for axis,(direct,detour,g) in enumerate(groups):
            for seen_only in [False,True]:
                aa=direct;bb=detour;gg=g
                if seen_only:
                    aa=aa[d['seen'][aa].all(1)];mask=d['seen'][bb].all(1);bb=bb[mask];gg=gg[mask]
                if not len(aa) or not len(bb):continue
                da=np.linalg.norm(z[aa[:,0]]-z[aa[:,1]],axis=1);db=np.linalg.norm(z[bb[:,0]]-z[bb[:,1]],axis=1)
                row['axes'].append({'axis':axis,'seen_only':seen_only,'direct_pairs':len(aa),'detour_pairs':len(bb),
                    'median_latent_detour_over_direct':float(np.median(db)/max(np.median(da),1e-15)),
                    'median_true_detour_over_direct':float(np.median(gg)/2),
                    'median_direct_distance':float(np.median(da)),'median_detour_distance':float(np.median(db))})
        rows.append(row)
    a.out.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'audits':len(rows),'out':str(a.out)}))

if __name__=='__main__':main()
