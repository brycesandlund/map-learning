"""Scientific geometry comparison on an identical replay frontier."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from map_learning.maze import Maze
from map_learning.config import load_config


def main():
    p=argparse.ArgumentParser();p.add_argument('audits',type=Path,nargs='+');p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--bottom',choices=['queue','noise'],default='queue');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if len(a.audits)!=len(a.labels):p.error('Labels must match audits')
    n=len(a.audits);fig=plt.figure(figsize=(4.1*n,8.4));all_rows=[]
    for j,(path,label) in enumerate(zip(a.audits,a.labels)):
        if path.suffix in {'.json','.npz'}:path=path.with_suffix('')
        data=dict(np.load(Path(str(path)+'.npz')));r=json.loads(Path(str(path)+'.json').read_text());all_rows.append(r)
        ax=fig.add_subplot(2,n,j+1,projection='3d')
        pc,points,seen=data['pc'],data['points'],data['seen'];adj=data['adjacency']
        # Spatial coordinates are colors/evaluation edges only, never learned targets.
        color=points[:,0]+.45*points[:,1]
        edge=np.array([(i,k) for i in range(len(points)) for k in adj[i,:4] if i<k and seen[i] and seen[k]])
        if len(edge):ax.add_collection3d(Line3DCollection(pc[edge],colors='#6f7c83',linewidths=.3,alpha=.4))
        ax.scatter(*pc[~seen].T,c='#cccccc',s=1,alpha=.3)
        ax.scatter(*pc[seen].T,c=color[seen],cmap='turbo',s=5)
        ax.set_title(f"{label}\nPC1 {data['spectrum'][0]:.1%}; PC1–3 {sum(data['spectrum'][:3]):.1%}",fontsize=10)
        ax.set(xlabel='PC1 / RMS',ylabel='PC2 / RMS',zlabel='PC3 / RMS');ax.view_init(elev=23,azim=-57)
        # Equal axis scales: do not visually inflate a tiny second dimension.
        radius=max(np.max(np.abs(pc)),1e-3)*1.05
        ax.set_xlim(-radius,radius);ax.set_ylim(-radius,radius);ax.set_zlim(-radius,radius);ax.set_box_aspect((1,1,1))
        ax.tick_params(labelsize=6)
        bx=fig.add_subplot(2,n,n+j+1)
        if a.bottom=='noise':
            scale=max(r['edge_distances'][1],1e-15)
            for y,key,color in [(1,'noise_pair_distances','#d86644'),(0,'edge_distances','#2497a5')]:
                q=np.array(r[key])/scale
                bx.hlines(y,q[0],q[2],color=color,lw=3)
                bx.scatter(q[1],y,color=color,s=45,zorder=3)
            bx.set_xscale('log');bx.set_xlim(.01,1000);bx.set_ylim(-.6,1.6)
            bx.set_yticks([0,1],['Open step','Noise pair'])
            bx.axvline(1,color='gray',ls=':',lw=1);bx.grid(axis='x',alpha=.2)
            bx.set_xlabel('8D distance / median open-step distance (log)',fontsize=8)
            bx.set_title(f"Full8D noise/move mean ratio: {r['noise_to_move']:.2f}\nPoints: medians; bars: 10–90th percentiles",fontsize=9)
        else:
            Maze(load_config(Path(r['source'])/'config.json').maze).draw(bx)
            bx.scatter(*points[seen].T,c='#dde8ed',s=4)
            order=np.argsort(-data['scores'],kind='stable')[:100];qids=data['queue_ids'][order]
            good=data['productive'][qids]
            bx.scatter(*points[qids[~good]].T,c='#d56a44',s=18,alpha=.65,label='Unproductive')
            bx.scatter(*points[qids[good]].T,c='#1a9c64',s=20,alpha=.8,label='Productive')
            q=r['queue'];bx.set_title(f"Top100 queue entries: {q['top100_productive_fraction']:.0%} productive\nNoise/move {r['noise_to_move']:.2f}; median stretch ratio {r['local_anisotropy'][1]:.1f}",fontsize=9)
            if j==0:bx.legend(fontsize=7,loc='lower left')
    prefix=all_rows[0]['prefix']
    fig.suptitle(f'Common replay at {prefix:,} actions • PCA of position centroids (32 noise draws averaged)\nAveraging hides nuisance spread: consult full8D noise/move ratios below. Lower panels quantify noise sensitivity separately.',fontsize=12)
    fig.tight_layout(rect=[0,.01,1,.89],h_pad=3.);a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=150);plt.close(fig)

if __name__=='__main__':main()
