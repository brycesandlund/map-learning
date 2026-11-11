"""Immutable latent L2 experiments with matched mean/no-KL and mean/KL controls."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--lambdas',default='0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1');p.add_argument('--seeds',default='9,10,11');p.add_argument('--jobs',type=int,default=6);p.add_argument('--reuse',action='store_true');p.add_argument('--reproduce-zero',action='store_true');a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
 jobs=[]
 for kind,lam,beta in [('l2',float(x),0.) for x in a.lambdas.split(',')]+[('kl',0.,1e-5)]:
  for seed in map(int,a.seeds.split(',')):
   reused=a.reuse and seed in [9,10,11] and lam==0
   path=f'runs/kl-mechanisms-online-20261110/online-{"mean_kl" if beta else "mean_zero"}-seed{seed}' if reused else str(a.out/f'{kind}-{lam:g}-seed{seed}')
   jobs.append(dict(kind=kind,latent_l2=lam,beta=beta,seed=seed,path=path,reused=reused,role='experiment'))
 if a.reproduce_zero:jobs.insert(0,dict(kind='l2',latent_l2=0.,beta=0.,seed=9,path=str(a.out/'zero-reproduction-seed9'),reused=False,role='reproduction'))
 (a.out/'manifest.json').write_text(json.dumps({'jobs':jobs,'args':{**vars(a),'out':str(a.out)}},indent=2)+'\n')
 def run(j):
  if j['reused']:return 0
  cmd=[sys.executable,'scripts/persistent_frontier.py','--arm','interleaved100','--seed',str(j['seed']),'--out',j['path'],'--repulsion-weight','0.03','--bottleneck-beta',str(j['beta']),'--dynamics-input','mean','--latent-l2',str(j['latent_l2'])]
  with (a.out/(Path(j['path']).name+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  print(json.dumps({**j,'exit_code':r.returncode}),flush=True);return r.returncode
 with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,jobs))
 if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
