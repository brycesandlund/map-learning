"""Fixed-data overfitting diagnostic; no exploration and no training on oracle data."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from map_learning.training import Trainer, seed_all
from map_learning.models import MapModel
from map_learning.probabilistic import MixtureDynamicsEnsemble, mixture_mean, mixture_log_prob, mixture_variance, ensemble_js, repulsion_loss
from map_learning.replay import Replay
from map_learning.maze import Maze, reachable_graph
from map_learning.experiment import log_metrics


def unique_transitions(obs, actions, nxt):
    keys = np.concatenate([obs, actions[:, None]], axis=1)
    _, first, inverse, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    if not np.array_equal(nxt, nxt[first][inverse]):
        raise ValueError('Identical observation/action inputs have conflicting targets')
    return first, inverse, counts


def prediction_loss(parameters, target, objective):
    if objective == 'nll':
        return -mixture_log_prob(parameters, target).mean()/target.shape[-1]
    return (mixture_mean(parameters)-target).square().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--encoder', choices=['frozen', 'joint', 'coordinates'], default='frozen')
    parser.add_argument('--objective', choices=['nll', 'mse'], default='mse')
    parser.add_argument('--updates', type=int, default=20000)
    parser.add_argument('--subset', type=int, default=0, help='Fixed number of unique training edges; 0 uses whole replay')
    parser.add_argument('--uniform-edges', action='store_true', help='Remove visitation weighting by training on unique edges')
    parser.add_argument('--hidden', type=int, help='Override dynamics width only')
    parser.add_argument('--encoder-hidden', type=int)
    parser.add_argument('--encoder-depth', type=int)
    parser.add_argument('--dynamics-depth', type=int)
    parser.add_argument('--fresh-encoder', action='store_true')
    parser.add_argument('--latent-activation', choices=['tanh', 'identity'])
    parser.add_argument('--encoder-lr-multiplier', type=float, default=1.)
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--repulsion', type=float, default=10.)
    args = parser.parse_args()
    if args.updates < 1 or args.subset < 0 or args.lr <= 0:
        parser.error('Invalid budget/subset/lr')
    trainer = Trainer.load(args.checkpoint)
    cfg = trainer.config
    if cfg.train.loss != 'probabilistic' or cfg.maze.observation != 'coordinates' or cfg.maze.nuisance_dims:
        parser.error('Requires probabilistic coordinate-observation checkpoint')
    replay, maze = Replay.load(args.data)
    assert maze == asdict(cfg.maze)
    seed_all(args.seed, 1)
    encoder_cfg = replace(cfg.model, hidden_dim=args.encoder_hidden or cfg.model.hidden_dim,
                          hidden_layers=args.encoder_depth or cfg.model.hidden_layers,
                          latent_activation=args.latent_activation or cfg.model.latent_activation)
    if args.fresh_encoder or args.encoder_hidden or args.encoder_depth:
        if args.encoder != 'joint':parser.error('Fresh/enlarged encoders require joint mode')
        trainer.model = MapModel((2,), encoder_cfg, 'probabilistic', cfg.train.horizon)
    elif args.latent_activation:
        trainer.model.latent_activation = args.latent_activation
    trainer.model.requires_grad_(args.encoder == 'joint')
    model_cfg = replace(cfg.model, hidden_dim=args.hidden or cfg.model.hidden_dim,
                        hidden_layers=args.dynamics_depth or cfg.model.hidden_layers, mixture_components=1)
    dynamics = MixtureDynamicsEnsemble(model_cfg)
    if args.encoder_lr_multiplier <= 0 or args.eval_every < 0:parser.error('Invalid learning rate / evaluation interval')
    groups = [{'params': list(dynamics.parameters()), 'lr': args.lr}]
    if args.encoder == 'joint':groups.append({'params': list(trainer.model.parameters()), 'lr': args.lr*args.encoder_lr_multiplier})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1/30+(1-1/30)*.5*(1+np.cos(np.pi*step/args.updates)))
    rng = np.random.default_rng(args.seed)
    data = replay.arrays()
    first, inverse, counts = unique_transitions(data['observations'], data['actions'], data['next_observations'])
    indices = np.arange(len(replay))
    if args.subset:
        assert args.subset <= len(first)
        indices = np.sort(np.random.default_rng(1701).choice(first, size=args.subset, replace=False))
    elif args.uniform_edges:
        indices = first
    x = trainer.tensor(data['observations'][indices])
    a = torch.as_tensor(data['actions'][indices])
    y = trainer.tensor(data['next_observations'][indices])
    train_first, _, _ = unique_transitions(x.numpy(), a.numpy(), y.numpy())
    n, batch, members = len(x), cfg.train.batch_size, cfg.model.ensemble_size
    args.out.mkdir(parents=True, exist_ok=False)
    def encode(obs):
        if args.encoder == 'coordinates':
            assert cfg.model.latent_dim % 2 == 0
            return obs.repeat(1, cfg.model.latent_dim//2)
        return trainer.model.encode(obs)
    # Freeze the actual encoded targets once, removing any chance of target drift.
    with torch.no_grad():
        fixed_x, fixed_y = encode(x).detach(), encode(y).detach()
    env = Maze(cfg.maze)
    points, adjacency, lookup = reachable_graph(env)
    catalogue = trainer.tensor(np.stack([env.observe(p) for p in points]))
    physical_sources = np.array([lookup[tuple(np.round(p, 5))] for p in data['positions'][indices]])
    physical_targets = np.array([lookup[tuple(np.round(p, 5))] for p in data['next_positions'][indices]])
    seen = np.zeros((len(points), 5), dtype=bool)
    seen[physical_sources, a.numpy()] = True
    unseen_source, unseen_action = np.nonzero(~seen)
    # Disjoint unseen state-actions, for evaluation only; never read by training.
    pick = np.random.default_rng(1801).choice(len(unseen_source), min(1024, len(unseen_source)), replace=False)
    unseen_source, unseen_action = unseen_source[pick], unseen_action[pick]
    metadata = {**{k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'training_transitions': n, 'unique_training_edges': len(train_first), 'replay_unique_edges': len(first),
        'unique_replay_states': len(np.unique(np.concatenate([data['observations'], data['next_observations']]), axis=0)),
        'encoder_parameters': sum(p.numel() for p in trainer.model.parameters()) if args.encoder != 'coordinates' else 0,
        'dynamics_parameters': sum(p.numel() for p in dynamics.parameters()),
        'encoder_config': asdict(encoder_cfg), 'dynamics_config': asdict(model_cfg),
        'encoder_initialization': 'fresh' if args.fresh_encoder or args.encoder_hidden or args.encoder_depth else 'checkpoint',
        'dynamics_initialization': 'fresh matched initialization; optimizer reset',
        'lr_schedule': 'cosine to initial_lr/30',
        'source_checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest()}
    (args.out/'experiment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.out/'script.py').write_bytes(Path(__file__).read_bytes())
    snapshot = args.out/'source'; snapshot.mkdir()
    import map_learning
    for path in Path(map_learning.__file__).parent.glob('*.py'):
        (snapshot/path.name).write_bytes(path.read_bytes())
    np.save(args.out/'training_indices.npy', indices)
    started = time.monotonic()

    @torch.no_grad()
    def evaluate(step):
        zcat = encode(catalogue)
        var = float(zcat.var(0, unbiased=False).mean())
        singular = torch.linalg.svdvals(zcat-zcat.mean(0))
        mass = singular.square()/singular.square().sum().clamp_min(1e-12)
        row = {'update':step, 'seconds':time.monotonic()-started, 'lr':optimizer.param_groups[0]['lr'],
               'latent_variance':var, 'effective_rank':float((-(mass*mass.clamp_min(1e-12).log()).sum()).exp()),
               'saturation':float((zcat.abs()>.99).float().mean()) if args.encoder != 'coordinates' and encoder_cfg.latent_activation == 'tanh' else None,
               'encoder_drift_mse':float((encode(x)-fixed_x).square().mean())}
        for label, sources, actions, targets in [('train',physical_sources,a.numpy(),physical_targets),
                   ('unseen',unseen_source,unseen_action,adjacency[unseen_source,unseen_action])]:
            if not len(sources):continue
            mse, exact, physical, variances, nll, js, individual_exact = [],[],[],[],[],[],[]
            member_correct=[]
            generator = torch.Generator().manual_seed(1901)
            for offset in range(0,len(sources),256):
                sl=slice(offset,offset+256)
                z, truth = zcat[sources[sl]], zcat[targets[sl]]
                params=dynamics.distribution(z,torch.as_tensor(actions[sl]))
                means=mixture_mean(params)
                prediction=means.mean(0)
                # Full catalogue distinguishes physical states even if encoder collapses them.
                nearest=torch.cdist(prediction,zcat,compute_mode='donot_use_mm_for_euclid_dist').argmin(-1).numpy()
                for_member=torch.cdist(means.reshape(-1,means.shape[-1]),zcat,compute_mode='donot_use_mm_for_euclid_dist').argmin(-1).reshape(members,-1).numpy()
                mse.extend((prediction-truth).square().mean(-1).tolist())
                exact.extend((nearest==targets[sl]).tolist())
                correct=for_member==targets[sl][None]
                member_correct.append(correct)
                individual_exact.extend(correct.mean(0).tolist())
                physical.extend(np.linalg.norm(points[nearest]-points[targets[sl]],axis=-1).tolist())
                variances.extend(mixture_variance(params).mean((0,2)).tolist())
                nll.extend((-mixture_log_prob(params,truth).mean(0)/truth.shape[-1]).tolist())
                if args.objective=='nll':js.extend(ensemble_js(params,16,generator).tolist())
            correctness=np.concatenate(member_correct,axis=1)
            member_unique=correctness[:, train_first] if label=='train' else correctness
            metric={'per_member_unique_accuracy':member_unique.mean(1).tolist(),
                    'all_members_correct_fraction':float(member_unique.all(0).mean()),
                    'mse':float(np.mean(mse)), 'max_mse':float(np.max(mse)),
                    'normalized_mse':float(np.mean(mse))/max(var,1e-12),
                    'exact_retrieval':float(np.mean(exact)), 'individual_exact_retrieval':float(np.mean(individual_exact)),
                    'physical_error':float(np.mean(physical)), 'p95_physical_error':float(np.quantile(physical,.95)),
                    'unique_edge_exact_retrieval':float(np.mean(np.asarray(exact)[train_first])) if label=='train' else float(np.mean(exact)),
                    'nll_per_dim':float(np.mean(nll)) if args.objective=='nll' else None,
                    'predictive_variance':float(np.mean(variances)) if args.objective=='nll' else None,
                    'js':float(np.mean(js)) if js else None}
            row[label]=metric
        return row

    orders=None; offset=n
    milestones={0,100,1000,5000,10000,20000,50000,100000,200000,args.updates}
    with (args.out/'learning.jsonl').open('w') as log:
        row=evaluate(0);log_metrics(log,row);print(json.dumps(row),flush=True)
        for step in range(1,args.updates+1):
            if offset>=n:
                orders=[rng.permutation(n) for _ in range(members)];offset=0
            optimizer.zero_grad()
            losses=[]
            for m in range(members):
                idx=orders[m][offset:offset+batch]
                if args.encoder=='joint':z,target=encode(torch.cat([x[idx],y[idx]])).chunk(2)
                else:z,target=fixed_x[idx],fixed_y[idx]
                losses.append(prediction_loss(dynamics.parameters_head(m,z,a[idx]),target,args.objective))
            loss=torch.stack(losses).mean()
            if args.encoder=='joint':
                pairs=rng.integers(2*n,size=2*batch)
                observations=torch.cat([x,y])
                left,right=encode(observations[pairs]).chunk(2)
                loss=loss+args.repulsion*repulsion_loss(left,right,1.)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(),10.)
            if args.encoder=='joint':torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),10.)
            optimizer.step();scheduler.step();offset+=batch
            if step in milestones or (args.eval_every and step % args.eval_every == 0):
                row=evaluate(step);log_metrics(log,row);print(json.dumps(row),flush=True)
    torch.save({'encoder':trainer.model.state_dict(),'dynamics':dynamics.state_dict(),
                'model_config':asdict(model_cfg),'experiment':metadata},args.out/'weights.pt')
    (args.out/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')


if __name__=='__main__':main()
