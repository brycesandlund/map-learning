"""Check whether uncertainty decreases on fixed replay, with/without encoder drift."""
import argparse
from pathlib import Path
import time
from map_learning.training import Trainer, seed_all
from map_learning.replay import Replay
from map_learning.experiment import prepare_run, log_metrics
from map_learning.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--freeze-encoder', action='store_true')
    args = parser.parse_args()
    trainer = Trainer.load(args.checkpoint)
    if trainer.config.train.loss != 'probabilistic' or args.updates < 1:
        raise ValueError('Positive update count and probabilistic checkpoint required')
    seed_all(trainer.config.seed, trainer.config.threads)
    replay, maze = Replay.load(args.data)
    from dataclasses import asdict
    if maze != asdict(trainer.config.maze):
        raise ValueError('Dataset/checkpoint environment mismatch')
    trainer.model.requires_grad_(not args.freeze_encoder)
    out = prepare_run(args.out, trainer.config, {'mode': 'fixed_replay_refit',
                       'checkpoint': str(Path(args.checkpoint).resolve()),
                       'dataset': str(Path(args.data).resolve()),
                       'additional_updates': args.updates, 'freeze_encoder': args.freeze_encoder})
    start = time.monotonic()
    with (out/'refit.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step:
                trainer.update(replay)
            if step%500 == 0 or step == args.updates:
                metric = evaluate(trainer, replay, out/f'update-{step}')
                metric.update(additional_updates=step, seconds=time.monotonic()-start)
                log_metrics(log, metric)
                print({k:metric[k] for k in ['additional_updates', 'replay_disagreement',
                       'replay_normalized_mse', 'oracle_all_normalized_mse', 'seconds']}, flush=True)
    trainer.save(out/'checkpoint.pt')


if __name__ == '__main__':
    main()
