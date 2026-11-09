"""Recover final evaluation only after validating a fully completed online run.

Never resumes or repeats training; preserves checkpoint, config and raw replay.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from map_learning.config import load_config
from map_learning.evaluation import evaluate
from map_learning.replay import Replay
from map_learning.training import Trainer, seed_all


def expected_updates(cfg):
    first = ((max(1, cfg.warmup_steps)+cfg.train_every-1)//cfg.train_every)*cfg.train_every
    events = list(range(first, cfg.collect_steps+1, cfg.train_every))
    if cfg.training_schedule != 'replay' and (not events or events[-1] != cfg.collect_steps):
        events.append(cfg.collect_steps)
    updates, previous = 0, 0
    for step in events:
        if cfg.training_schedule != 'replay':
            size = step if cfg.training_schedule == 'epoch' else step-previous
            updates += (size+cfg.train.batch_size-1)//cfg.train.batch_size
        if cfg.training_schedule != 'epoch':
            updates += cfg.updates_per_train
        previous = step
    return updates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--reason',required=True)
    args=parser.parse_args()
    path=Path(args.run)
    cfg=load_config(path/'config.json')
    metadata=json.loads((path/'metadata.json').read_text())
    if metadata['extra']['mode']!='online':
        raise ValueError('Recovery supports completed online runs only')
    trainer=Trainer.load(path/'checkpoint.pt')
    replay,maze=Replay.load(path/'replay.npz')
    if asdict(trainer.config)!=asdict(cfg) or maze!=asdict(cfg.maze):
        raise ValueError('Config, checkpoint and replay do not match')
    updates=expected_updates(cfg)
    progress=json.loads((path/'exploration.jsonl').read_text().splitlines()[-1])
    if len(replay)!=cfg.collect_steps or trainer.updates!=updates or progress['environment_step']!=len(replay):
        raise ValueError('Training/interaction budget was not completed; refusing evaluation recovery')
    protected=['config.json','checkpoint.pt','replay.npz']
    hashes={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in protected}
    seed_all(cfg.seed,cfg.threads)
    metrics=evaluate(trainer,replay,path)
    assert hashes=={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in protected}
    record={'recovered_at':datetime.now(timezone.utc).isoformat(),'reason':args.reason,
            'environment_steps':len(replay),'updates':trainer.updates,'unchanged_artifact_sha256':hashes}
    (path/'evaluation_recovery.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'run':str(path),'recovered':True,'coverage_lattice':metrics['coverage_lattice']}))


if __name__=='__main__':main()
