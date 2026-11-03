import argparse
import json
from pathlib import Path
from .config import load_config
from .experiment import collect, train, online, rollout
from .training import Trainer, seed_all
from .replay import Replay
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Learn distances and plan through a latent maze map")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["collect", "train", "online"]:
        p = sub.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("--out", required=True)
        if name == "train":
            p.add_argument("--data", required=True)
            p.add_argument("--resume")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("rollout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("sweep")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--codebook-sizes", type=int, nargs="+", default=[0, 32, 128])
    args = parser.parse_args()
    if args.command == "rollout":
        if args.steps < 1:
            parser.error("--steps must be positive")
        rollout(args.checkpoint, args.out, args.steps, args.set, args.goal, args.device)
        return
    if args.command == "sweep":
        import csv
        root = Path(args.out)
        root.mkdir(parents=True, exist_ok=False)
        rows = []
        for seed in args.seeds:
            for size in args.codebook_sizes:
                for loss in ("qrl", "time_myopic"):
                    cfg = load_config(args.config, [*args.set, f"seed={seed}", f"model.codebook_size={size}", f"train.loss={loss}"])
                    path = train(cfg, args.data, root/f"{loss}-k{size}-seed{seed}")
                    metrics = json.loads((path/"metrics.json").read_text())
                    rows.append({"loss": loss, "codebook_size": size, "seed": seed, **metrics})
                    columns = sorted(set().union(*(row.keys() for row in rows)))
                    with (root/"summary.csv").open("w") as file:
                        writer = csv.DictWriter(file, fieldnames=columns)
                        writer.writeheader()
                        writer.writerows(rows)
        return
    if args.command == "evaluate":
        trainer = Trainer.load(args.checkpoint, args.device)
        seed_all(trainer.config.seed, trainer.config.threads)
        replay, maze = Replay.load(args.data)
        from dataclasses import asdict
        if maze != asdict(trainer.config.maze):
            raise ValueError("Dataset environment differs from checkpoint environment")
        if Path(args.out).exists():
            raise FileExistsError(args.out)
        print(json.dumps(evaluate(trainer, replay, args.out)), flush=True)
        return
    config = load_config(args.config, args.set)
    if args.command == "collect":
        collect(config, args.out)
    elif args.command == "train":
        train(config, args.data, args.out, args.resume)
    else:
        online(config, args.out)


if __name__ == "__main__":
    main()
