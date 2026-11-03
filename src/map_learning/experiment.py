from dataclasses import asdict
import json
import hashlib
import platform
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from .config import MazeConfig
from .maze import Maze
from .replay import Replay
from .training import Trainer, seed_all
from .planning import LatentPlanner
from .evaluation import coverage_plot, evaluate, split_replay


def prepare_run(output, config, extra=None):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    config.save(path/"config.json")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "git_status": status.stdout, "extra": extra}
    sources = Path(__file__).parent
    snapshot = path/"source"/"map_learning"
    snapshot.mkdir(parents=True)
    hashes = {}
    for source in sorted(sources.glob("*.py")):
        content = source.read_bytes()
        (snapshot/source.name).write_bytes(content)
        hashes[source.name] = hashlib.sha256(content).hexdigest()
    info["source_sha256"] = hashes
    (path/"metadata.json").write_text(json.dumps(info, indent=2)+"\n")
    return path


def log_metrics(file, metrics):
    file.write(json.dumps(metrics, allow_nan=False)+"\n")
    file.flush()


def collect(config, output):
    seed_all(config.seed, config.threads)
    path = prepare_run(output, config, {"mode": "collect", "behavior": "persistent_random"})
    env, replay = Maze(config.maze, config.seed), Replay()
    planner = LatentPlanner(None, config.planner, config.seed+5)
    observation = env.reset()
    episode = 0
    for step in range(config.collect_steps):
        action = planner.random_action()
        position = env.position.copy()
        nxt, done = env.step(action)
        replay.add(observation, action, nxt, episode, position, env.position)
        observation = nxt
        if done:
            observation = env.reset()
            planner.reset()
            episode += 1
    replay.save(path/"replay.npz", config.maze)
    coverage_plot(env, replay, path/"coverage.png")
    print(json.dumps({"replay": str(path/"replay.npz"), "transitions": len(replay), "episodes": episode+int(not done)}), flush=True)
    return path


def train(config, dataset, output, resume=None):
    seed_all(config.seed, config.threads)
    replay, maze = Replay.load(dataset)
    # Dataset provenance is authoritative: never silently reinterpret observation geometry.
    config.maze = MazeConfig(**maze)
    config.validate()
    learning, heldout = split_replay(replay, seed=config.seed)
    if resume:
        trainer = Trainer.load(resume, config.device)
        # Resuming must retain architecture, losses, optimizer settings and split seed.
        old, new = asdict(trainer.config), asdict(config)
        for allowed in ("updates", "log_every"):
            old["train"].pop(allowed)
            new["train"].pop(allowed)
        if old != new:
            raise ValueError("Resume config differs; only train.updates/log_every can change")
        trainer.config = config
    else:
        trainer = Trainer(config, learning.observations[0].shape)
    path = prepare_run(output, config, {"mode": "offline", "dataset": str(Path(dataset).resolve()),
                                       "training_transitions": len(learning), "heldout_transitions": len(heldout),
                                       "resume": resume})
    started = time.monotonic()
    with (path/"training.jsonl").open("w") as file:
        for _ in range(config.train.updates):
            metrics = trainer.update(learning)
            if trainer.updates % config.train.log_every == 0 or trainer.updates == 1:
                metrics["seconds"] = time.monotonic()-started
                log_metrics(file, metrics)
                print(json.dumps(metrics), flush=True)
    trainer.save(path/"checkpoint.pt")
    metrics = evaluate(trainer, learning, path, heldout)
    print(json.dumps({"output": str(path), "metrics": metrics}), flush=True)
    return path


def online(config, output):
    seed_all(config.seed, config.threads)
    path = prepare_run(output, config, {"mode": "online"})
    env, replay = Maze(config.maze, config.seed), Replay()
    trainer = Trainer(config, env.observation_shape)
    planner = LatentPlanner(trainer, config.planner, config.seed+5)
    obs = env.reset()
    origin = obs.copy()
    episode, scale = 0, None
    scale_rng = np.random.default_rng(config.seed+6000)
    started = time.monotonic()
    with (path/"training.jsonl").open("w") as file:
        for step in range(config.collect_steps):
            ready = step >= config.warmup_steps and trainer.updates > 0
            action = planner.act(obs, origin, scale) if ready else planner.random_action()
            position = env.position.copy()
            nxt, done = env.step(action)
            replay.add(obs, action, nxt, episode, position, env.position)
            obs = nxt
            if step+1 >= max(1, config.warmup_steps) and (step+1) % config.train_every == 0:
                for _ in range(config.updates_per_train):
                    metrics = trainer.update(replay)
                with torch.no_grad():
                    sample = replay.sample_states(256, scale_rng)
                    scale = trainer.model.encode(trainer.tensor(sample)).std(0).clamp_min(1e-3)
                if trainer.updates % config.train.log_every < config.updates_per_train:
                    metrics.update(environment_step=step+1, seconds=time.monotonic()-started)
                    log_metrics(file, metrics)
                    print(json.dumps(metrics), flush=True)
            if done:
                obs = env.reset()
                origin = obs.copy()
                planner.reset()
                episode += 1
    trainer.initialize(replay)
    trainer.save(path/"checkpoint.pt")
    replay.save(path/"replay.npz", config.maze)
    metrics = evaluate(trainer, replay, path)
    print(json.dumps({"output": str(path), "metrics": metrics}), flush=True)
    return path


def rollout(checkpoint, output, steps, overrides=(), goal=None, device="cpu"):
    """Use a frozen learned map for exploration or an explicitly supplied goal."""
    from .config import load_config
    trainer = Trainer.load(checkpoint, device)
    config = load_config(overrides=overrides)
    # Only behavior options are overrideable here; representation and environment
    # must match the checkpoint. Use online for learning during interaction.
    for entry in overrides:
        if not entry.startswith("planner."):
            raise ValueError("rollout --set accepts only planner.* overrides")
    planner_config = asdict(trainer.config.planner)
    for entry in overrides:
        key = entry.split("=", 1)[0].split(".")[1]
        planner_config[key] = getattr(config.planner, key)
    from .config import PlannerConfig
    trainer.config.planner = PlannerConfig(**planner_config)
    if goal is None and not overrides:
        trainer.config.planner.behavior = "distance"
    config = trainer.config
    config.collect_steps = steps
    seed_all(config.seed, config.threads)
    env, replay = Maze(config.maze, config.seed), Replay()
    if goal is not None and (np.any(np.asarray(goal) <= 0) or np.any(np.asarray(goal) >= config.maze.size)):
        raise ValueError("Goal must lie inside the maze")
    path = prepare_run(output, config, {"mode": "frozen_rollout", "checkpoint": str(checkpoint), "goal": goal})
    planner = LatentPlanner(trainer, config.planner, config.seed+5)
    obs = env.reset()
    origin = obs.copy()
    goal_observation = env.observe(goal) if goal is not None else None
    episode, successes, length = 0, [], 0
    for step in range(steps):
        action = planner.act(obs, origin, goal_observation=goal_observation)
        position = env.position.copy()
        nxt, done = env.step(action)
        replay.add(obs, action, nxt, episode, position, env.position)
        obs = nxt
        length += 1
        reached = goal is not None and np.linalg.norm(env.position-goal) <= config.maze.step_size/2
        if reached or done:
            successes.append({"episode": episode, "reached": bool(reached), "steps": length})
            obs = env.reset()
            origin = obs.copy()
            planner.reset()
            episode += 1
            length = 0
    replay.save(path/"replay.npz", config.maze)
    coverage_plot(env, replay, path/"coverage.png", f"Frozen map: {config.planner.behavior}")
    (path/"episodes.json").write_text(json.dumps(successes, indent=2)+"\n")
    print(json.dumps({"output": str(path), "completed_episodes": len(successes),
                      "successes": sum(row["reached"] for row in successes)}), flush=True)
    return path
