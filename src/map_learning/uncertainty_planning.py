"""Temporary stochastic search trees; no graph merging, actor, or oracle geometry."""
import math
import numpy as np
import torch
from .planning import LatentPlanner
from .probabilistic import ensemble_js, ensemble_sample, mixture_mean


class SearchNode:
    def __init__(self, state, actions):
        self.state = state
        self.counts = np.zeros(actions, dtype=np.int64)
        self.returns = np.zeros(actions)
        self.children = [[] for _ in range(actions)]
        self.parameters = None
        self.rewards = None


class UncertaintyPlanner(LatentPlanner):
    """MCTS with UCT action nodes and progressive widening at chance nodes.

    Chance outcomes are draws from the uniform ensemble predictive distribution.
    Existing outcomes are reused uniformly, not selected for high reward. Models
    and rewards stay fixed within a search; all nodes are discarded on the next
    real observation. An alternative batched particle random-shooting planner
    uses the same distribution and reward, but fixed open-loop action sequences.
    """
    def __init__(self, trainer, config, seed=0):
        super().__init__(trainer, config, seed)
        self.generator = torch.Generator(device=trainer.device).manual_seed(seed+9000)
        self.last_metrics = {}
        self.mean_mode = trainer.config.train.prediction_objective == "mse"
        self.point_mode = getattr(trainer.config.train, "loss", None) == "dynamics"
        self.reward_variance = 1.0
        self.uct_scale = 1.0

    def successor(self, params):
        if self.point_mode:
            return params.mean(0)
        return mixture_mean(params).mean(0) if self.mean_mode else ensemble_sample(params, self.generator)

    def distribution_reward(self, state, actions):
        if self.point_mode:
            predictions = self.trainer.dynamics(state, actions)
            return predictions, predictions.var(0, unbiased=False).mean(-1)/self.reward_variance
        params = self.trainer.dynamics.distribution(state, actions)
        reward = (mixture_mean(params).var(0, unbiased=False).mean(-1)/self.reward_variance
                  if self.mean_mode else ensemble_js(params, self.config.disagreement_samples, self.generator))
        return params, reward

    def prepare(self, node):
        if node.parameters is None:
            n = self.trainer.dynamics.actions
            node.parameters, reward = self.distribution_reward(
                node.state.expand(n, -1), torch.arange(n, device=self.trainer.device))
            node.rewards = reward.cpu().numpy()

    def simulate(self, node, depth):
        if depth == self.config.horizon:
            return 0.
        self.prepare(node)
        unused = np.flatnonzero(node.counts == 0)
        if len(unused):
            action = int(self.rng.choice(unused))
        else:
            average = node.returns/node.counts
            bonus = self.config.uct_exploration*self.uct_scale*np.sqrt(np.log(node.counts.sum()+1)/node.counts)
            scores = average+bonus
            action = int(self.rng.choice(np.flatnonzero(scores == scores.max())))
        value = float(node.rewards[action])
        if depth+1 < self.config.horizon:
            outcomes = node.children[action]
            limit = max(1, int(math.ceil(self.config.widening_coefficient*
                                         (node.counts[action]+1)**self.config.widening_exponent)))
            if len(outcomes) < (1 if self.mean_mode else limit):
                params = (node.parameters[:, action:action+1] if self.point_mode else
                          tuple(p[:, action:action+1] for p in node.parameters))
                nxt = self.successor(params)
                child = SearchNode(nxt, self.trainer.dynamics.actions)
                outcomes.append(child)
                self.nodes_created += 1
            else:
                child = outcomes[int(self.rng.integers(len(outcomes)))]
            value += self.config.uncertainty_discount*self.simulate(child, depth+1)
        node.counts[action] += 1
        node.returns[action] += value
        return value

    def mcts(self, z):
        root = SearchNode(z, self.trainer.dynamics.actions)
        self.nodes_created = 1
        self.uct_scale = 1.0
        if self.mean_mode:
            self.prepare(root)
            # Keep UCT exploration commensurate with the small MSE reward.
            self.uct_scale = max(float(root.rewards.max()), 1e-12)*sum(
                self.config.uncertainty_discount**i for i in range(self.config.horizon))
        for _ in range(self.config.simulations):
            self.simulate(root, 0)
        choices = np.flatnonzero(root.counts == root.counts.max())
        action = int(self.rng.choice(choices))
        self.last_metrics = {"root_disagreement": float(root.rewards[action]),
                             "search_return": float(root.returns[action]/root.counts[action]),
                             "search_nodes": self.nodes_created,
                             "root_visits": root.counts.tolist(),
                             "root_rewards": root.rewards.tolist(),
                             "root_q": (root.returns/np.maximum(root.counts, 1)).tolist()}
        return action

    def shooting(self, z):
        c, device = self.config, self.trainer.device
        sequences = torch.as_tensor(self.rng.integers(self.trainer.dynamics.actions,
                                    size=(c.candidates, c.horizon)), device=device)
        states = z.expand(c.candidates*c.particles, -1)
        returns = torch.zeros(c.candidates*c.particles, device=device)
        root_rewards = None
        for depth in range(c.horizon):
            actions = sequences[:, depth].repeat_interleave(c.particles)
            params, reward = self.distribution_reward(states, actions)
            if depth == 0:
                root_rewards = reward.reshape(c.candidates, c.particles).mean(-1)
            returns += c.uncertainty_discount**depth*reward
            if depth+1 < c.horizon:
                states = self.successor(params)
        scores = returns.reshape(c.candidates, c.particles).mean(-1)
        best = int(scores.argmax())
        self.last_metrics = {"root_disagreement": float(root_rewards[best]),
                             "search_return": float(scores[best])}
        return int(sequences[best, 0])

    @torch.no_grad()
    def act(self, observation, start_observation=None, latent_scale=None, goal_observation=None):
        if goal_observation is not None:
            raise ValueError("Probabilistic exploration has no goal-matching objective yet")
        self.last_metrics = {}
        if self.config.behavior == "random" or self.rng.random() < self.config.epsilon_random:
            return self.random_action()
        if self.mean_mode:
            self.reward_variance = max(float(latent_scale), 1e-8) if latent_scale is not None else 1.0
        z = self.trainer.model.encode(self.trainer.tensor(np.asarray(observation)[None]))
        action = self.mcts(z) if self.config.search == "mcts" else self.shooting(z)
        self.previous_action = action
        return action


def make_planner(trainer, config, seed):
    if trainer.config.train.loss in {"probabilistic", "dynamics"}:
        return UncertaintyPlanner(trainer, config, seed)
    return LatentPlanner(trainer, config, seed)
