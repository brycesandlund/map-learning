import numpy as np
import torch


class LatentPlanner:
    """Receding-horizon beam search: learned transitions only, no policy training.

    Each ensemble member propagates its own trajectory. Disagreement is measured
    before optional projection onto codebook nodes. Return only the first action;
    the next decision starts again from the next real observation.
    """
    def __init__(self, trainer, config, seed=0):
        self.trainer = trainer
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.previous_action = None

    def reset(self):
        self.previous_action = None

    def random_action(self):
        if self.previous_action is None or self.rng.random() > self.config.action_repeat_probability:
            self.previous_action = int(self.rng.integers(5))
        return self.previous_action

    @torch.no_grad()
    def act(self, observation, start_observation, latent_scale=None, goal_observation=None):
        c, t = self.config, self.trainer
        if c.behavior == "random" and goal_observation is None:
            return self.random_action()
        if goal_observation is None and self.rng.random() < c.epsilon_random:
            return self.random_action()
        z = t.model.encode(t.tensor(np.asarray(observation)[None]))
        reference = start_observation if c.distance_reference == "episode_start" else observation
        anchor = t.model.encode(t.tensor(np.asarray(reference)[None]))
        goal = None if goal_observation is None else t.model.encode(t.tensor(np.asarray(goal_observation)[None]))
        scale = torch.ones_like(z) if latent_scale is None else latent_scale.clamp_min(1e-3)
        members = len(t.dynamics.heads)
        states = z.unsqueeze(0).expand(members, 1, -1)
        first_actions = torch.zeros(1, dtype=torch.long, device=t.device)
        uncertainty_sum = torch.zeros(1, device=t.device)
        for depth in range(c.horizon):
            count = states.shape[1]
            actions = torch.arange(5, device=t.device).repeat(count)
            predicted = torch.stack([t.dynamics.predict_head(i, states[i].repeat_interleave(5, dim=0), actions)
                                     for i in range(members)])
            disagreement = (predicted/scale).var(dim=0, unbiased=False).mean(-1)
            total_uncertainty = uncertainty_sum.repeat_interleave(5) + c.uncertainty_discount**depth * disagreement
            if t.model.bottleneck:
                predicted = t.model.bottleneck.project(predicted)
            # Average scores over member predictions, not distance of the average
            # latent (which could fall between valid codebook nodes).
            if goal is not None:
                scores = -t.model.distance(predicted, goal).mean(0)
            else:
                distance = t.model.distance(anchor, predicted).mean(0)
                if c.behavior == "distance":
                    scores = distance
                elif c.behavior == "disagreement":
                    scores = total_uncertainty
                else:
                    scores = distance + c.disagreement_weight * total_uncertainty
            parents_first = actions if depth == 0 else first_actions.repeat_interleave(5)
            best = torch.topk(scores, min(c.beam_width, len(scores))).indices
            states = predicted[:, best]
            first_actions = parents_first[best]
            uncertainty_sum = total_uncertainty[best]
        self.previous_action = int(first_actions[0])
        return self.previous_action
