"""Unlearned continuous random control; no map, bins, or novelty labels."""
import numpy as np


class RandomWalk:
    def __init__(self, seed, persistence=0., stay_probability=.2, turn_on_block=True):
        if not 0 <= persistence < 1 or not 0 <= stay_probability < 1:
            raise ValueError('Invalid probability')
        self.rng = np.random.default_rng(seed)
        self.persistence = persistence
        self.stay_probability = stay_probability
        self.turn_on_block = turn_on_block
        self.heading = None

    def reset(self):
        self.heading = None

    def action(self):
        stay, retain, angle, length = self.rng.random(4)
        if self.heading is None or retain >= self.persistence:
            self.heading = 2 * np.pi * angle
        if stay < self.stay_probability:
            return np.zeros(2, dtype=np.float32)
        return np.asarray([np.cos(self.heading), np.sin(self.heading)], dtype=np.float32) * np.float32(.5 + .5 * length)

    def observe(self, obs, successor, action):
        # The first two observable inputs are normalized xy. No latent/state ID.
        if self.turn_on_block and np.any(action) and np.array_equal(obs[:2], successor[:2]):
            self.reset()
