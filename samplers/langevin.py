from samplers.sampler import Sampler
from bandits.bandit import Bandit
import numpy as np
import scipy as sp
import util

class LangevinSampler(Sampler):
    B = 100
    epsilon = 0.005

    def __init__(self, bandit: Bandit, alpha, beta, stochastic=None, window=None):
        super().__init__(bandit, alpha, beta)
        self.history = []
        self.stochastic = stochastic
        self.window = window

    def update(self, t, a, r):
        assert len(self.history) == t - 1
        self.history.append((a, r))

    def sample(self, t):
        if self.window and t - 1 > self.window:
            history = self.history[t - self.window - 1:t - 1]
        else:
            history = self.history[:t - 1]
        return util.langevin_sampling(
            self.bandit,
            history,
            self.alpha,
            self.beta,
            self.epsilon,
            self.stochastic,
            self.B)
