from samplers.sampler import Sampler
from bandits.bandit import Bandit
import numpy as np
import scipy as sp
import util

class LangevinSampler(Sampler):
    B = 100
    epsilon = 0.01

    def __init__(self, bandit: Bandit, alpha, beta, stochastic=None):
        super().__init__(bandit, alpha, beta)
        self.history = []
        self.stochastic = stochastic
        E = len(bandit.graph.edges)

    def update(self, t, a, r):
        assert len(self.history) == t - 1
        self.history.append((a, r))

    def sample(self, t):
        return util.langevin_sampling(
            self.bandit,
            self.history[:t - 1],
            self.alpha,
            self.beta,
            self.epsilon,
            self.stochastic,
            self.B)
