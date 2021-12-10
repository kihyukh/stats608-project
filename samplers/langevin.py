from samplers.sampler import Sampler
from bandit import Bandit
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
        history = util.stochastic_sampling(self.history, t, None)
        theta = util.find_mode(
            self.bandit, history, self.alpha, self.beta)
        assert np.min(theta) > 0
        _, hessian = util.gradient_hessian(
            theta, self.bandit, history, self.alpha, self.beta)
        A = -np.linalg.inv(hessian)
        A_sqrt = util.sqrtm(A)

        E = len(self.bandit.graph.edges)
        for b in range(self.B):
            history = util.stochastic_sampling(
                self.history, t, self.stochastic)
            gradient, _ = util.gradient_hessian(
                theta, self.bandit, history, self.alpha, self.beta)
            W = np.random.normal(0, 1, E)
            theta += (
                self.epsilon * (A @ gradient) +
                np.sqrt(2 * self.epsilon) * (A_sqrt @ W)
            )

        return theta
