from samplers.sampler import Sampler
from bandit import Bandit
import numpy as np
import scipy as sp
import util

class LangevinSampler(Sampler):
    B = 100
    epsilon = 0.01

    def __init__(self, bandit: Bandit, alpha, beta):
        super().__init__(bandit, alpha, beta)
        self.history = []
        E = len(bandit.graph.edges)
        self.modes = [np.array([2.0] * E)] * bandit.T

    def update(self, t, a, r):
        assert len(self.history) == t - 1
        self.history.append((a, r))

    def sample(self, t):
        theta = util.find_mode(
            self.bandit, self.history, t, self.alpha, self.beta)
        assert np.min(theta) > 0
        self.modes[t - 1] = theta

        _, hessian = util.gradient_hessian(
            theta, self.bandit, self.history, t, self.alpha, self.beta)
        A = -np.linalg.inv(hessian)
        A_sqrt = util.sqrtm(A)

        E = len(self.bandit.graph.edges)
        for b in range(self.B):
            gradient, _ = util.gradient_hessian(
                theta, self.bandit, self.history, t, self.alpha, self.beta)
            W = np.random.normal(0, 1, E)
            theta += (
                self.epsilon * (A @ gradient) +
                np.sqrt(2 * self.epsilon) * (A_sqrt @ W)
            )

        return theta
