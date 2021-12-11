from samplers.sampler import Sampler
from bandits.bandit import Bandit
import util
import numpy as np

class LaplaceSampler(Sampler):
    def __init__(self, bandit: Bandit, alpha, beta):
        super().__init__(bandit, alpha, beta)
        self.history = []
        E = len(bandit.graph.edges)
        self.modes = [np.array([2.0] * E)] * bandit.T

    def update(self, t, a, r):
        assert len(self.history) == t - 1
        self.history.append((a, r))

    def objective(self, theta, t):
        ret = np.sum(np.log(theta) * (self.alpha - 1) - self.beta * theta)
        for i, (path, y) in enumerate(self.history[:t - 1]):
            cost = self.bandit.graph.path_cost(path, theta)
            if y == 1:
                ret -= np.log(1 + np.exp(cost - self.bandit.M))
            else:
                ret -= np.log(1 + np.exp(-cost + self.bandit.M))
        return ret

    def sample(self, t):
        theta_0 = self.modes[max(0, t - 2)].copy()
        theta = util.find_mode(
            self.bandit, self.history, t, self.alpha, self.beta, theta_0=theta_0)
        self.modes[t - 1] = theta

        _, hessian = util.gradient_hessian(
            theta, self.bandit, self.history, t, self.alpha, self.beta)
        cov = -np.linalg.inv(hessian)
        return np.random.multivariate_normal(theta, cov)
