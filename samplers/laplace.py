from samplers.sampler import Sampler
from bandit import Bandit
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

    def _gradient_hessian(self, theta, t):
        E = len(self.bandit.graph.edges)
        p = np.array([
            self.bandit.expected_reward(a, theta)
            for i, (a, r) in enumerate(self.history) if i < t - 1
        ])
        y = np.array([r for i, (a, r) in enumerate(self.history) if i < t - 1])
        gradient = (self.alpha - 1) / theta - self.beta
        for i, (path, _) in enumerate(self.history[:t - 1]):
            for edge in path:
                gradient[edge] += p[i] - y[i]
        hessian = np.zeros((E, E))
        for e in range(E):
            hessian[e, e] = -(self.alpha - 1) / theta[e] / theta[e]
        for i, (path, _) in enumerate(self.history[:t - 1]):
            for e in path:
                hessian[e, e] -= p[i] * (1 - p[i])
            for e in path:
                for f in path:
                    if e == f:
                        continue
                    hessian[e, f] = -p[i] * (1 - p[i])
        return gradient, hessian

    def _find_mode(self, t):
        assert len(self.history) >= t - 1
        E = len(self.bandit.graph.edges)
        theta = self.modes[max(0, t - 2)].copy()
        for _ in range(20):
            gradient, hessian = self._gradient_hessian(theta, t)
            target = theta - np.linalg.inv(hessian) @ gradient
            if np.min(target) <= 0:
                target = theta - np.linalg.inv(hessian) @ gradient * 0.1
            if np.max(np.abs(target - theta)) < 0.00001:
                break
            theta = target
        self.modes[t - 1] = theta
        return theta

    def sample(self, t):
        theta = self._find_mode(t)
        _, hessian = self._gradient_hessian(theta, t)
        cov = -np.linalg.inv(hessian)
        return np.random.multivariate_normal(theta, cov)
