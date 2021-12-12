from samplers.sampler import Sampler
from bandits.bandit import Bandit
import numpy as np
import scipy as sp
import util

class LangevinGibbsSampler(Sampler):
    B = 100
    GB = 20
    epsilon = 0.005

    def __init__(self, bandit: Bandit, alpha, beta, stochastic=None):
        super().__init__(bandit, alpha, beta)
        self.history = []
        self.stochastic = stochastic
        E = len(bandit.graph.edges)

    def update(self, t, a, r):
        assert len(self.history) == t - 1
        self.history.append((a, r))

    def _sample_theta(self, t, tau):
        cutoff = min(tau - 1, t - 1)
        history1 = self.history[:cutoff]
        history2 = self.history[cutoff:t - 1]
        theta1 = util.langevin_sampling(
            self.bandit.bandit1,
            history1,
            self.alpha,
            self.beta,
            self.epsilon,
            self.stochastic,
            self.B)
        theta2 = util.langevin_sampling(
            self.bandit.bandit2,
            history2,
            self.alpha,
            self.beta,
            self.epsilon,
            self.stochastic,
            self.B)
        return theta1, theta2

    def _sample_tau(self, t, theta1, theta2):
        history = self.history[:t - 1]
        path_cost_1 = np.array([
            sum([theta1[e] for e in path]) for path, _ in history
        ])
        path_cost_2 = np.array([
            sum([theta2[e] for e in path]) for path, _ in history
        ])
        rewards = np.array([r for _, r in history])
        log_f1 = -(
            rewards * np.logaddexp(0, path_cost_1 - self.bandit.M) +
            (1 - rewards) * np.logaddexp(0, -path_cost_1 + self.bandit.M)
        )
        log_f2 = -(
            rewards * np.logaddexp(0, path_cost_2 - self.bandit.M) +
            (1 - rewards) * np.logaddexp(0, -path_cost_2 + self.bandit.M)
        )
        cumsum_log_f1 = np.cumsum(log_f1)
        rev_cumsum_log_f2 = np.cumsum(log_f2[::-1])[::-1]
        log_f_tau = [0.] * self.bandit.T
        if t > 1:
            for tau in range(self.bandit.T):
                if tau == 0:
                    log_f_tau[tau] = rev_cumsum_log_f2[tau]
                elif tau > 0 and tau < t - 1:
                    log_f_tau[tau] = cumsum_log_f1[tau - 1] + rev_cumsum_log_f2[tau]
                else:
                    log_f_tau[tau] = cumsum_log_f1[t - 2]
        log_f_tau = np.array(log_f_tau)
        log_f_tau -= np.max(log_f_tau)
        f_tau = np.exp(log_f_tau)
        p = f_tau / np.sum(f_tau)
        return np.random.choice(range(self.bandit.T), 1, p=p)[0] + 1

    def sample(self, t):
        tau = np.random.choice(range(self.bandit.T)) + 1
        theta1, theta2 = None, None
        for _ in range(self.GB):
            theta1, theta2 = self._sample_theta(t, tau)
            tau = self._sample_tau(t, theta1, theta2)

        return theta1 if t < tau else theta2
