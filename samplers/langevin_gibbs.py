from samplers.sampler import Sampler
from bandits.bandit import Bandit
import numpy as np
import scipy as sp
import util

class LangevinGibbsSampler(Sampler):
    B = 100
    GB = 100
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

    def _sample_theta1(self, t):
        return util.langevin_sampling(
            self.bandit.bandit1,
            self.history[:t - 1],
            self.alpha,
            self.beta,
            self.epsilon,
            self.stochastic,
            self.B)

    def _posterior_tau(self, t, theta1, theta2):
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
        log_f_tau = [0.] * (t - 200)
        if t - 200 > 1:
            for tau in range(t - 200):
                if tau < t - 1 - 200:
                    log_f_tau[tau] = cumsum_log_f1[tau + 200 - 1] + rev_cumsum_log_f2[tau + 200]
                else:
                    log_f_tau[tau] = cumsum_log_f1[t - 2]
        log_f_tau = np.array(log_f_tau)
        log_f_tau -= np.max(log_f_tau)
        f_tau = np.exp(log_f_tau)
        p = f_tau / np.sum(f_tau)
        return p

    def sample(self, t):
        if t <= 200:
            return self._sample_theta1(t), []
        tau_p = [0.] * (t - 200)
        tau = max(201, int((t - 1 + 200) / 2))
        theta1, theta2 = None, None
        gibbs_sample_count = 0
        for b in range(self.GB):
            theta1, theta2 = self._sample_theta(t, tau + 200)
            p = self._posterior_tau(t, theta1, theta2)
            tau = np.random.choice(range(t - 200), 1, p=p)[0] + 1
            if b >= 70:
                gibbs_sample_count += 1
                tau_p += p

        if gibbs_sample_count:
            tau_p = tau_p / gibbs_sample_count
        theta1, theta2 = self._sample_theta(t, tau)
        return (theta1, tau_p.tolist()) if t < tau else (theta2, tau_p.tolist())
