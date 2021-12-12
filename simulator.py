from bandits.bandit import Bandit
from algorithm import Algorithm
from graph import Graph
from samplers.random import RandomSampler
from samplers.laplace import LaplaceSampler
from samplers.langevin import LangevinSampler
import json

class Simulator:
    t = 0
    regret = 0
    def __init__(self, bandit: Bandit, algorithm: Algorithm, logger=None):
        self.bandit = bandit
        self.algorithm = algorithm
        self.logger = logger

    def __iter__(self):
        self.t = 0
        return self

    def __len__(self):
        return self.bandit.T

    def __next__(self):
        self.t += 1
        if self.t > self.bandit.T:
            raise StopIteration
        a = self.algorithm.action(self.t)
        r = self.bandit.run(self.t, a)
        self.regret += self.bandit.best_reward(self.t) - self.bandit.expected_reward(self.t, a)
        self.algorithm.update(self.t, a, r)
        if self.logger:
            self.logger.write(
                '{}\t{}\t{}\t{}\n'.format(
                    self.t,
                    json.dumps(a),
                    r,
                    self.regret)
            )
        return (self.t, a, r, self.regret)
