from bandit import Bandit
from algorithm import Algorithm
from graph import Graph
from samplers.random import RandomSampler
from samplers.laplace import LaplaceSampler

class Simulator:
    t = 0
    regret = 0
    def __init__(self, bandit: Bandit, algorithm: Algorithm):
        self.bandit = bandit
        self.algorithm = algorithm

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
        r = self.bandit.run(a)
        self.regret += self.bandit.best_reward() - self.bandit.expected_reward(a)
        self.algorithm.update(self.t, a, r)
        return (self.t, a, r, self.regret)

if __name__ == '__main__':
    g = Graph(4, [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ], [1, 2, 3, 4])
    bandit = Bandit(g, 4, 0, 3, 1000)
    sampler = LaplaceSampler(bandit, 2, 2)
    algorithm = Algorithm(bandit, sampler)
    sim = Simulator(bandit, algorithm)
    for t, a, r, regret in sim:
        print(a, regret / t)
