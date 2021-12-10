from bandit import Bandit
from algorithm import Algorithm
from graph import Graph
from demo import demo_graph1, demo_graph2
from samplers.random import RandomSampler
from samplers.laplace import LaplaceSampler
from samplers.langevin import LangevinSampler

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
    g = demo_graph2()
    bandit = Bandit(
        graph=g, M=2, source=0, destination=11, T=1000)
    sampler = LangevinSampler(bandit, 1.5, 1.5)
    algorithm = Algorithm(bandit, sampler)
    sim = Simulator(bandit, algorithm)
    for t, a, r, regret in sim:
        print(t, a, regret / t)
