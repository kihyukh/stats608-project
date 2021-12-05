from bandit import Bandit
from samplers.sampler import Sampler
import numpy as np

class Algorithm:
    def __init__(self, bandit: Bandit, sampler: Sampler):
        self.bandit = bandit
        self.sampler = sampler

    def update(self, t, a, r):
        self.sampler.update(t, a, r)

    def action(self):
        c = self.sampler.sample()
        return self.bandit.graph.shortest_path(self.bandit.source, self.bandit.destination, c)
