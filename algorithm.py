from bandit import Bandit
import numpy as np

class Algorithm:
    def __init__(self, bandit: Bandit, sampler):
        self.bandit = bandit
        self.sampler = sampler

    def update(self, t, a, r):
        # sampler specific logic
        pass

    def action(self):
        # sampler specific logic
        c = np.random.random(len(self.bandit.graph.edges))
        return self.bandit.graph.shortest_path(self.bandit.source, self.bandit.destination, c)
