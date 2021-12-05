from bandit import Bandit
import numpy as np

class Sampler:
    def __init__(self, bandit: Bandit):
        self.bandit = bandit

    def update(self, t, a, r):
        pass

    def sample(self):
        return np.random.random(len(self.bandit.graph.edges))
