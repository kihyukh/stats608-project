from samplers.sampler import Sampler
import numpy as np

class RandomSampler(Sampler):
    def update(self, t, a, r):
        pass

    def sample(self, t):
        return np.random.random(len(self.bandit.graph.edges))
