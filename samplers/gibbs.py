from samplers.sampler import Sampler

class Gibbs(Sampler):
    def update(self, t, a, r):
        pass

    def sample(self):
        return np.random.random(len(self.bandit.graph.edges))
