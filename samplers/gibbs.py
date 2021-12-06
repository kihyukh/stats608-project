from samplers.sampler import Sampler

class Gibbs(Sampler):
    history = []

    def update(self, t, a, r):
        history.append((a, r))

    def sample(self):
        return np.random.random(len(self.bandit.graph.edges))
