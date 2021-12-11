import numpy as np
from graph import Graph
from bandits.bandit import Bandit

class SimpleBandit(Bandit):
    _best_action = None
    _best_reward = None

    def run(self, t, path):
        assert self.graph.edges[path[0]][0] == self.source and self.graph.edges[path[-1]][1] == self.destination
        cost = self.graph.path_cost(path)
        p = 1 / (1 + np.exp(cost - self.M))
        return 1 if np.random.random() < p else 0

    def expected_reward(self, t, path, edge_costs=None):
        assert self.graph.edges[path[0]][0] == self.source and self.graph.edges[path[-1]][1] == self.destination
        cost = self.graph.path_cost(path, edge_costs)
        return 1 / (1 + np.exp(cost - self.M))

    def best_action(self, t):
        if self._best_action is None:
            self._best_action = self.graph.shortest_path(self.source, self.destination)
        return self._best_action

    def best_reward(self, t):
        if self._best_reward is None:
            self._best_reward = self.expected_reward(t, self.best_action(t))
        return self._best_reward

if __name__ == '__main__':
    g = Graph(4, [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ], [1, 2, 3, 4])
    bandit = Bandit(g, 4, 0, 3)
    print(bandit.run([0, 2]))
