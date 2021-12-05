import numpy as np
from graph import Graph

class Bandit:
    _best_action = None
    _best_reward = None

    def __init__(self, graph: Graph, M, source, destination):
        self.graph = graph
        self.M = M
        self.source = source
        self.destination = destination

    def run(self, path):
        assert path[0][0] == self.source and path[-1][1] == self.destination
        cost = self.graph.path_cost(path)
        p = 1 / (1 + np.exp(cost - self.M))
        return 1 if np.random.random() < p else 0

    def expected_reward(self, path):
        assert path[0][0] == self.source and path[-1][1] == self.destination
        cost = self.graph.path_cost(path)
        return 1 / (1 + np.exp(cost - self.M))

    def best_action(self):
        if self._best_action is None:
            self._best_action = self.graph.shortest_path(self.source, self.destination)
        return self._best_action

    def best_reward(self):
        if self._best_reward is None:
            self._best_reward = self.expected_reward(self.best_action())
        return self._best_reward

if __name__ == '__main__':
    g = Graph(4, [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ], [1, 2, 3, 4])
    bandit = Bandit(g, 4, 0, 3)
    print(bandit.run([(0, 1), (1, 3)]))