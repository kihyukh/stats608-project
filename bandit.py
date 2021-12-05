import numpy as np
from graph import Graph

class Bandit:
    def __init__(self, graph: Graph, M, source, destination):
        self.graph = graph
        self.M = M
        self.source = source
        self.destination = destination

    def action(self, path):
        assert path[0][0] == self.source and path[-1][1] == self.destination
        cost = self.graph.path_cost(path)
        p = 1 / (1 + np.exp(cost - self.M))
        return 1 if np.random.random() < p else 0

if __name__ == '__main__':
    g = Graph(4, [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ], [1, 2, 3, 4])
    bandit = Bandit(g, 4, 0, 3)
    print(bandit.action([(0, 1), (1, 3)]))
