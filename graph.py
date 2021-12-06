import numpy as np
import heapq as hq

class Graph:

    def __init__(self, num_vertices, edges, costs):
        self.num_vertices = num_vertices
        self.edges = edges
        self.costs = costs
        self.edge_lists = {
            k: [i for i, e in enumerate(edges) if e[0] == k] for k in range(num_vertices)
        }
        self.edge_index_map = {
            (u, v): i for i, (u, v) in enumerate(edges)
        }

    def is_path(self, path):
        for i, p in enumerate(path):
            if p < 0 or p >= len(self.edges):
                return False
            if i > 0 and self.edges[path[i - 1]][1] != self.edges[p][0]:
                return False
        return True

    def path_cost(self, path, costs=None):
        assert self.is_path(path)
        costs = self.costs if costs is None else costs
        return sum([costs[p] for p in path])

    def shortest_path(self, source, destination, costs=None):
        costs = self.costs if costs is None else costs
        visited = [False] * self.num_vertices
        weights = [np.infty] * self.num_vertices
        path = [None] * self.num_vertices
        queue = []
        weights[source] = 0
        hq.heappush(queue, (0, source))
        while len(queue) > 0:
            g, u = hq.heappop(queue)
            visited[u] = True
            for e in self.edge_lists[u]:
                w = costs[e]
                v = self.edges[e][1]
                if not visited[v]:
                    f = g + w
                    if f < weights[v]:
                        weights[v] = f
                        path[v] = u
                        hq.heappush(queue, (f, v))
            if u == destination:
                break
        ret = []
        c = destination
        while path[c] is not None:
            ret.append(self.edge_index_map[path[c], c])
            c = path[c]
        ret.reverse()
        return ret


if __name__ == '__main__':
    g = Graph(4, [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ], [1, 2, 3, 4])

    print(g.edge_lists)
    print(g.shortest_path(0, 3))