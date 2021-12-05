import numpy as np
import heapq as hq

class Graph:

    def __init__(self, num_vertices, edges, costs):
        self.num_vertices = num_vertices
        self.edges = dict(zip(edges, costs))
        self.adjacency_lists = {
            k: [e[1] for e in edges if e[0] == k] for k in range(num_vertices)
        }

    def shortest_path(self, source, destination, costs=None):
        if costs:
            edges = dict(zip(self.edges.keys(), costs))
        else:
            edges = self.edges
        visited = [False] * self.num_vertices
        weights = [np.infty] * self.num_vertices
        path = [None] * self.num_vertices
        queue = []
        weights[source] = 0
        hq.heappush(queue, (0, source))
        while len(queue) > 0:
            g, u = hq.heappop(queue)
            visited[u] = True
            for v in self.adjacency_lists[u]:
                w = self.edges[u, v]
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
            ret.append((path[c], c))
            c = path[c]
        return ret


if __name__ == '__main__':
    g = Graph(4, [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ], [1, 2, 3, 4])

    print(g.adjacency_lists)
    print(g.shortest_path(0, 3))