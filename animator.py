from graph import Graph
import networkx as nx
import matplotlib.pyplot as plt

class Animator:
    def __init__(self, graph: Graph):
        self.graph = graph

    def show(self):
        G = self.graph.to_graph()
        nx.draw(G, pos=self.graph.get_layout())
        plt.show()


if __name__ == '__main__':
    from demo import demo_graph1
    g = demo_graph1()
    animator = Animator(g)
    animator.show()