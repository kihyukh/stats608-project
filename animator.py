from graph import Graph
from bandit import Bandit
from samplers.langevin import LangevinSampler
from algorithm import Algorithm
from simulator import Simulator
import networkx as nx
import matplotlib.pyplot as plt
from demo import demo_graph1

def _show_action(graph: Graph, action):
    G = graph.to_graph()

    edge_color = ['k'] * len(graph.edges)
    for a in action:
        edge_color[a] = 'r'
    nx.draw(
        G,
        pos=graph.get_layout(),
        edge_color=edge_color,
        width=[(1 / c) for c in graph.costs],
    )


class Animator:
    def __init__(self, simulator: Simulator):
        self.simulator = simulator
        self.graph = simulator.bandit.graph

    def show_action(self, action):
        _show_action(self.graph, action)

    def show_graph(self):
        _show_action(self.graph, [])

    def run(self):
        with plt.ion():
            for t, a, r, regret in self.simulator:
                print(t, a, regret / t)
                self.show_action(a)
                plt.pause(0.001)
                plt.clf()
                self.show_graph()
                plt.pause(0.001)
        plt.show()

if __name__ == '__main__':
    g = demo_graph1()
    bandit = Bandit(
        graph=g, M=3, source=0, destination=11, T=100)
    sampler = LangevinSampler(bandit, 3, 3)
    algorithm = Algorithm(bandit, sampler)
    sim = Simulator(bandit, algorithm)
    animator = Animator(sim)
    animator.run()
