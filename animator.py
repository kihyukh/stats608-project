from graph import Graph
from bandits.bandit import Bandit
from bandits.simple_bandit import SimpleBandit
from samplers.langevin import LangevinSampler
from algorithm import Algorithm
from simulator import Simulator
import networkx as nx
import matplotlib.pyplot as plt
from demo import demo_graph1

def _show_action(graph: Graph, action):
    G = graph.to_graph()

    edge_color = ['k'] * len(graph.edges)
    node_color = ['k'] * graph.num_vertices
    for a in action:
        edge_color[a] = 'r'
        node_color[graph.edges[a][0]] = 'r'
        node_color[graph.edges[a][1]] = 'r'
    nx.draw(
        G,
        pos=graph.get_layout(),
        edge_color=edge_color,
        node_color=node_color,
        width=[max(1.5, (1 / c)) for c in graph.costs],
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
    bandit = SimpleBandit(
        graph=g, M=3, source=0, destination=11, T=200)
    sampler = LangevinSampler(bandit, 2, 2, stochastic=50)
    algorithm = Algorithm(bandit, sampler)
    sim = Simulator(bandit, algorithm)
    animator = Animator(sim)
    animator.run()
