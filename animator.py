from graph import Graph
from bandits.bandit import Bandit
from bandits.simple_bandit import SimpleBandit
from samplers.langevin import LangevinSampler
from algorithm import Algorithm
from simulator import Simulator
import networkx as nx
import matplotlib.pyplot as plt

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
        self.bandit = simulator.bandit

    def show_action(self, t, action):
        _show_action(self.bandit.get_graph(t), action)

    def show_graph(self, t):
        _show_action(self.bandit.get_graph(t), [])

    def run(self):
        with plt.ion():
            for t, a, r, regret in self.simulator:
                print(t, a, regret / t)
                self.show_action(t, a)
                plt.pause(0.001)
                plt.clf()
                self.show_graph(t)
                plt.pause(0.001)
        plt.show()
