from graph import Graph
from bandits.bandit import Bandit
from bandits.simple_bandit import SimpleBandit
from samplers.langevin import LangevinSampler
from algorithm import Algorithm
from simulator import Simulator
import networkx as nx
import matplotlib.pyplot as plt

def _show_action(graph: Graph, action, t, ax):
    G = graph.to_graph()

    edge_color = ['k'] * len(graph.edges)
    node_color = ['k'] * graph.num_vertices
    for a in action:
        edge_color[a] = 'r'
        node_color[graph.edges[a][0]] = 'r'
        node_color[graph.edges[a][1]] = 'r'
    nx.draw(
        G,
        ax=ax,
        pos=graph.get_layout(),
        edge_color=edge_color,
        node_color=node_color,
        width=[max(1., (1 / c)) for c in graph.costs],
    )
    plt.text(-1, -1, str(t))


class Animator:
    def __init__(self, simulator, bandit: Bandit, pause=0.001, skip=0):
        self.simulator = simulator
        self.bandit = bandit
        self.pause = pause

    def show_action(self, t, action, ax):
        _show_action(self.bandit.get_graph(t), action, t, ax)

    def show_graph(self, t, ax):
        _show_action(self.bandit.get_graph(t), [], ax)

    def show_regret(self, avg_regret, ax):
        ax.plot(list(range(len(avg_regret))), avg_regret, color='k')

    def run(self):
        fig, axs = plt.subplots(2)
        avg_regret = []
        with plt.ion():
            for t, a, r, regret in self.simulator:
                print(t, a, regret / t)
                avg_regret.append(regret / t)
                self.show_action(t, a, axs[0])
                self.show_regret(avg_regret, axs[1])
                plt.pause(self.pause)
                axs[1].cla()
                axs[0].cla()
        plt.show()
