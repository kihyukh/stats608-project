import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from bandits.simple_bandit import SimpleBandit
from simulator import Simulator
from animator import Animator
from graph import Graph
import json

def demo_graph1():
    g = Graph(
        12,
        {
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (0, 4): 1,
            (0, 5): 1,
            (1, 6): 1,
            (2, 6): 1,
            (3, 7): 0.2,
            (4, 8): 1,
            (5, 8): 1,
            (6, 9): 1,
            (7, 9): 1,
            (7, 10): 1,
            (8, 10): 1,
            (9, 11): 0.2,
            (10, 11): 1,
        },
        {
            0: [-1, 0],
            1: [-0.5, 1],
            2: [-0.5, 0.5],
            3: [-0.5, 0],
            4: [-0.5, -0.5],
            5: [-0.5, -1],
            6: [0, 0.5],
            7: [0, 0],
            8: [0, -0.5],
            9: [0.5, 0.25],
            10: [0.5, -0.25],
            11: [1, 0],
        }
    )
    return g

def demo_graph2():
    g = Graph(
        12,
        {
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (0, 4): 0.2,
            (0, 5): 1,
            (1, 6): 1,
            (2, 6): 1,
            (3, 7): 0.2,
            (4, 8): 0.2,
            (5, 8): 1,
            (6, 9): 1,
            (7, 9): 1,
            (7, 10): 1,
            (8, 10): 0.2,
            (9, 11): 0.2,
            (10, 11): 0.2,
        },
        {
            0: [-1, 0],
            1: [-0.5, 1],
            2: [-0.5, 0.5],
            3: [-0.5, 0],
            4: [-0.5, -0.5],
            5: [-0.5, -1],
            6: [0, 0.5],
            7: [0, 0],
            8: [0, -0.5],
            9: [0.5, 0.25],
            10: [0.5, -0.25],
            11: [1, 0],
        }
    )
    return g


def filestream(filename):
    for line in open(filename, 'r'):
        t, a, r, regret = line.rstrip('\n').split('\t', 3)
        regret = float(regret.split('\t')[0])
        yield int(t), json.loads(a), int(r), regret


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--log')
parser.add_argument('--bandit')

args = parser.parse_args()

if args.bandit == '1':
    bandit = SimpleBandit(
        graph=demo_graph1(), M=3, source=0, destination=11, T=1000)
else:
    bandit1 = SimpleBandit(
        graph=demo_graph1(), M=3, source=0, destination=11, T=1000)
    bandit2 = SimpleBandit(
        graph=demo_graph2(), M=3, source=0, destination=11, T=1000)
    bandit = SwitchingBandit(bandit1, bandit2, 501)

filename = args.log

simulator = filestream(filename)
animator = Animator(simulator, bandit, pause=0.000001)
animator.run()