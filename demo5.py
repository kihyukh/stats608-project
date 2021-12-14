from graph import Graph
from samplers.langevin import LangevinSampler
from bandits.simple_bandit import SimpleBandit
from bandits.switching_bandit import SwitchingBandit
from algorithm import Algorithm
from simulator import Simulator
from animator import Animator
import numpy as np

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

if __name__ == '__main__':
    np.random.seed(608)
    bandit1 = SimpleBandit(
        graph=demo_graph1(), M=3, source=0, destination=11, T=600)
    bandit2 = SimpleBandit(
        graph=demo_graph2(), M=3, source=0, destination=11, T=600)
    bandit = SwitchingBandit(bandit1, bandit2, 201)
    sampler = LangevinSampler(bandit1, 2, 0.2, stochastic=50, window=40)
    algorithm = Algorithm(bandit1, sampler)
    sim = Simulator(bandit, algorithm)
    animator = Animator(sim, bandit)
    animator.run()
