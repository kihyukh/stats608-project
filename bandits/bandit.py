import numpy as np
from graph import Graph
from abc import ABC, abstractmethod


class Bandit(ABC):
    def __init__(self, graph: Graph, M, source, destination, T):
        self.graph = graph
        self.M = M
        self.source = source
        self.destination = destination
        self.T = T

    @abstractmethod
    def run(self, t, action):
        pass

    @abstractmethod
    def expected_reward(self, t, action):
        pass

    @abstractmethod
    def best_action(self, t):
        pass

    @abstractmethod
    def best_reward(self, t):
        pass