from bandits.bandit import Bandit
import numpy as np
from abc import ABC, abstractmethod

class Sampler(ABC):
    def __init__(self, bandit: Bandit, alpha, beta):
        self.bandit = bandit
        self.alpha = alpha
        self.beta = beta

    @abstractmethod
    def update(self, t, a, r):
        pass

    @abstractmethod
    def sample(self, t):
        return np.random.random(len(self.bandit.graph.edges))
