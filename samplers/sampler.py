from bandit import Bandit
import numpy as np
from abc import ABC, abstractmethod

class Sampler(ABC):
    def __init__(self, bandit: Bandit):
        self.bandit = bandit

    @abstractmethod
    def update(self, t, a, r):
        pass

    @abstractmethod
    def sample(self):
        return np.random.random(len(self.bandit.graph.edges))
