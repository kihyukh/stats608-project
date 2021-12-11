from demo import demo_graph1, demo_graph2
from bandits.simple_bandit import SimpleBandit
from bandits.bandit import Bandit

class SwitchingBandit(Bandit):
    _best_action = None
    _best_reward = None

    def __init__(self, bandit1: Bandit, bandit2: Bandit, switch_time):
        super().__init__(
            bandit1.graph,
            bandit1.M,
            bandit1.source,
            bandit1.destination,
            bandit1.T)
        self.bandit1 = bandit1
        self.bandit2 = bandit2
        self.switch_time = switch_time

    def run(self, t, action):
        if t < self.switch_time:
            return self.bandit1.run(t, action)
        return self.bandit2.run(t, action)

    def expected_reward(self, t, action, edge_costs=None):
        if t < self.switch_time:
            return self.bandit1.expected_reward(t, action, edge_costs)
        return self.bandit2.expected_reward(t, action, edge_costs)

    def best_action(self, t):
        if t < self.switch_time:
            return self.bandit1.best_action(t)
        return self.bandit2.best_action(t)

    def best_reward(self, t):
        if t < self.switch_time:
            return self.bandit1.best_reward(t)
        return self.bandit2.best_reward(t)

    def get_graph(self, t):
        if t < self.switch_time:
            return self.bandit1.get_graph(t)
        return self.bandit2.get_graph(t)


if __name__ == '__main__':
    bandit1 = SimpleBandit(
        graph=demo_graph1(),
        M=2,
        source=0,
        destination=11,
        T=100)
    bandit2 = SimpleBandit(
        graph=demo_graph2(),
        M=2,
        source=0,
        destination=11,
        T=100)
    switching_bandit = SwitchingBandit(bandit1, bandit2, switch_time)
    switching_bandit.run(30, [1, 2])
