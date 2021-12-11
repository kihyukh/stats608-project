from bandits.bandit import Bandit
import numpy as np

def gradient_hessian(theta, bandit: Bandit, history, alpha, beta):
    E = len(bandit.graph.edges)
    p = np.array([
        bandit.expected_reward(a, theta)
        for i, (a, r) in enumerate(history)
    ])
    y = np.array([r for i, (a, r) in enumerate(history)])
    gradient = (alpha - 1) / theta - beta
    for i, (path, _) in enumerate(history):
        for edge in path:
            gradient[edge] += p[i] - y[i]
    hessian = np.zeros((E, E))
    for e in range(E):
        hessian[e, e] = -(alpha - 1) / theta[e] / theta[e]
    for i, (path, _) in enumerate(history):
        for e in path:
            hessian[e, e] -= p[i] * (1 - p[i])
        for e in path:
            for f in path:
                if e == f:
                    continue
                hessian[e, f] = -p[i] * (1 - p[i])
    return gradient, hessian


def find_mode(bandit: Bandit, history, alpha, beta, theta_0=None):
    E = len(bandit.graph.edges)
    theta = np.array([2.0] * E) if theta_0 is None else theta_0
    for _ in range(20):
        gradient, hessian = gradient_hessian(
            theta, bandit, history, alpha, beta)
        target = theta - np.linalg.inv(hessian) @ gradient
        n = 1
        while np.min(target) <= 0:
            target = theta - np.linalg.inv(hessian) @ gradient * (2 ** (-n))
            n += 1
            if n >= 16:
                print(theta)
                raise
        if np.max(np.abs(target - theta)) < 0.00001:
            break
        theta = target
    return theta

def sqrtm(A):
    evalues, evectors = np.linalg.eigh(A)
    assert (evalues >= 0).all()
    return evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)

def stochastic_sampling(history, t, stochastic):
    if t == 1:
        ret = []
    elif stochastic:
        if stochastic >= t - 1:
            ret = history[:t - 1]
        else:
            rand_indices = np.random.choice(range(t - 1), stochastic, replace=False)
            ret = [history[a] for a in rand_indices]
    else:
        ret = history[:t - 1]
    return ret
