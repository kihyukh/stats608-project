from bandits.bandit import Bandit
import numpy as np
import random

def gradient_hessian(theta, bandit: Bandit, history, alpha, beta):
    E = len(bandit.graph.edges)
    p = np.array([
        bandit.expected_reward(1, a, theta)
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

def stochastic_sampling(history, stochastic):
    if not history:
        return []
    if not stochastic:
        return history
    if stochastic >= len(history):
        return history
    indices = np.random.choice(range(len(history)), stochastic, replace=False)
    return [history[i] for i in indices]

def langevin_sampling(bandit: Bandit, history, alpha, beta, epsilon, stochastic, B):
    theta = find_mode(bandit, history, alpha, beta)
    assert np.min(theta) > 0
    _, hessian = gradient_hessian(
        theta, bandit, history, alpha, beta)
    A = -np.linalg.inv(hessian)
    A_sqrt = sqrtm(A)

    E = len(bandit.graph.edges)
    for b in range(B):
        h = stochastic_sampling(history, stochastic)
        gradient, _ = gradient_hessian(
            theta, bandit, h, alpha, beta)
        W = np.random.normal(0, 1, E)
        theta += (
            epsilon * (A @ gradient) +
            np.sqrt(2 * epsilon) * (A_sqrt @ W)
        )

    return theta
