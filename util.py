from bandit import Bandit
import numpy as np

def gradient_hessian(theta, bandit: Bandit, history, t, alpha, beta):
    E = len(bandit.graph.edges)
    p = np.array([
        bandit.expected_reward(a, theta)
        for i, (a, r) in enumerate(history) if i < t - 1
    ])
    y = np.array([r for i, (a, r) in enumerate(history) if i < t - 1])
    gradient = (alpha - 1) / theta - beta
    for i, (path, _) in enumerate(history[:t - 1]):
        for edge in path:
            gradient[edge] += p[i] - y[i]
    hessian = np.zeros((E, E))
    for e in range(E):
        hessian[e, e] = -(alpha - 1) / theta[e] / theta[e]
    for i, (path, _) in enumerate(history[:t - 1]):
        for e in path:
            hessian[e, e] -= p[i] * (1 - p[i])
        for e in path:
            for f in path:
                if e == f:
                    continue
                hessian[e, f] = -p[i] * (1 - p[i])
    return gradient, hessian


def find_mode(bandit: Bandit, history, t, alpha, beta, theta_0=None):
    assert len(history) >= t - 1
    E = len(bandit.graph.edges)
    theta = np.array([2.0] * E) if theta_0 is None else theta_0
    for _ in range(20):
        gradient, hessian = gradient_hessian(
            theta, bandit, history, t, alpha, beta)
        target = theta - np.linalg.inv(hessian) @ gradient
        if np.min(target) <= 0:
            target = theta - np.linalg.inv(hessian) @ gradient * 0.1
        if np.max(np.abs(target - theta)) < 0.00001:
            break
        theta = target
    return theta

def sqrtm(A):
    evalues, evectors = np.linalg.eig(A)
    assert (evalues >= 0).all()
    return evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
