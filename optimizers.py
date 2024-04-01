import numpy as np


def MGD(gradient, weight, prev_weight, learning_rate: float, momentum: float):
    """It's a standard heavy ball approach, so it applies the gradient descend with momentum.
    If fixed_step is False, the learning rate will be divided by the norm of the gradient."""

    new_weight = np.add(np.subtract(weight, (learning_rate * gradient)), (momentum * np.subtract(weight, prev_weight)))

    return new_weight, weight


def DSG(subgradient, weight, lasso, gamma, R, rho, r, delta, beta, fx, f_ref, f_bar, prev_d):
    def threshold_function(x, lamb):
        return np.where(x > 0, 1, -1) * np.greater(np.abs(x) - lamb, 0).astype(np.float64)

    d = np.add(gamma * subgradient, (1 - gamma) * prev_d)
    prev_d = d
    alpha = beta * (np.subtract(fx, np.subtract(f_ref, delta)) / (np.linalg.norm(d) ** 2))
    temp = np.subtract(weight, alpha * d)
    new_weight = threshold_function(temp, lasso)

    if fx <= (f_ref - (delta / 2)):  # Sufficient descent direction
        f_ref = f_bar
        r = 0
        exit()

    elif r > R:  # Target infeasibility condition
        delta = delta * rho
        r = 0

    else:
        s = beta * (np.subtract(fx, np.subtract(f_ref, delta)) / np.linalg.norm(d))
        r = r + s

    f_bar = np.minimum(f_bar, fx)

    return new_weight, r, delta, f_ref, f_bar, prev_d
