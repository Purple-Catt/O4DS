import numpy as np


def MGD(gradient, weight, learning_rate: float, **kwargs):
    """It's a standard heavy ball approach, so it applies the gradient descend with momentum"""
    momentum = kwargs.get("momentum", 0.9)
    prev_weight = kwargs.get("prev_weight")
    new_weight = np.add(np.subtract(weight, (learning_rate * gradient)), (momentum * np.subtract(weight, prev_weight)))

    return new_weight, weight


def DSG():
    pass
