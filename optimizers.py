import numpy as np


def MGD(gradient, weight, learning_rate: float, **kwargs):
    momentum = kwargs.get("momentum", 0.9)
    prev_weight = kwargs.get("prev_weight")
    # Apply the gradient descent method with momentum
    new_weight = np.add(np.subtract(weight, (learning_rate * gradient)), (momentum * np.subtract(weight, prev_weight)))

    return new_weight, weight
