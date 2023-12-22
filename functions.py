import numpy as np

from layers import Layer
from network import ELM
from optimizers import *
import matplotlib.pyplot as plt


def l1_regularizer(x, l1: float):
    """Return the LASSO regularizer given a layer output."""
    penalty = l1 * np.sum(np.abs(x))

    return penalty


def mse(y_true, y_pred, weight, lasso: float = None):
    """Compute the Mean Square Error with an optional L1 regularization term, if given."""
    if lasso is not None:
        reg = l1_regularizer(weight, l1=lasso)
        loss = 0.5 * np.sum(np.add(np.power(y_true - y_pred, 2), reg))

    else:
        loss = 0.5 * np.sum(np.power(y_true - y_pred, 2))

    return loss


def gradient(y_true, y_pred, weight, inputs, lasso: float = 0.0):
    matr = np.matmul(inputs.T, (y_pred - y_true))
    grad = np.where(weight < 0, matr - lasso, matr + lasso)

    return grad


def relu(x, prime: bool = False):
    """Compute the ReLU activation function. if *prime* is True, compute the first derivative."""
    if prime:
        return np.greater(x, 0).astype(int)

    else:
        return np.maximum(0, x)


def tanh(x, prime: bool = False):
    """Compute the tanh activation function. if *prime* is True, compute the first derivative."""
    if prime:
        return 1 - (np.tanh(x) ** 2)

    else:
        return np.tanh(x)


def sigmoid(x, prime: bool = False):
    """Compute the sigmoid activation function. If *prime* is True, compute the first derivative."""
    if prime:
        return (1 / (1 + np.exp(- x))) * (1 - (1 / (1 + np.exp(- x))))

    else:
        return 1 / (1 + np.exp(- x))


def splitting_function(x: np.array, y: np.array, train: float, val: float = 0.0):
    """Split the given numpy arrays into train, test and eventually validation sets."""
    n = len(x)
    x_train = x[:int(n * (train - val))]
    y_train = y[:int(n * (train - val))]
    x_test = x[int(n * train):]
    y_test = y[int(n * train):]
    if val != 0.0:
        x_val = x[int(n * train - val):int(n * train)]
        y_val = y[int(n * train - val):int(n * train)]
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test


def learning_curve(history: dict):
    plt.plot(range(history["epochs"]), history["val_loss"], "b-", label="Validation loss")
    plt.plot(range(history["epochs"]), history["loss"], "r-", label="Train loss")
    plt.title("Learning curve")
    plt.legend()
    plt.show()


def get_model(input_dim: int, mid_dim: int, output_dim: int):
    elm = ELM(
        layers=[
            Layer(name="hidden",
                  input_dim=input_dim,
                  output_dim=mid_dim,
                  weight_initializer="std",
                  activation=tanh,
                  trainable=False),
            Layer(name="output",
                  input_dim=mid_dim,
                  output_dim=output_dim,
                  weight_initializer="xavier",
                  trainable=True)
        ],
        loss=mse
    )

    return elm


def gridsearch_mgd(x_train, y_train, x_test, y_test):
    """IT STILL NEEDS TO BE IMPLEMENTED"""
    pass
