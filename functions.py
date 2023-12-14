import numpy as np
from layers import Layer
from network import ELM
from optimizers import *


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


def tanh(x, prime: bool = False):

    if prime:
        res = 1 - (np.tanh(x) ** 2)

    else:
        res = np.tanh(x)

    return res


def sigmoid(x, prime: bool = False):
    """Compute the sigmoid activation function. If *prime* is True, compute the first derivative."""
    if prime:
        res = (1 / (1 + np.exp(- x))) * (1 - (1 / (1 + np.exp(- x))))
    else:
        res = 1 / (1 + np.exp(- x))

    return res


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
    learning_rate = np.arange(start=0.008, stop=0.016, step=0.002)
    learning_rate = [float(round(i, 4)) for i in list(learning_rate)]

    momentum = np.arange(start=0.5, stop=1, step=0.1)
    momentum = [float(round(i, 1)) for i in list(momentum)]

    lasso = np.arange(start=0.0001, stop=0.001, step=0.0002)
    lasso = [float(round(i, 4)) for i in list(lasso)]

    for lr in learning_rate:
        for m in momentum:
            for l1 in lasso:
                model = get_model(input_dim=28*28, mid_dim=100, output_dim=10)
                model.fit(x_train, y_train, epochs=100, optimizer=MGD, lasso=l1, learning_rate=lr, momentum=m)
