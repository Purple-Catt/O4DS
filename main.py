import numpy as np
from layers import *
from functions import *
from optimizers import *
from network import ELM
import datasets

units = 28 * 28
mid_out = 28 * 28 * 2
output_dim = 10

if __name__ == "__main__":
    elm = ELM(
        layers=[
            Layer(name="hidden",
                  input_dim=units,
                  output_dim=mid_out,
                  weight_initializer="std",
                  activation=tanh,
                  trainable=False),
            Layer(name="output",
                  input_dim=mid_out,
                  output_dim=output_dim,
                  weight_initializer="xavier",
                  trainable=True)
        ],
        loss=mse
    )

    x_train, y_train, x_test, y_test = datasets.get_mnist()

    elm.fit(x_train[:1000], y_train[:1000], x_test[:200], y_test[:200], epochs=5000, optimizer=MGD, lasso=0.0001, learning_rate=0.0001, momentum=0.8)
