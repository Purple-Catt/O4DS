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

    history = elm.fit(x_train, y_train,
                      x_test, y_test,
                      epochs=100,
                      optimizer=MGD,
                      lasso=0.0001,
                      learning_rate=0.0001,
                      history=True,
                      patience=3,
                      momentum=0.8)

    learning_curve(history=history)
