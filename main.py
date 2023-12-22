from functions import *
from optimizers import *
from network import ELM
import datasets

units, mid_out, output_dim = datasets.get_dim("mnist")

elm = ELM(
    layers=[
        Layer(name="hidden",
              input_dim=units,
              output_dim=mid_out,
              weight_initializer="std",
              activation=relu,
              trainable=False),
        Layer(name="output",
              input_dim=mid_out,
              output_dim=output_dim,
              weight_initializer="he",
              trainable=True)
    ],
    loss=mse
)

x_train, y_train, x_test, y_test = datasets.get_mnist()

history = elm.fit(x_train, y_train,
                  x_test, y_test,
                  epochs=1000,
                  optimizer=MGD,
                  lasso=0.00001,
                  learning_rate=0.00001,
                  history=True,
                  patience=3,
                  momentum=0.4)

learning_curve(history=history)
