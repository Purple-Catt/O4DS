from functions import *
from optimizers import *
from network import ELM
import datasets

var = True
grad = True
units, mid_out, output_dim = datasets.get_dim("matyas")
x_train, y_train, x_test, y_test = datasets.get_matyas()

if var:
    elm = ELM(
        layers=[
            Layer(name="hidden",
                  input_dim=units,
                  output_dim=mid_out,
                  weight_initializer="std",
                  activation=softplus,
                  trainable=False),
            Layer(name="output",
                  input_dim=mid_out,
                  output_dim=output_dim,
                  weight_initializer="he",
                  trainable=True)
        ],
        loss=mse
    )
    if grad:
        history = elm.fit(x_train, y_train,
                          x_test, y_test,
                          epochs=1000,
                          optimizer=MGD,
                          lasso=0.0001,
                          history=True,
                          patience=300,
                          verbose=1,
                          learning_rate=0.004,
                          momentum=0.5)

    else:
        history = elm.fit(x_train[:1000], y_train[:1000],
                          x_test[:200], y_test[:200],
                          epochs=1000,
                          optimizer=DSG,
                          lasso=0.00001,
                          history=True,
                          patience=3,
                          verbose=2,
                          gamma=0.9,
                          R=1,
                          rho=0.5,
                          delta=0.1,
                          beta=0.8)

    learning_curve(history=history)

else:
    gridsearch_mgd(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                   input_dim=units, mid_dim=mid_out, output_dim=output_dim)
