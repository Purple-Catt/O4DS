from functions import *
from optimizers import *
from network import ELM
import datasets

var = False
grad = True
units, mid_out, output_dim = datasets.get_dim("cifar100")
x_train, y_train, x_test, y_test = datasets.get_cifar100()

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
                          epochs=10000,
                          optimizer=MGD,
                          lasso=0.000001,
                          history=True,
                          patience=100,
                          verbose=1,
                          learning_rate=0.000001,
                          momentum=0.9)

    else:
        history = elm.fit(x_train, y_train,
                          x_test, y_test,
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
    gridsearch(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
               input_dim=units, output_dim=output_dim, epochs=100, gs_type="mgd", lasso=0.0000001, mid_dim=mid_out)
