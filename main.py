import numpy as np
import pandas as pd
from functions import *
from optimizers import *
from network import ELM
import datasets

var = True
grad = True
units, mid_out, output_dim = datasets.get_dim("mnist")
x_train, y_train, x_test, y_test = datasets.get_mnist(True)
# matrix = np.array(pd.read_csv("03-hidden_matrix.csv", header=None))

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
        history = elm.fit(x=x_train, y=y_train,
                          x_test=x_test, y_test=y_test,
                          epochs=1000000,
                          optimizer=mgd,
                          lasso=0.0000001,
                          history=True,
                          patience=10,
                          loss_evaluation="train",
                          verbose=1,
                          learning_rate=0.000001,
                          momentum=0.9)

    else:
        history = elm.fit(x=x_train, y=y_train,
                          x_test=x_test, y_test=y_test,
                          epochs=1000,
                          optimizer=dsg,
                          lasso=0.000001,
                          history=True,
                          patience=3,
                          verbose=2,
                          gamma=0.9,
                          R=1,
                          rho=0.5,
                          delta=0.1,
                          beta=0.8)

    learning_curve(history=history)
    pd.DataFrame(elm.get_weight).to_csv("03-hidden_matrix.csv", header=False, index=False)

else:
    gridsearch(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
               input_dim=units, output_dim=output_dim, epochs=100, gs_type="mgd", lasso=0.0000001, mid_dim=mid_out)
