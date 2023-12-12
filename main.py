import numpy as np
from layers import *
from functions import *
from optimizers import *
from network import ELM
import datasets

units = 54
output_dim = 7

if __name__ == "__main__":
    elm = ELM(
        layers=[
            Layer(name="hidden",
                  input_dim=units,
                  output_dim=14,
                  weight_initializer="std",
                  activation=tanh,
                  trainable=False),
            Layer(name="output",
                  input_dim=14,
                  output_dim=output_dim,
                  weight_initializer="xavier",
                  trainable=True)
        ],
        loss=mse
    )
    # Uncomment the dataset you'd like to use
    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

    x_train, y_train, x_test, y_test = datasets.get_covertype()

    elm.fit(x_train, y_train, epochs=100, optimizer=MGD, lasso=0.001, learning_rate=0.005, momentum=0.6)

    out = elm.predict(x_test[0:3])
    print("\n")
    print("predicted values : ")
    print(out, end="\n")
    print("true values : ")
    print(y_test[0:3])
