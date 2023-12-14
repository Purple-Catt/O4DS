import numpy as np
from numpy.random import standard_normal, uniform, seed
from optimizers import MGD

# Set a random seed for replicability purposes
seed(5)


class Layer:

    def __init__(self, name: str, input_dim: int, output_dim: int,
                 weight_initializer: str, activation=None, trainable: bool = False):
        """Parameters:\n
        - *input_dim*: the number of input units (neurons) of the layer
        - *output_dim*: the number of output units
        - *weight_initializer*: *str* between 'std' for Standard Normal or 'xavier' for Normalized Xavier
        - *activation*: *str* between 'tanh' for hyperbolic tangent or 'sigm' for sigmoidal function. Default is *None*, so no activation function is used."""

        self.name = name
        if weight_initializer == "std":
            self.weight = standard_normal(size=(input_dim, output_dim))
        elif weight_initializer == "xavier":
            bound = (np.sqrt(6)/np.sqrt(input_dim + output_dim))
            self.weight = uniform(low=-bound, high=bound, size=(input_dim, output_dim))
        else:
            raise ValueError(f"*str* between 'std' and 'xavier' expected, got {weight_initializer} instead.")

        self.trainable = trainable
        self.activation = activation
        self.prev_weight = None
        self.inputs = None
        self.output = None

    def call(self, inputs):
        """When the layer is called during the training process, this function will be used
        to compute the output of the single layer."""
        self.inputs = inputs

        if self.activation is None:
            self.output = np.matmul(self.inputs, self.weight)

        else:
            self.output = self.activation(np.matmul(self.inputs, self.weight))

        return self.output

    def weights_update(self, gradient, optimizer, learning_rate: float, **kwargs):
        params = kwargs
        if self.trainable:
            if self.activation is None:

                if optimizer == MGD:

                    if self.prev_weight is None:
                        self.prev_weight = self.weight.copy()

                    params["prev_weight"] = self.prev_weight
                self.weight, self.prev_weight = optimizer(gradient=gradient,
                                                          weight=self.weight,
                                                          learning_rate=learning_rate,
                                                          **params)

            else:
                raise NotImplementedError("The backpropagation algorithm for trainable layers hasn't been "
                                          "implemented and it won't be, due to the fact that the ELM used in the "
                                          "project doesn't need it.")

        else:
            raise AttributeError("The called layer doesn't have trainable weights.")
