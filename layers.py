from numpy import matmul
from numpy.random import standard_normal, normal, uniform, seed
from optimizers import *

# Set a random seed for replicability purposes
seed(5)


class Layer:

    def __init__(self, name: str, input_dim: int, output_dim: int,
                 weight_initializer: str, activation=None, trainable: bool = False):
        """Parameters:\n
            *input_dim*: the number of input units (neurons) of the layer\n
            *output_dim*: the number of _output units\n
            *weight_initializer*: *str* between 'std' for Standard Normal, 'xavier' for Normalized Xavier
            or 'he' for He Normal.\n
            *activation*: *str* between 'tanh' for hyperbolic tangent or 'sigm' for sigmoidal function. Default is
            *None*, so no activation function is used."""

        self.name = name
        if weight_initializer == "std":
            self._weight = standard_normal(size=(input_dim, output_dim))
        elif weight_initializer == "xavier":
            bound = (np.sqrt(6)/np.sqrt(input_dim + output_dim))
            self._weight = uniform(low=-bound, high=bound, size=(input_dim, output_dim))
        elif weight_initializer == "he":
            self._weight = normal(loc=0.0, scale=np.sqrt(2 / input_dim), size=(input_dim, output_dim))
        else:
            raise ValueError(f"*str* between 'std' and 'xavier' expected, got {weight_initializer} instead.")

        self._trainable = trainable
        self._activation = activation
        self.__additional_parameters = {}
        self._inputs = None
        self._output = None

    def __call__(self, inputs):
        """When the layer is called during the training process, this function will be used
        to compute the _output of the single layer."""
        self._inputs = inputs

        if self._activation is None:
            self._output = matmul(self._inputs, self._weight)

        else:
            self._output = self._activation(matmul(self._inputs, self._weight))

        return self._output

    def weights_update(self, gradient, optimizer, **kwargs):
        if self._trainable:
            if self._activation is None:

                if optimizer == MGD:

                    if self.__additional_parameters.get("prev_weight") is None:
                        self.__additional_parameters["prev_weight"] = self._weight.copy()

                    self._weight, self.__additional_parameters["prev_weight"] = (
                        optimizer(gradient=gradient,
                                  weight=self._weight,
                                  prev_weight=self.__additional_parameters["prev_weight"],
                                  learning_rate=kwargs["learning_rate"],
                                  momentum=kwargs["momentum"]
                                  ))

                elif optimizer == DSG:

                    # Initialize the parameters before the algorithm runs for the first time
                    if self.__additional_parameters == {}:
                        self.__additional_parameters.update(kwargs)
                        self.__additional_parameters["r"] = 0
                        self.__additional_parameters["f_ref"] = self.__additional_parameters["fx"]
                        self.__additional_parameters["f_bar"] = self.__additional_parameters["fx"]
                        self.__additional_parameters["prev_d"] = 0.0

                    # Update the function value 'fx' from the second iteration on
                    else:
                        self.__additional_parameters["fx"] = kwargs["fx"]

                    (self._weight, r, delta, f_ref,
                     f_bar, prev_d) = optimizer(subgradient=gradient,
                                                weight=self._weight,
                                                lasso=self.__additional_parameters["lasso"],
                                                gamma=self.__additional_parameters["gamma"],
                                                R=self.__additional_parameters["R"],
                                                rho=self.__additional_parameters["rho"],
                                                r=self.__additional_parameters["r"],
                                                delta=self.__additional_parameters["delta"],
                                                beta=self.__additional_parameters["beta"],
                                                fx=self.__additional_parameters["fx"],
                                                f_ref=self.__additional_parameters["f_ref"],
                                                f_bar=self.__additional_parameters["f_bar"],
                                                prev_d=self.__additional_parameters["prev_d"])

                    self.__additional_parameters["r"] = r
                    self.__additional_parameters["delta"] = delta
                    self.__additional_parameters["f_ref"] = f_ref
                    self.__additional_parameters["f_bar"] = f_bar
                    self.__additional_parameters["prev_d"] = prev_d

                else:
                    raise ValueError(f"One between MGD and DSG expected, got {optimizer} instead.")

            else:
                raise NotImplementedError("The backpropagation algorithm for trainable layers hasn't been "
                                          "implemented and it won't be, due to the fact that the ELM used in the "
                                          "project doesn't need it.")

        else:
            raise AttributeError("The called layer doesn't have trainable weights.")

    @property
    def inputs(self):
        return self._inputs

    @property
    def weight(self):
        return self._weight
