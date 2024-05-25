from numpy import matmul
from numpy.random import PCG64, SeedSequence, Generator
from optimizers import *


class Layer:

    def __init__(self, name: str, input_dim: int, output_dim: int,
                 weight_initializer: str | np.ndarray, activation=None,
                 trainable: bool = False, random_state: int = 42):
        """Class to create layers into an ELM.
        Parameters:
            name: name of the layer
            input_dim: the number of input units (neurons) of the layer
            output_dim: the number of _output units
            weight_initializer: *str* between 'std' for Standard Normal, 'xavier' for Normalized Xavier
            or 'he' for He Normal, if numpy array is given, it'll be used as weight matrix
            activation: *str* between 'tanh' for hyperbolic tangent or 'sigm' for sigmoidal function. Default is *None*,
                so no activation function is used
            trainable: Default False. If True, the layer's weights are trained during the optimization process
            random_state: random seed to allow replicability."""

        self.name = name
        self.rng = Generator(PCG64(SeedSequence(random_state)))

        if isinstance(weight_initializer, np.ndarray):
            self._weight = weight_initializer

        elif weight_initializer == "std":
            self._weight = self.rng.standard_normal(size=(input_dim, output_dim))

        elif weight_initializer == "xavier":
            bound = (np.sqrt(6) / np.sqrt(input_dim + output_dim))
            self._weight = self.rng.uniform(low=-bound, high=bound, size=(input_dim, output_dim))

        elif weight_initializer == "he":
            self._weight = self.rng.normal(loc=0.0, scale=np.sqrt(2 / input_dim), size=(input_dim, output_dim))

        else:
            raise ValueError(f"*str* between 'he', 'std' and 'xavier' expected, got {weight_initializer} instead.")

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

                if optimizer == mgd:

                    if self.__additional_parameters.get("prev_weight") is None:
                        self.__additional_parameters["prev_weight"] = self._weight.copy()

                    self._weight, self.__additional_parameters["prev_weight"] = (
                        optimizer(gradient=gradient,
                                  weight=self._weight,
                                  prev_weight=self.__additional_parameters["prev_weight"],
                                  learning_rate=kwargs["learning_rate"],
                                  momentum=kwargs["momentum"]
                                  ))

                elif optimizer == dsg:

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
