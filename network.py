import numpy as np


class ELM:

    def __init__(self, layers: list, loss):
        self.layers = layers
        self.loss = loss

    def predict(self, x):
        samples = len(x)
        result = []

        for i in range(samples):
            output = x[i]
            for layer in self.layers:
                output = layer.call(output)
            result.append(output)

        return np.array(result, dtype=np.float64)

    def fit(self, x, y, epochs: int, optimizer, lasso, learning_rate, **kwargs):
        samples = len(x)
        for epoch in range(epochs):
            err = 0
            for sample in range(samples):
                # Forward propagation
                output = x[sample]
                for layer in self.layers:
                    output = layer.call(output)

                err += self.loss(y_true=y[sample], y_pred=output, lasso=lasso)

                # Backward propagation
                error = self.loss(y_true=y[sample], y_pred=output, prime=True)
                for layer in reversed(self.layers):
                    if layer.trainable:
                        error = layer.weights_update(loss=error, optimizer=optimizer, learning_rate=learning_rate,
                                                     **kwargs)

            err /= samples
            print("epoch %d/%d   error=%f" % (epoch + 1, epochs, err))
