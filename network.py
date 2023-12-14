import numpy as np
import time
import functions


class ELM:
    """This function builds an extreme learning machine with one hidden layer with a fixed weight matrix
    and an output layer with a trainable weight matrix."""
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

    def fit(self, x: np.array, y: np.array, x_test: np.array, y_test: np.array,
            epochs: int, optimizer, lasso, learning_rate, **kwargs):
        """This is the main method of this class, it computes the entire training process with an *online learning*
        method and validate it on the test set.\n
        Despite other high-level libraries, such as Keras or Pytorch, implement a validation spit method so has to have
        training, validation and test sets, it has been decided not to do that here because the nature of the project
        doesn't require it, so it's a better choice to keep as many observations as possible for the training set."""
        samples = len(x)
        for epoch in range(epochs):
            start_time = time.time()
            err_tr = 0
            for sample in range(samples):
                # Forward propagation
                output = x[sample]
                for layer in self.layers:
                    output = layer.call(output)

                err_tr += self.loss(y_true=y[sample],
                                    y_pred=output,
                                    weight=self.layers[-1].weight,
                                    lasso=lasso)

                grad = functions.gradient(y_true=y[sample],
                                          y_pred=output,
                                          weight=self.layers[-1].weight,
                                          inputs=self.layers[-1].inputs,
                                          lasso=lasso)

                self.layers[-1].weights_update(gradient=grad,
                                               optimizer=optimizer,
                                               learning_rate=learning_rate,
                                               **kwargs)

            # Forward propagation for validation set
            err_val = 0
            val_samples = len(x_test)
            for val_sample in range(val_samples):
                output = x_test[val_sample]

                for layer in self.layers:
                    output = layer.call(output)

                err_val += self.loss(y_true=y_test[val_sample],
                                     y_pred=output,
                                     weight=self.layers[-1].weight,
                                     lasso=lasso)

            err_val /= val_samples

            err_tr /= samples
            print("epoch %d/%d   train_error=%f   val_error=%f   time: %.3f s" % (epoch + 1, epochs, err_tr,
                                                                                  err_val, time.time()-start_time))
