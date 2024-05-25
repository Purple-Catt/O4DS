import time
import functions
from optimizers import *


class ELM:
    """This function builds an extreme learning machine with one hidden layer with a fixed _weight matrix
    and an _output layer with a _trainable _weight matrix."""
    def __init__(self, layers: list, loss):
        self.layers = layers
        self.loss = loss

    def predict(self, x):
        samples = len(x)
        result = []

        for i in range(samples):
            output = x[i]
            for layer in self.layers:
                output = layer(output)
            result.append(output)

        return np.array(result, dtype=np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, optimizer, lasso,
            x_test: np.ndarray = None, y_test: np.ndarray = None,
            history: bool = True, patience: int = None,
            loss_evaluation: str = "val", verbose: int = 1,
            **kwargs):
        """This is the main method of this class, it computes the entire training process with an *online learning*
        method and validate it on the test set.\n
        Despite other high-level libraries, such as Keras or Pytorch, implement a validation spit method so has to have
        training, validation and test sets, it has been decided not to do that here because the nature of the project
        doesn't require it, so it's a better choice to keep as many observations as possible for the training set.
        Parameters:
            x: numpy array containing the training set
            y: numpy array containing the target values of the training set
            x_test: (optional) numpy array containing the test set
            y_test: (optional) numpy array containing the target values of the test set
            epochs: maximum number of epochs used to train the model
            optimizer: optimizer function used to minimize the loss in the training process
            lasso: value of lambda for the LASSO regularizer from the range (0.0, 1.0)
            history: if True a history dictionary containing all the (training and validation) loss values and epochs is
                returned
            patience: Default *None*. If a value is given, an earlystopping callback is added with the given parameter
            loss_evaluation: wether counting patience on validation or training loss, string between 'val' and 'train'
            verbose: Integer between 0, 1, 2 where 0 is silence, 1 print losses after each epoch, 2 print losses after
                each iteration
        """
        params = kwargs
        params["lasso"] = lasso
        err_tr_list = []
        err_val_list = []
        count = 0
        act_epoch = 0
        prev_error = np.inf
        samples = len(x)
        iteration = 0

        for epoch in range(epochs):
            start_time = time.time()
            act_epoch += 1
            err_tr = 0.0

            for sample in range(samples):
                # Forward propagation
                output = x[sample]

                for layer in self.layers:
                    output = layer(output)

                act_err = self.loss(y_true=y[sample],
                                    y_pred=output,
                                    weight=self.layers[-1].weight,
                                    lasso=lasso)
                err_tr += act_err

                grad = functions.gradient(y_true=y[sample],
                                          y_pred=output,
                                          weight=self.layers[-1].weight,
                                          inputs=self.layers[-1].inputs,
                                          lasso=lasso)

                params["fx"] = act_err
                self.layers[-1].weights_update(gradient=grad,
                                               optimizer=optimizer,
                                               **params)

                iteration += 1
                if verbose > 1:
                    print("iter=%d   error=%f" % (iteration, act_err))

            if x_test is not None:
                # Forward propagation for validation set
                err_val = 0
                val_samples = len(x_test)
                for val_sample in range(val_samples):
                    output = x_test[val_sample]

                    for layer in self.layers:
                        output = layer(output)

                    err_val += self.loss(y_true=y_test[val_sample],
                                         y_pred=output,
                                         weight=self.layers[-1].weight,
                                         lasso=lasso)

                err_val /= val_samples
                err_val_list.append(err_val)

            err_tr /= samples
            err_tr_list.append(err_tr)

            if x_test is not None:
                if verbose > 0:
                    print("epoch %d/%d   train_error=%.10f   val_error=%f   time: %.3f s" % (epoch + 1, epochs, err_tr,
                                                                                             err_val,
                                                                                             time.time() - start_time))

            else:
                if verbose > 0:
                    print("epoch %d/%d   train_error=%.10f   time: %.3f s" % (epoch + 1, epochs, err_tr,
                                                                              time.time() - start_time))

            # Early stopping
            if isinstance(patience, int):
                # Wether refer to the training of validation error to count the patience
                if loss_evaluation == "val":
                    patience_error = err_val

                elif loss_evaluation == "train":
                    patience_error = err_tr

                else:
                    raise ValueError(f"str between 'val' and 'train' expected, got {loss_evaluation} instead.")

                if patience_error >= prev_error:
                    count += 1

                if prev_error - patience_error < 0.000000001:
                    count += 1

                if count == patience:
                    print(f"EarlyStopping intervened on epoch {epoch}")
                    print(f"Learning rate decreased from {params['learning_rate']} to {params['learning_rate'] / 5}")
                    params["learning_rate"] /= 5
                    if input("Continue? [Y/n]").lower() == "n":
                        break
                    count = 0
                prev_error = patience_error

        if history:
            return dict(
                val_loss=err_val_list,
                loss=err_tr_list,
                epochs=act_epoch
            )

    @property
    def get_weight(self):
        """Utility for saving the weight matrices for reproducibility purposes."""
        return self.layers[-1].weight
