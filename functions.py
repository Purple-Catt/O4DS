from layers import Layer
from network import ELM
from optimizers import *
import matplotlib.pyplot as plt


def l1_regularizer(x, l1: float):
    """Return the LASSO regularizer given a layer output."""
    penalty = l1 * np.sum(np.abs(x))

    return penalty


def mse(y_true, y_pred, weight, lasso: float = None):
    """Compute the Mean Square Error with an optional L1 regularization term, if given."""
    if lasso is not None:
        reg = l1_regularizer(weight, l1=lasso)
        loss = 0.5 * np.sum(np.add(np.power(y_true - y_pred, 2), reg))

    else:
        loss = 0.5 * np.sum(np.power(y_true - y_pred, 2))

    return loss


def gradient(y_true, y_pred, weight, inputs, lasso: float = 0.0):
    """Compute the (sub)gradient."""
    matr = np.matmul(inputs.T, (y_pred - y_true))
    grad = np.where(weight < 0, matr - lasso, matr + lasso)
    grad = np.where(weight == 0, matr, grad)

    return grad


def relu(x, prime: bool = False):
    """Compute the ReLU activation function. If *prime* is True, compute the first derivative."""
    if prime:
        return np.greater(x, 0).astype(int)

    else:
        return np.maximum(0, x)


def softplus(x, prime: bool = False):
    """Compute the Softplus activation function. If *prime* is True, compute the first derivative."""
    if prime:
        return 1 / (1 + np.exp(- x))

    else:
        return np.log(1 + np.exp(x))


def tanh(x, prime: bool = False):
    """Compute the tanh activation function. If *prime* is True, compute the first derivative."""
    if prime:
        return 1 - (np.tanh(x) ** 2)

    else:
        return np.tanh(x)


def sigmoid(x, prime: bool = False):
    """Compute the sigmoid activation function. If *prime* is True, compute the first derivative."""
    if prime:
        return (1 / (1 + np.exp(- x))) * (1 - (1 / (1 + np.exp(- x))))

    else:
        return 1 / (1 + np.exp(- x))


def splitting_function(x: np.array, y: np.array, train: float, val: float = 0.0):
    """Split the given numpy arrays into train, test and eventually validation sets."""
    n = len(x)
    x_train = x[:int(n * (train - val))]
    y_train = y[:int(n * (train - val))]
    x_test = x[int(n * train):]
    y_test = y[int(n * train):]
    if val != 0.0:
        x_val = x[int(n * train - val):int(n * train)]
        y_val = y[int(n * train - val):int(n * train)]
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test


def learning_curve(history: dict, save: bool = False, path: str = None):
    plt.plot(range(history["epochs"]), history["val_loss"], "b-", label="Validation loss")
    plt.plot(range(history["epochs"]), history["loss"], "r-", label="Train loss")
    loss = history["val_loss"][-1]
    plt.title(f"Learning curve-val_loss={round(loss, 5)}")
    plt.legend()
    if save:
        plt.savefig(path)
        plt.clf()

    else:
        plt.show()
        plt.clf()


def get_model(input_dim: int, mid_dim: int, output_dim: int):
    elm = ELM(
        layers=[
            Layer(name="hidden",
                  input_dim=input_dim,
                  output_dim=mid_dim,
                  weight_initializer="std",
                  activation=softplus,
                  trainable=False),
            Layer(name="output",
                  input_dim=mid_dim,
                  output_dim=output_dim,
                  weight_initializer="he",
                  trainable=True)
        ],
        loss=mse
    )

    return elm


def gridsearch(x_train: np.ndarray,
               y_train: np.ndarray,
               x_test: np.ndarray,
               y_test: np.ndarray,
               input_dim: int,
               output_dim: int,
               epochs: int = 100,
               patience: int = 5,
               gs_type: str = "architecture",
               **kwargs):
    """This function compute the gridsearch over the given model to find the optimal parameters. The specific kind of
    parameters to optimize can be decided.\n
    Parameters
        *x_train, y_train*: numpy arrays containing respectively the training set and the target values for it.\n
        *x_test, y_test*: numpy arrays containing respectively the test set and the target values for it.\n
        *input_dim*: the number of input units (neurons) of the layer\n
        *output_dim*: the number of _output units\n
        *epochs*: Default 100. Maximum number of epochs used to train the model.\n
        *patience*: Default 100. If a value is given, an earlystopping callback is added with the given parameter.\n
        *gs_type*: Default 'architecture'. String between 'architecture', 'lasso', 'mgd and 'dsg' that specifies the
        type of gridsearch to run. 'architecture' is used for finding the best number of units, 'lasso' for the l1
        penalty term, 'mgd' and 'dsg' respectively for the Momentum Gradient Descent and Deflected SubGradient
        algorithm optimal parameters' selection.\n
        *kwargs*: it's used to define all the necessary parameters for each type of gridsearch, in particular, for
        *architecture* the 'learning_rate', 'momentum' and 'lasso' are needed, for *lasso* 'mid_mid', 'learning_rate'
        and 'momentum' and for the algorithms *mgd* and *dsg* 'lasso' and 'mid_dim'.
        Considering the use of this function, the definition of ranges and stepsizes needs to be done directly here,
        modifying the values into the function, to keep the implementation as practical as possible.
    """
    evaluation = []

    if gs_type == "architecture":
        lasso = kwargs.get("lasso")
        lr = kwargs.get("learning_rate")
        mom = kwargs.get("momentum")

        multiplier = np.arange(start=2, stop=20, step=2)
        multiplier = [int(i) for i in list(multiplier)]

        n_comb = len(multiplier)
        print(f"{n_comb} parameters combinations needed to compute the gridsearch")

        for mul in multiplier:
            mid_dim = input_dim * mul
            model = get_model(input_dim=input_dim, mid_dim=mid_dim, output_dim=output_dim)

            history = model.fit(x_train, y_train,
                                x_test, y_test,
                                epochs=epochs,
                                optimizer=mgd,
                                lasso=lasso,
                                history=True,
                                patience=patience,
                                verbose=1,
                                learning_rate=lr,
                                momentum=mom)

            metrics = dict(multiplier=mul,
                           val_loss=history["val_loss"][-1],
                           train_loss=history["loss"][-1])
            evaluation.append(metrics)
            learning_curve(history, save=True, path=f"Plots\\Architecture\\{mul}.png")

            vl = history["val_loss"][-1]
            trl = history["loss"][-1]
            print(f"Tested model --> Multiplier={mul}, Val_loss={vl}, train_loss={trl}")

    elif gs_type == "lasso":
        mid_dim = kwargs.get("mid_dim")
        lr = kwargs.get("learning_rate")
        mom = kwargs.get("momentum")

        lmb = np.arange(start=0.0000001, stop=0.000001, step=0.0000001)
        lmb = [float(round(i, 8)) for i in list(lmb)]
        lmb = [0.0000001, 0.000001, 0.00001]

        n_comb = len(lmb)
        print(f"{n_comb} parameters combinations needed to compute the gridsearch")

        for lasso in lmb:
            model = get_model(input_dim=input_dim, mid_dim=mid_dim, output_dim=output_dim)

            history = model.fit(x_train, y_train,
                                x_test, y_test,
                                epochs=epochs,
                                optimizer=mgd,
                                lasso=lasso,
                                learning_rate=lr,
                                history=True,
                                patience=patience,
                                verbose=0,
                                momentum=mom)

            metrics = dict(lasso=lasso,
                           val_loss=history["val_loss"][-1],
                           train_loss=history["loss"][-1])
            evaluation.append(metrics)
            learning_curve(history, save=True, path=f"Plots\\LASSO\\{lasso}.png")

            vl = history["val_loss"][-1]
            trl = history["loss"][-1]
            print(f"Tested model --> Lasso={lasso}, Val_loss={vl}, train_loss={trl}")

    elif gs_type == "mgd":
        lasso = kwargs.get("lasso")
        mid_dim = kwargs.get("mid_dim")
        learning_rate = np.arange(start=0.000001, stop=0.00001, step=0.000002)
        learning_rate = [float(round(i, 8)) for i in list(learning_rate)]
        learning_rate = [0.00000005]

        momentum = np.arange(start=0.4, stop=1.0, step=0.1)
        momentum = [float(round(i, 1)) for i in list(momentum)]
        momentum = [0.2]

        n_comb = len(learning_rate) * len(momentum)
        print(f"{n_comb} parameters combination(s) needed to compute the gridsearch")

        for lr in learning_rate:
            for mom in momentum:
                model = get_model(input_dim=input_dim, mid_dim=mid_dim, output_dim=output_dim)

                history = model.fit(x_train, y_train,
                                    x_test=x_test, y_test=y_test,
                                    epochs=epochs,
                                    optimizer=mgd,
                                    lasso=lasso,
                                    learning_rate=lr,
                                    history=True,
                                    patience=patience,
                                    verbose=1,
                                    momentum=mom)

                metrics = dict(learning_rate=lr,
                               momentum=mom,
                               val_loss=history["val_loss"][-1],
                               train_loss=history["loss"][-1])
                evaluation.append(metrics)
                learning_curve(history, save=True, path=f"Plots\\MGD\\{lr}_{mom}.png")

                vl = history["val_loss"][-1]
                trl = history["loss"][-1]
                print(f"Tested model --> alpha={lr}, Beta={mom}, Val_loss={vl}, train_loss={trl}")

    elif gs_type == "dsg":
        lasso = kwargs.get("lasso")
        mid_dim = kwargs.get("mid_dim")

        beta = np.arange(start=0.1, stop=1.0, step=0.2)
        beta = [float(round(i, 1)) for i in list(beta)]

        rho = [0.85, 0.9, 0.95, 0.99]

        delta_0 = [0.001, 0.0001, 0.00001, 0.000001]

        R = [1]

        for b in beta:
            for r in rho:
                for d in delta_0:
                    for r_big in R:
                        model = get_model(input_dim=input_dim, mid_dim=mid_dim, output_dim=output_dim)

                        history = model.fit(x_train, y_train,
                                            x_test=x_test, y_test=y_test,
                                            epochs=epochs,
                                            optimizer=dsg,
                                            lasso=lasso,
                                            history=True,
                                            patience=patience,
                                            verbose=1,
                                            beta=b,
                                            rho=r,
                                            delta=d,
                                            R=r_big)

                        metrics = dict(beta=b,
                                       rho=r,
                                       delta=d,
                                       R=r_big,
                                       val_loss=history["val_loss"][-1],
                                       train_loss=history["loss"][-1])
                        evaluation.append(metrics)

                        vl = history["val_loss"][-1]
                        trl = history["loss"][-1]
                        print(f"Tested model --> beta={b}, rho={r}, delta0={d}, R={r_big} val_loss={vl}, "
                              f"train_loss={trl}")

    else:
        raise ValueError(f"String between 'architecture', 'lasso', 'mgd and 'dsg' expected, got {gs_type} instead.")

    print("Evaluating best model...")
    best = 10000
    for mod in evaluation:
        if mod["val_loss"] < best:
            best = mod["val_loss"]
            bestm = mod

    print(f"Best model: {bestm}")
