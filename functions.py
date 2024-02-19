from layers import Layer
from network import ELM
from optimizers import *
import matplotlib.pyplot as plt


def l1_regularizer(x, l1: float):
    """Return the LASSO regularizer given a layer _output."""
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


def gradient(y_true, y_pred, weight, inputs, lasso: float = 0.0, subgradient: bool = False):
    matr = np.matmul(inputs.T, (y_pred - y_true))
    grad = np.where(weight < 0, matr - lasso, matr + lasso)
    grad = np.where(weight == 0, matr, grad)

    return grad


def relu(x, prime: bool = False):
    """Compute the ReLU _activation function. If *prime* is True, compute the first derivative."""
    if prime:
        return np.greater(x, 0).astype(int)

    else:
        return np.maximum(0, x)


def softplus(x, prime: bool = False):
    """Compute the Softplus _activation function. If *prime* is True, compute the first derivative."""
    if prime:
        return 1 / (1 + np.exp(- x))

    else:
        return np.log(1 + np.exp(x))


def tanh(x, prime: bool = False):
    """Compute the tanh _activation function. If *prime* is True, compute the first derivative."""
    if prime:
        return 1 - (np.tanh(x) ** 2)

    else:
        return np.tanh(x)


def sigmoid(x, prime: bool = False):
    """Compute the sigmoid _activation function. If *prime* is True, compute the first derivative."""
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
                  activation=tanh,
                  trainable=False),
            Layer(name="_output",
                  input_dim=mid_dim,
                  output_dim=output_dim,
                  weight_initializer="xavier",
                  trainable=True)
        ],
        loss=mse
    )

    return elm


def gridsearch_mgd(x_train: np.ndarray,
                   y_train: np.ndarray,
                   x_test: np.ndarray,
                   y_test: np.ndarray,
                   input_dim: int,
                   mid_dim: int,
                   output_dim: int,
                   epochs: int = 50,
                   patience: int = 5):
    evaluation = []
    learning_rate = np.arange(start=0.00001, stop=0.00002, step=0.00005)
    learning_rate = [float(round(i, 6)) for i in list(learning_rate)]

    momentum = np.arange(start=0.8, stop=1.0, step=0.1)
    momentum = [float(round(i, 1)) for i in list(momentum)]
    momentum = [0.8]

    lmb = np.arange(start=0.000001, stop=0.000003, step=0.000002)
    lmb = [float(round(i, 6)) for i in list(lmb)]

    for lr in learning_rate:
        for mom in momentum:
            for lasso in lmb:
                model = get_model(input_dim=input_dim, mid_dim=mid_dim, output_dim=output_dim)

                history = model.fit(x_train, y_train,
                                    x_test, y_test,
                                    epochs=epochs,
                                    optimizer=MGD,
                                    lasso=lasso,
                                    learning_rate=lr,
                                    history=True,
                                    patience=patience,
                                    verbose=1,
                                    momentum=mom)

                metrics = dict(learning_rate=lr,
                               momentum=mom,
                               lmb=lasso,
                               val_loss=history["val_loss"][-1],
                               train_loss=history["loss"][-1])
                evaluation.append(metrics)
                learning_curve(history, save=False, path=f"Plots\\MGD\\{lr}_{mom}_{lasso}.png")

                vl = history["val_loss"][-1]
                trl = history["loss"][-1]
                print(f"Tested model --> alpha={lr}, Beta={mom}, L1={lasso}, Val_loss={vl}, train_loss={trl}")

    print("Evaluating best model...")
    best = 10000
    for mod in evaluation:
        if mod["val_loss"] < best:
            best = mod["val_loss"]
            bestm = mod

    print(f"Best model: {bestm}")
