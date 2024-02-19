import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from functions import splitting_function
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def get_matyas():
    """Return the Matyas function used for testing purposes."""
    x1 = np.arange(-10.0, 10.0, 0.02, dtype=np.float64)
    x2 = np.arange(-10.0, 10.0, 0.02, dtype=np.float64)
    y = 0.26 * ((x1 ** 2) + (x2 ** 2)) - 0.48 * x1 * x2
    y = y.reshape(1000, 1)
    arr = np.array([x1, x2], dtype=np.float64).T
    arr = arr.reshape(arr.shape[0], 1, 2)
    x_train, x_test, y_train, y_test = train_test_split(arr, y, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test


def get_mnist():
    """Return the MNIST dataset after some data manipulation to allow a better training of the model."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_cifar10():
    """Return the CIFAR10 dataset after some data manipulation to allow a better training of the model."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, 3072)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 3072)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_cifar100():
    """Return the CIFAR100 dataset after some data manipulation to allow a better training of the model."""
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
    x_train = x_train.reshape(x_train.shape[0], 1, 3072)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 3072)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_covertype():
    """Return the CoverType dataset after some data manipulation to allow a better training of the model."""
    scaler = MinMaxScaler()
    arr = np.genfromtxt("covtype.csv", delimiter=",")
    [x, y] = np.split(ary=arr, indices_or_sections=[-1], axis=1)
    x_train, y_train, x_test, y_test = splitting_function(x=x, y=y, train=0.8)
    x_train = scaler.fit_transform(x_train)
    x_train = x_train.reshape(x_train.shape[0], 1, 54)
    y_train = y_train - 1
    y_train = to_categorical(y_train)

    x_test = scaler.fit_transform(x_test)
    x_test = x_test.reshape(x_test.shape[0], 1, 54)
    y_test = y_test - 1
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_dim(name: str):
    if name == "mnist":
        return 784, 1568, 10
    elif name == "cifar10":
        return 3072, 6144, 10
    elif name == "cifar100":
        return 3072, 6144, 100
    elif name == "covertype":
        return 54, 108, 7
    elif name == "matyas":
        return 2, 4, 1
    else:
        raise ValueError(f"*str* between 'mnist', 'cifar10', 'cifar100', 'covertype', 'matyas' expected, "
                         f"got {name} instead.")
