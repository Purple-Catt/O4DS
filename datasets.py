import numpy as np
from numpy.random import RandomState
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from functions import splitting_function

mult = 2


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


def get_mnist(small: bool = True):
    """Return the MNIST dataset after some data manipulation to allow a better training of the model."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if small:
        x_train = x_train.reshape(x_train.shape[0], 784)
        y_train = y_train.reshape(y_train.shape[0], 1)
        x_test = x_test.reshape(x_test.shape[0], 784)
        y_test = y_test.reshape(y_test.shape[0], 1)

        data_train = np.hstack((x_train, y_train))
        data_test = np.hstack((x_test, y_test))

        rndst = RandomState(27)
        rndst.shuffle(data_train)
        rndst.shuffle(data_test)

        x_train = data_train[:2000, :-1]
        y_train = data_train[:2000, -1]
        x_test = data_test[:200, :-1]
        y_test = data_test[:200, -1]

    x_train = x_train.reshape(x_train.shape[0], 1, 784)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 784)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_cifar10(small: bool = True):
    """Return the CIFAR10 dataset after some data manipulation to allow a better training of the model."""
    allx = list()
    ally = list()

    def unpickle(file):
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='bytes')
        return dictionary

    files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    for i in files:
        data = unpickle(f"Datasets\\cifar10\\{i}")
        allx.append(data[b"data"])
        ally.append(data[b"labels"])

    x_train = np.concatenate(tuple(allx))
    y_train = np.concatenate(tuple(ally))
    test = unpickle("Datasets\\cifar10\\test_batch")
    x_test = np.array(test[b"data"])
    y_test = np.array(test[b"labels"])

    if small:
        x_train = x_train.reshape(x_train.shape[0], 3072)
        y_train = y_train.reshape(y_train.shape[0], 1)
        x_test = x_test.reshape(x_test.shape[0], 3072)
        y_test = y_test.reshape(y_test.shape[0], 1)

        data_train = np.hstack((x_train, y_train))
        data_test = np.hstack((x_test, y_test))

        rndst = RandomState(27)
        rndst.shuffle(data_train)
        rndst.shuffle(data_test)

        x_train = data_train[:2000, :-1]
        y_train = data_train[:2000, -1]
        x_test = data_test[:200, :-1]
        y_test = data_test[:200, -1]

    x_train = x_train.reshape(x_train.shape[0], 1, 3072)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 3072)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_cifar100(small: bool = True):
    """Return the CIFAR100 dataset after some data manipulation to allow a better training of the model."""
    allx = list()
    ally = list()

    def unpickle(file):
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='bytes')
        return dictionary

    files = ["data_batch_1", "data_batch_2"]
    for i in files:
        data = unpickle(f"Datasets\\cifar100\\{i}")
        allx.append(data[b"data"])
        ally.append(data[b"fine_labels"])

    x_train = np.concatenate(tuple(allx))
    y_train = np.concatenate(tuple(ally))
    test = unpickle("Datasets\\cifar100\\test_batch")
    x_test = np.array(test[b"data"])
    y_test = np.array(test[b"fine_labels"])

    if small:
        x_t = x_train.reshape(x_train.shape[0], 3072)
        y_t = y_train.reshape(y_train.shape[0], 1)
        x_te = x_test.reshape(x_test.shape[0], 3072)
        y_te = y_test.reshape(y_test.shape[0], 1)

        data_train = np.hstack((x_t, y_t))
        data_test = np.hstack((x_te, y_te))

        rndst = RandomState(27)
        rndst.shuffle(data_train)
        rndst.shuffle(data_test)

        x_train = data_train[:2000, :-1]
        y_train = data_train[:2000, -1]
        x_test = data_test[:200, :-1]
        y_test = data_test[:200, -1]

    x_train = x_train.reshape(x_train.shape[0], 1, 3072)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 3072)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_covertype(small: bool = True):
    """Return the CoverType dataset after some data manipulation to allow a better training of the model."""
    scaler = MinMaxScaler()
    arr = np.genfromtxt("Datasets\\covtype.csv", delimiter=",")
    [x, y] = np.split(ary=arr, indices_or_sections=[-1], axis=1)
    x_train, y_train, x_test, y_test = splitting_function(x=x, y=y, train=0.8)
    x_train = scaler.fit_transform(x_train)

    if small:
        x_t = x_train.reshape(x_train.shape[0], 54)
        y_t = y_train.reshape(y_train.shape[0], 1)
        x_te = x_test.reshape(x_test.shape[0], 54)
        y_te = y_test.reshape(y_test.shape[0], 1)

        data_train = np.hstack((x_t, y_t))
        data_test = np.hstack((x_te, y_te))

        rndst = RandomState(27)
        rndst.shuffle(data_train)
        rndst.shuffle(data_test)

        x_train = data_train[:2000, :-1]
        y_train = data_train[:2000, -1]
        x_test = data_test[:200, :-1]
        y_test = data_test[:200, -1]

    x_train = x_train.reshape(x_train.shape[0], 1, 54)
    y_train = y_train - 1
    y_train = to_categorical(y_train)

    x_test = scaler.fit_transform(x_test)
    x_test = x_test.reshape(x_test.shape[0], 1, 54)
    y_test = y_test - 1
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_dim(name: str, multiplier: int = mult):
    if name == "mnist":
        return 784, 784 * multiplier, 10
    elif name == "cifar10":
        return 3072, 3072 * multiplier, 10
    elif name == "cifar100":
        return 3072, 3072 * multiplier, 100
    elif name == "covertype":
        return 54, 54 * multiplier, 7
    elif name == "matyas":
        return 2, 2 * multiplier, 1
    else:
        raise ValueError(f"*str* between 'mnist', 'cifar10', 'cifar100', 'covertype', 'matyas' expected, "
                         f"got {name} instead.")
