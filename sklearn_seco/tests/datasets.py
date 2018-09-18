import itertools
from collections.__init__ import namedtuple
from typing import Union

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state


def Dataset(x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray = None,
            y_test: np.ndarray = None,
            categorical_features: Union[None, str, np.ndarray] = None):
    _Dataset = namedtuple('Dataset', ['x_train', 'y_train', 'x_test', 'y_test',
                                      'categorical_features'])
    return _Dataset(x_train, y_train, x_test, y_test, categorical_features)


# datasets


def binary_categorical():
    """Generate binary discrete problem with little noise.
    """
    dim = 8
    n = 100
    X_numeric, y8 = make_blobs(n_samples=2 * n, n_features=dim, centers=dim,
                               random_state=7)
    X = np.rint(X_numeric)
    y = y8 % 2 + 10
    return Dataset(X[:n], y[:n], X[n:], y[n:], 'all')


def binary_mixed():
    """
    Generate medium-size binary problem with categorical and numeric features.
    """
    dim = 5
    n = 128
    categorical = np.array([True, False, True, True, False])
    xnum, ymc = make_blobs(n_samples=2 * n, n_features=dim, centers=dim,
                           random_state=2)
    x = np.where(categorical, np.rint(xnum), xnum)
    y = ymc % 2 + 20
    return Dataset(x[:n], y[:n], x[n:], y[n:], categorical)


def xor_2d():
    """Generate numeric low-noise 2D binary XOR problem"""
    rnd = check_random_state(11)
    n = 100
    centers = itertools.product([0, 4], [0, 4])
    t = np.vstack(np.hstack((rnd.normal(loc=(x, y), size=(n, 2)),
                             [[99 + (x == y)]] * n))
                  for x, y in centers)
    rnd.shuffle(t)
    split = len(t) // 2
    x_train = t[:split, :-1]
    y_train = t[:split, -1]
    x_test = t[split:, :-1]
    y_test = t[split:, -1]
    return Dataset(x_train, y_train, x_test, y_test)


def checkerboard_2d(binary=True, categorical=True):
    """
    Generate huge low-noise 2D multi-class problem with interleaved class clusters (similar to XOR)

    :param binary: If True, data has 2 classes, otherwise 9.
    :param categorical: If True, features are few distinct integers, otherwise
        a float distribution.
    """
    rnd = check_random_state(1)
    n = 99999
    centers = itertools.product([1, 5, 9], [4, -1, 8])
    t = np.vstack(
        np.hstack((rnd.normal(loc=(x, y), size=(n, 2)), [[x * y]] * n))
        for x, y in centers)
    rnd.shuffle(t)
    split = len(t) // 2
    x_train = t[:split, :-1]
    y_train = t[:split, -1]
    x_test = t[split:, :-1]
    y_test = t[split:, -1]
    cm = None
    if binary:
        y_train %= 2
        y_test %= 2
    if categorical:
        x_train = np.rint(x_train)
        x_test = np.rint(x_test)
        cm = 'all'
    return Dataset(x_train, y_train, x_test, y_test, cm)


def binary_slight_overlap():
    """Generate two normal distributed, slightly overlapping classes."""
    random = check_random_state(42)
    dim = 8
    n_samples = 160  # we have 80 samples for each class for each (train, test)
    mean = np.zeros(dim)
    cov = np.eye(dim)
    raw = np.block([
        [random.multivariate_normal(mean, cov, size=n_samples),  # features
         np.ones((n_samples, 1))],  # positive class label
        [random.multivariate_normal(mean + 3, cov, size=n_samples),  # features
         np.zeros((n_samples, 1))]  # negative class label
    ])
    random.shuffle(raw)
    train, test = np.array_split(raw, 2, axis=0)
    X = train[:, :-1]  # all but last column
    y = train[:, -1]  # last column
    X_test = test[:, :-1]
    y_test = test[:, -1]
    return Dataset(X, y, X_test, y_test)
