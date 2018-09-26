import itertools
from collections import namedtuple
from typing import Union

import numpy as np
from sklearn.datasets import make_blobs, make_classification
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


def binary_categorical(n_features=8, n_samples=100, random=7):
    """Generate binary discrete problem with little noise.
    """
    X_numeric, y8 = make_blobs(n_samples=2 * n_samples, n_features=n_features,
                               centers=n_features, random_state=random)
    X = np.rint(X_numeric)
    y = y8 % 2 + 10
    return Dataset(X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:],
                   'all')


def binary_mixed(n_samples=128, n_features=5, random=2):
    """
    Generate medium-size binary problem with categorical and numeric features.
    """
    categorical = np.array([True, False] * n_features)[:n_features]  # T,F,T,…
    xnum, ymc = make_blobs(n_samples=2 * n_samples, n_features=n_features,
                           centers=n_features, random_state=random)
    x = np.where(categorical, np.rint(xnum), xnum)
    y = ymc % 2 + 20
    return Dataset(x[:n_samples], y[:n_samples],
                   x[n_samples:], y[n_samples:],
                   categorical)


def xor_2d(n_samples=400, random=None):
    """Generate numeric low-noise 2D binary XOR problem"""
    if random is None:
        random = check_random_state(11)
    n = n_samples // 4
    centers = itertools.product([0, 4], [0, 4])
    t = np.vstack(np.hstack((random.normal(loc=(x, y), size=(n, 2)),
                             [[99 + (x == y)]] * n))
                  for x, y in centers)
    random.shuffle(t)
    split = len(t) // 2
    x_train = t[:split, :-1]
    y_train = t[:split, -1]
    x_test = t[split:, :-1]
    y_test = t[split:, -1]
    return Dataset(x_train, y_train, x_test, y_test)


def checkerboard_2d(n_samples=10**5, binary=True, categorical=True,
                    random=None):
    """
    Generate huge 2D multi-class problem with interleaved class clusters (similar to XOR)

    :param binary: If True, data has 2 classes, otherwise 9.
    :param categorical: If True, features are few distinct integers, otherwise
        a float distribution.
    """
    if random is None:
        random = check_random_state(1)
    centers = itertools.product([0, 7, 15], [20, 30, 39])
    n = n_samples
    t = np.vstack(
        np.hstack((random.normal(loc=(x, y), size=(n_samples, 2)), [[cls]] * n_samples))
        for cls, (x, y) in enumerate(centers))
    random.shuffle(t)
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


def binary_slight_overlap(
        n_samples=160,  # 80 samples for each class and each of (train, test)
        n_features=8,
        random=None):
    """Generate two normal distributed, slightly overlapping classes."""
    if random is None:
        random = check_random_state(42)
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
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


def sklearn_make_classification(n_samples=100, n_features=2,
                                random=1,
                                *, categorize=False, **kwargs):
    n_informative = max(2, n_features // 3 * 2)
    x, y = make_classification(n_samples, n_features,
                               n_informative=n_informative,
                               n_redundant=0,
                               class_sep=1.5,
                               random_state=random,
                               **kwargs)
    cf = None
    if categorize:
        x = np.rint(x)
        cf = 'all'
    return Dataset(x, y, categorical_features=cf)
