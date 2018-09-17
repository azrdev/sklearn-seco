"""pytest fixtures for the test cases in this directory."""
import itertools
from collections import namedtuple
from typing import Union, List

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state

from sklearn_seco.concrete import SimpleSeCoEstimator, CN2Estimator


# pytest plugin, to print theory on test failure
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    default = yield
    report = default.get_result()
    if report.failed and report.user_properties:
        for name, prop in report.user_properties:
            if name == 'theory':
                report.longrepr.addsection(name, str(prop))
                break
    return default


@pytest.fixture
def record_theory(record_property):
    def _record(theory: Union[np.ndarray, List[np.ndarray]]):
        record_property("theory", theory)
    return _record


# helpers


def count_conditions(theory):
    """
    :return: the number of conditions (i.e. non-infinite bounds) of all rules
      in `theory`.
    """
    return np.count_nonzero(np.isfinite(theory))


_Dataset = namedtuple('Dataset', ['x_train', 'y_train', 'x_test', 'y_test', 'categorical_features'])
def Dataset(x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray = None,
            y_test: np.ndarray = None,
            categorical_features: Union[None, str, np.ndarray] = None):
    return _Dataset(x_train, y_train, x_test, y_test, categorical_features)


# fixtures


@pytest.fixture(params=[SimpleSeCoEstimator, CN2Estimator])
def seco_estimator_class(request):
    """Fixture running for each of the pre-defined estimator classes from
    `sklearn_seco.concrete`.

    :return: An estimator class.
    """
    return request.param


@pytest.fixture
def seco_estimator(seco_estimator_class):
    """Fixture running for each of the pre-defined estimators from
    `sklearn_seco.concrete`.

    :return: An estimator instance.
    """
    return seco_estimator_class()


@pytest.fixture
def trivial_decision_border():
    """Generate normal distributed, linearly separated binary problem.

    :return: tuple(X, y)
    """
    random = check_random_state(42)
    samples = np.array([random.normal(size=50),
                        random.random_sample(50),
                        np.zeros(50)]).T
    X = samples[:, :-1]  # all but last column
    y = samples[:, -1]  # last column
    samples[0:25, 1] += 1
    y[0:25] = 1
    random.shuffle(samples)
    return Dataset(X, y)


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def checkerboard_2d():
    """
    Generate huge low-noise 2D multi-class problem with interleaved class clusters (similar to XOR)
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
    return Dataset(x_train, y_train, x_test, y_test)


@pytest.fixture
def checkerboard_2d_binary(checkerboard_2d):
    """Generate a binary (2 class) variant of `checkerboard_2d`."""
    x_train, y_train, x_test, y_test, cm = checkerboard_2d
    return Dataset(x_train, y_train % 2, x_test, y_test % 2)


@pytest.fixture
def checkerboard_2d_binary_categorical(checkerboard_2d_binary):
    """Generate a categorical, binary (2 class) variant of `checkerboard_2d`.
    """
    x_train, y_train, x_test, y_test, cm = checkerboard_2d_binary
    return Dataset(np.rint(x_train), y_train, np.rint(x_test), y_test, 'all')


@pytest.fixture
def perfectly_correlated_multiclass():
    """Generate 10-class problem with 10 features each matching one class."""
    n = 10
    y = np.arange(1, n + 1)
    x = np.eye(n, dtype=int) * y
    return Dataset(x, y, categorical_features='all')

