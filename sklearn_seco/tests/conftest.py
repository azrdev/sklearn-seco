"""pytest fixtures for the test cases in this directory."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state

from sklearn_seco.concrete import SimpleSeCoEstimator, CN2Estimator


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
    np.random.shuffle(samples)
    return X, y


@pytest.fixture
def binary_slight_overlap():
    """Generate two normal distributed, slightly overlapping classes.

    :return: tuple(X_train, y_train, X_test, y_test)
    """
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
    return X, y, X_test, y_test


@pytest.fixture
def binary_categorical():
    """Generate binary discrete problem with little noise.

    :return: tuple(X, y, X_test, y_test)
    """
    distribution = {
        'n_samples': 100,
        'n_features': 10,
        'centers': 2,
        'cluster_std': 2.5,
        'random_state': check_random_state(99)
    }
    X_numeric, y = make_blobs(**distribution)
    X = np.rint(X_numeric)
    X_test_numeric, y_test = make_blobs(**distribution)
    X_test = np.rint(X_test_numeric)
    return X, y, X_test, y_test
