"""Tests for `sklearn_seco.concrete`."""

import numpy as np
import pytest
from numpy import NINF, PINF
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import check_estimator
from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.concrete import \
    SimpleSeCoImplementation, SimpleSeCoEstimator, CN2Estimator


def test_base_trivial():
    """Test SimpleSeCo with a trivial test set of 2 instances."""
    categorical_mask = np.array([True, False])
    X_train = np.array([[100, 0.0],
                        [111, 1.0]])
    y_train = np.array([1, 2])
    est = _BinarySeCoEstimator(SimpleSeCoImplementation())
    est.fit(X_train, y_train, categorical_mask)

    assert est.target_class_ == 1
    assert len(est.theory_) == 1
    # first refinement wins (tie breaking)
    assert_array_equal(est.theory_[0], [[100, NINF], [PINF, PINF]])

    assert_array_equal(est.predict(X_train), y_train)

    X_test = np.array([[100, 14],
                       [111, -15],
                       [100, -16],
                       [100, 0],
                       [111, 1]
                       ])
    y_test = np.array([1, 2, 1, 1, 2])
    assert_array_equal(est.predict(X_test), y_test)


# FIXME: broken since inner_stopping_criterion = (n == 0), commit 2d6f261
def test_base_easyrules():
    """Test SimpleSeCo with a linearly separable, 4 instance binary test set.
    """
    categorical_mask = np.array([True, False])
    X_train = np.array([[0, -1.0],
                        [0, -2.0],
                        [0,  1.0],
                        [1, -1.0]])
    y_train = np.array([1, 1, 2, 2])
    est = _BinarySeCoEstimator(SimpleSeCoImplementation())
    est.fit(X_train, y_train, categorical_mask)

    assert est.target_class_ == 1
    assert len(est.theory_) == 2
    assert_array_equal(est.theory_[0], np.array([[NINF, NINF], [PINF, -1.5]]))
    assert_array_equal(est.theory_[1], np.array([[   0, NINF], [PINF,    0]]))

    assert_array_equal(est.predict(X_train), y_train)

    X_test = np.array([[0, 14],
                       [1, -15],
                       [0, -16],
                       [0, 0],
                       [1, 1]
                       ])
    y_test = np.array([2, 1, 1, 1, 2])
    assert_array_equal(est.predict(X_test), y_test)


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


def test_trivial_decision_border(seco_estimator):
    """
    Check recognition of the linear border between to normal distributed classes.
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

    seco_estimator.fit(X, y)
    # check recognition of binary problem
    base = seco_estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    assert base.target_class_ == 0
    # check expected rule
    assert len(base.theory_) == 1
    assert_array_almost_equal(base.theory_[0], [[NINF, NINF], [PINF, 1.0]],
                              decimal=1)


def test_blackbox_accuracy_binary(seco_estimator):
    """
    Generate two normal distributed, slightly overlapping classes and expect high accuracy_score.
    """
    random = check_random_state(42)
    dim = 8
    n_cls = 80
    cov = np.eye(dim)
    training = np.block([
        [random.multivariate_normal(np.zeros(dim), cov, size=n_cls),
         np.ones((n_cls, 1))],
        [random.multivariate_normal(np.zeros(dim) + 3, cov, size=n_cls),
         np.zeros((n_cls, 1))]
    ])
    X = training[:, :-1]  # all but last column
    y = training[:, -1]  # last column
    np.random.shuffle(training)
    testing = np.block([
        [random.multivariate_normal(np.zeros(dim), cov, size=n_cls),
         np.ones((n_cls, 1))],
        [random.multivariate_normal(np.zeros(dim) + 3, cov, size=n_cls),
         np.zeros((n_cls, 1))]
    ])
    X_testing = testing[:, :-1]  # all but last column
    y_testing = testing[:, -1]  # last column

    seco_estimator.fit(X, y)
    # check recognition of binary problem
    base = seco_estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    assert base.target_class_ == 0
    # check accuracy
    y_predicted = seco_estimator.predict(X_testing)
    assert accuracy_score(y_testing, y_predicted) > 0.8


def test_sklearn_check_estimator(seco_estimator_class):
    """Run check_estimator from `sklearn.utils.estimator_checks`.

    # TODO: Unwrap :func:`sklearn.utils.estimator_checks.check_estimator`, so
    our report shows which ones actually failed. Waiting for <https://github.com/scikit-learn/scikit-learn/issues/11622>
    """
    check_estimator(seco_estimator_class)
