"""Tests for `sklearn_seco.concrete`."""

import numpy as np
from numpy import NINF, PINF
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import check_estimator

from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.common import UPPER
from sklearn_seco.concrete import SimpleSeCoImplementation
from .conftest import count_conditions
from .datasets import perfectly_correlated_multiclass


def test_base_trivial(record_theory):
    """Test SimpleSeCo with a trivial test set of 2 instances."""
    categorical_mask = np.array([True, False])
    X_train = np.array([[100, 0.0],
                        [111, 1.0]])
    y_train = np.array([1, 2])
    est = _BinarySeCoEstimator(SimpleSeCoImplementation(), categorical_mask)
    est.fit(X_train, y_train)
    record_theory(est.theory_)

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


def test_base_easyrules(record_theory):
    """Test SimpleSeCo with a linearly separable, 4 instance binary test set.

    Compare `docs/test_base_easyrules_featurespace.png`.
    """
    categorical_mask = np.array([True, False])
    X_train = np.array([[0, -1.0],
                        [0, -2.0],
                        [0,  1.0],
                        [1, -1.0]])
    y_train = np.array([1, 1, 2, 2])
    est = _BinarySeCoEstimator(SimpleSeCoImplementation(), categorical_mask)
    est.fit(X_train, y_train)
    record_theory(est.theory_)

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


def test_trivial_decision_border(seco_estimator, trivial_decision_border,
                                 record_theory):
    """Check recognition of the linear border in `trivial_decision_border`."""
    seco_estimator.fit(trivial_decision_border.x_train,
                       trivial_decision_border.y_train)
    # check recognition of binary problem
    base = seco_estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    record_theory(base.theory_)
    assert base.target_class_ == 0
    # check expected rule
    assert len(base.theory_) == 1
    assert_array_almost_equal(base.theory_[0], [[NINF, NINF], [PINF, 1.0]],
                              decimal=1)


def test_perfectly_correlated_categories_multiclass(seco_estimator,
                                                    record_theory):
    """Expect perfect rules on `perfectly_correlated_multiclass` problem."""
    dataset = perfectly_correlated_multiclass()
    seco_estimator.fit(dataset.x_train, dataset.y_train,
                       categorical_features=dataset.categorical_features)
    bases = assert_multiclass_problem(seco_estimator)
    record_theory([b.theory_ for b in bases])
    # check rules
    for base in bases:
        assert len(base.theory_) == 1
        assert count_conditions(base.theory_) == 1
        assert count_conditions(base.theory_[:, UPPER]) == 0
    assert_array_equal(dataset.y_train,
                       seco_estimator.predict(dataset.x_train))


def test_sklearn_check_estimator(seco_estimator_class):
    """Run check_estimator from `sklearn.utils.estimator_checks`.

    # TODO: Unwrap :func:`sklearn.utils.estimator_checks.check_estimator`, so
    our report shows which ones actually failed. Waiting for <https://github.com/scikit-learn/scikit-learn/issues/11622>
    """
    check_estimator(seco_estimator_class)


def test_blackbox_accuracy(seco_estimator, blackbox_test, record_theory):
    """Expect high accuracy from each of our blackbox test cases"""
    x_train, y_train, x_test, y_test, cm = blackbox_test
    seco_estimator.fit(x_train, y_train, categorical_features=cm)

    is_binary = len(np.unique(y_train)) == 2
    if is_binary:
        base = assert_binary_problem(seco_estimator)
        record_theory(base.theory_)
        print("{} rules:\n{}".format(len(base.theory_), base.theory_))
    else:
        bases = assert_multiclass_problem(seco_estimator)
        record_theory([b.theory_ for b in bases])
        print("{} theories:\n".format(len(bases)))
        for base_ in bases:
            print("{} rules:\n{}\n".format(len(base_.theory_), base_.theory_))

    assert_prediction_performance(seco_estimator,
                                  x_train, y_train, x_test, y_test)


# helpers

def assert_binary_problem(estimator):
    """Check recognition of binary problem by `estimator`.

    :return: the "base_estimator_" `_BinarySeCoEstimator` instance
    """
    base = estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    assert len(base.theory_)
    assert count_conditions(
        base.theory_[:, UPPER, base.categorical_mask_]) == 0
    return base


def assert_multiclass_problem(estimator):
    """Check recognition of multi-class problem.

    :return: the list of "base_estimator_" `_BinarySeCoEstimator` instances
    """
    assert not isinstance(estimator.base_estimator_, _BinarySeCoEstimator)
    bases = estimator.base_estimator_.estimators_
    for base_ in bases:
        assert isinstance(base_, _BinarySeCoEstimator)
        assert len(base_.theory_)
        assert count_conditions(
            base_.theory_[:, UPPER, base_.categorical_mask_]) == 0
    return bases


def assert_prediction_performance(estimator, x_train, y_train, x_test, y_test):
    # check accuracy,precision on training data
    y_predicted_train = estimator.predict(x_train)
    assert accuracy_score(y_train, y_predicted_train) > 0.9
    if x_test is not None:
        # check accuracy on test data
        y_predicted = estimator.predict(x_test)
        assert accuracy_score(y_test, y_predicted) > 0.8

        from sklearn.metrics import classification_report, confusion_matrix
        print()
        print(confusion_matrix(y_test, y_predicted))
        print(classification_report(y_test, y_predicted))
