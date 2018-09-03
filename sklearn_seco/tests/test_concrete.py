"""Tests for `sklearn_seco.concrete`."""

import numpy as np
from numpy import NINF, PINF
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import check_estimator
from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.concrete import SimpleSeCoImplementation


def count_conditions(theory):
    """
    :return: the number of conditions (i.e. non-infinite bounds) of all rules
      in `theory`.
    """
    return np.count_nonzero(np.isfinite(theory))


def test_base_trivial(record_property):
    """Test SimpleSeCo with a trivial test set of 2 instances."""
    categorical_mask = np.array([True, False])
    X_train = np.array([[100, 0.0],
                        [111, 1.0]])
    y_train = np.array([1, 2])
    est = _BinarySeCoEstimator(SimpleSeCoImplementation(), categorical_mask)
    est.fit(X_train, y_train)
    record_property("theory", est.theory_)

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
def test_base_easyrules(record_property):
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
    record_property("theory", est.theory_)

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


def test_trivial_decision_border(seco_estimator,
                                 trivial_decision_border,
                                 record_property):
    """Check recognition of the linear border in `trivial_decision_border`."""
    seco_estimator.fit(*trivial_decision_border)
    # check recognition of binary problem
    base = seco_estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    record_property("theory", base.theory_)
    assert base.target_class_ == 0
    # check expected rule
    assert len(base.theory_) == 1
    assert_array_almost_equal(base.theory_[0], [[NINF, NINF], [PINF, 1.0]],
                              decimal=1)


def test_blackbox_accuracy_binary(seco_estimator,
                                  binary_slight_overlap,
                                  record_property):
    """Expect high accuracy_score on `binary_slight_overlap`."""
    X, y, X_test, y_test = binary_slight_overlap
    seco_estimator.fit(X, y)
    # check recognition of binary problem
    base = seco_estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    record_property("theory", base.theory_)
    assert base.target_class_ == 0
    # check accuracy
    y_predicted = seco_estimator.predict(X_test)
    assert accuracy_score(y_test, y_predicted) > 0.8


def test_perfectly_correlated_categories_multiclass(
        seco_estimator, perfectly_correlated_multiclass, record_property):
    """Expect perfect rules on multiclass problem with (feature_i == class_i).
    """
    x, y = perfectly_correlated_multiclass
    # check recognition of multiclass problem
    seco_estimator.fit(x, y, categorical_features='all')
    assert not isinstance(seco_estimator.base_estimator_, _BinarySeCoEstimator)
    bases = seco_estimator.base_estimator_.estimators_
    for base in bases:
        assert isinstance(base, _BinarySeCoEstimator)
    record_property("theory", [b.theory_ for b in bases])
    # check rules
    for base in bases:
        theory = base.theory_
        assert len(theory) == 1
        assert count_conditions(theory) == 1
    assert_array_equal(y, seco_estimator.predict(x))


def test_blackbox_accuracy_binary_categorical(seco_estimator,
                                              binary_categorical,
                                              record_property):
    """Expect high accuracy on `binary_categorical`."""
    X, y, X_test, y_test = binary_categorical
    seco_estimator.fit(X, y, categorical_features='all')
    # check recognition of binary problem
    base = seco_estimator.base_estimator_
    record_property("theory", base.theory_)
    assert isinstance(base, _BinarySeCoEstimator)
    # check accuracy
    assert_array_equal(seco_estimator.predict(X), y)
    y_predicted = seco_estimator.predict(X_test)
    assert accuracy_score(y_test, y_predicted) > 0.8


def test_sklearn_check_estimator(seco_estimator_class):
    """Run check_estimator from `sklearn.utils.estimator_checks`.

    # TODO: Unwrap :func:`sklearn.utils.estimator_checks.check_estimator`, so
    our report shows which ones actually failed. Waiting for <https://github.com/scikit-learn/scikit-learn/issues/11622>
    """
    check_estimator(seco_estimator_class)
