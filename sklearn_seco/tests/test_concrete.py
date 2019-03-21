"""Tests for `sklearn_seco.concrete`."""

from numbers import Real as RealNumber
from typing import List

import numpy as np
import pytest
from numpy import NINF, PINF
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import check_estimator

from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.common import Rule
from sklearn_seco.concrete import grow_prune_split, SimpleSeCoEstimator
from .conftest import count_conditions
from .datasets import perfectly_correlated_multiclass


@pytest.mark.parametrize(
    ['y', 'ratio', 'grow', 'prune'],
    [pytest.param([10, 20], 0.5, [1], [0], id="2 samples, ratio 1/2, test indices"),
     pytest.param([10, 20], 0, [0, 1], [], id="2 samples, ratio 0, test indices"),
     pytest.param([10, 20], 1, [], [0, 1], id="2 samples, ratio 1, test indices"),
     pytest.param([10, 20]*4, 0.5, 0.5, 0.5, id="8 samples, ratio 1/2 test"),
     pytest.param([10, 20]*4, 0, 1, 0, id="8 samples, ratio 0 test"),
     pytest.param([10, 20]*4, 1, 0, 1, id="8 samples, ratio 1 test"),
     pytest.param([10, 20, 20, 50]*250, 1, 0, 1, id="1k samples, ratio 1/3 test"),
     ])
def test_grow_prune_split(y, ratio, grow, prune):
    grow_act, prune_act = grow_prune_split(y, ratio, check_random_state(1))
    # cv, cc = np.unique(y[grow_act], return_counts=True)
    assert np.ndim(grow_act) == 1
    assert np.ndim(prune_act) == 1
    if isinstance(grow, RealNumber):
        # assert cc[0] > grow_act  # TODO: ratio of positive. TODO: non-binary
        assert len(grow_act) / len(y) >= grow
    else:
        assert_array_equal(grow, sorted(grow_act))
    if isinstance(prune, RealNumber):
        assert len(prune_act) / len(y) <= prune
        # assert cc[1] > prune_act  # TODO: ratio of negative. TODO: non-binary
    else:
        assert_array_equal(prune, sorted(prune_act))


def test_base_trivial(record_theory):
    """Test SimpleSeCo with a trivial test set of 2 instances."""
    categorical_mask = np.array([True, False])
    X_train = np.array([[100, 0.0],
                        [111, 1.0]])
    y_train = np.array([1, 2])
    est = SimpleSeCoEstimator() \
        .fit(X_train, y_train, categorical_features=categorical_mask) \
        .base_estimator_
    assert isinstance(est, _BinarySeCoEstimator)
    record_theory(est.theory_)

    assert est.target_class_ == 1
    assert len(est.theory_) == 1
    # first refinement wins (tie breaking)
    assert_array_equal(est.theory_[0].body, [[100, NINF], [PINF, PINF]])

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

    B, numerical ordinal
      ^
      |
     1+ n
     0+
    -1+ p     n
    -2+ p
      +-+-----+-> A, categorical
        0     1
    """
    categorical_mask = np.array([True, False])
    X_train = np.array([[0, -1.0],
                        [0, -2.0],
                        [0,  1.0],
                        [1, -1.0]])
    y_train = np.array([1, 1, 2, 2])
    est = SimpleSeCoEstimator() \
        .fit(X_train, y_train, categorical_features=categorical_mask) \
        .base_estimator_
    assert isinstance(est, _BinarySeCoEstimator)
    record_theory(est.theory_)

    assert est.target_class_ == 1
    assert len(est.theory_) == 2
    assert_array_equal(est.theory_[0].body, np.array([[NINF, NINF], [PINF, -1.5]]))
    assert_array_equal(est.theory_[1].body, np.array([[   0, NINF], [PINF,    0]]))

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
    assert_array_almost_equal(base.theory_[0].body,
                              [[NINF, NINF], [PINF, 1.0]],
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
        assert count_conditions(base.theory_, limit=Rule.UPPER) == 0
    assert_array_equal(dataset.y_train,
                       seco_estimator.predict(dataset.x_train))


def test_sklearn_check_estimator(seco_estimator_class):
    """Run check_estimator from `sklearn.utils.estimator_checks`.

    # TODO: Unwrap :func:`sklearn.utils.estimator_checks.check_estimator`, so
    our report shows which ones actually failed. Waiting for <https://github.com/scikit-learn/scikit-learn/issues/11622>
    """
    # TODO: check_classifiers_predictions sometimes fails for Irep/Ripper: due to bad grow-prune-splits & small n_samples we only recognize 1/3 classes
    check_estimator(seco_estimator_class)


def test_blackbox_accuracy(seco_estimator, blackbox_test, record_theory):
    """Expect high accuracy from each of our blackbox test cases"""
    feature_names = blackbox_test.get_opt("feature_names")

    seco_estimator.fit(
        blackbox_test.x_train, blackbox_test.y_train,
        categorical_features=blackbox_test.categorical_features,
        explicit_target_class=blackbox_test.get_opt("target_class"))

    is_binary = len(np.unique(blackbox_test.y_train)) == 2
    if is_binary:
        base = assert_binary_problem(seco_estimator)
        record_theory(base.theory_)
        print("{} rules:\n{}".format(len(base.theory_),
                                     base.export_text(feature_names)))
    else:
        bases = assert_multiclass_problem(seco_estimator)
        record_theory([b.theory_ for b in bases])
        print("{} theories:\n".format(len(bases)))
        for base_ in bases:
            print("{} rules:\n{}\n".format(len(base_.theory_),
                                           base_.export_text(feature_names)))

    assert_prediction_performance(seco_estimator,
                                  blackbox_test.x_train, blackbox_test.y_train,
                                  blackbox_test.get_opt("x_test"),
                                  blackbox_test.get_opt("y_test"))


# test helpers

def assert_binary_problem(estimator) -> _BinarySeCoEstimator:
    """Check recognition of binary problem by `estimator`.

    :return: the "base_estimator_" `_BinarySeCoEstimator` instance
    """
    base = estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    assert len(base.theory_)
    assert count_conditions(base.theory_, Rule.UPPER,
                            base.categorical_mask_) == 0
    return base


def assert_multiclass_problem(estimator) -> List[_BinarySeCoEstimator]:
    """Check recognition of multi-class problem.

    :return: the list of "base_estimator_" `_BinarySeCoEstimator` instances
    """
    assert not isinstance(estimator.base_estimator_, _BinarySeCoEstimator)
    bases = estimator.base_estimator_.estimators_
    for base_ in bases:
        assert isinstance(base_, _BinarySeCoEstimator)
        assert len(base_.theory_)
        assert count_conditions(base_.theory_, Rule.UPPER,
                                base_.categorical_mask_) == 0
    return bases


def assert_prediction_performance(estimator, x_train, y_train, x_test, y_test):
    # check accuracy,precision on training data
    assert estimator.score(x_train, y_train) > 0.8
    if x_test is not None:
        # check accuracy on test data
        y_predicted = estimator.predict(x_test)
        assert accuracy_score(y_test, y_predicted) > 0.8

        from sklearn.metrics import classification_report, confusion_matrix
        print()
        print(confusion_matrix(y_test, y_predicted))
        print(classification_report(y_test, y_predicted))
