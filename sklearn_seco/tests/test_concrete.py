"""Tests for `sklearn_seco.concrete`."""

from typing import List

import numpy as np
import pytest
from numpy import NINF, PINF
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.metrics import accuracy_score, precision_score
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import check_estimator

from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.common import Rule
from sklearn_seco.concrete import grow_prune_split, SimpleSeCoEstimator, \
    TopDownSearchImplementation
from sklearn_seco.util import TargetTransformingMetaEstimator, \
    BySizeLabelEncoder
from .conftest import count_conditions
from .datasets import perfectly_correlated_multiclass


@pytest.mark.parametrize(['y', 'ratio', 'grow', 'prune'], [
    pytest.param([10, 20], 0, [0, 1], [], id="2 samples, ratio 0"),
    pytest.param([10, 20], 1, [], [0, 1], id="2 samples, ratio 1"),
    # below testcases expect all samples in the growing set, because the
    # multiclass grow_prune_split doesn't know which class would be negative
    # and could safely be omitted from the growing set
    pytest.param([10, 20], 0.5, [0, 1], [], id="2 samples, ratio 1/2"),
    pytest.param([10, 20], 1/3, [0, 1], [], id="2 samples, ratio 1/3"),
])
def test_grow_prune_split_concrete(y, ratio: float,
                                   grow: List[int], prune: List[int]):
    grow_act, prune_act = grow_prune_split(y, ratio, check_random_state(1))
    assert np.ndim(grow_act) == 1
    assert np.ndim(prune_act) == 1
    assert_array_equal(grow, sorted(grow_act))
    assert_array_equal(prune, sorted(prune_act))


@pytest.mark.parametrize('ratio', [
    pytest.param(0.5, id="ratio 1/2"),
    pytest.param(0.0, id="ratio 0"),
    pytest.param(0.1, id="ratio 0,1",
                 marks=pytest.mark.xfail(reason="for small n_samples and/or "
                                                "small classes, all samples "
                                                "end up in growing set. "
                                                "maybe a bug")),
    pytest.param(1.0, id="ratio 1"),
    pytest.param(1/3, id="ratio 1/3"),
])
@pytest.mark.parametrize('y', [
    pytest.param([10, 20]*4, id="8 samples balanced binary"),
    pytest.param([11, 12, 13]*7, id="21 samples 3 balanced classes"),
    pytest.param([10, 20, 20, 50]*250, id="1k samples 4 little imbalanced classes"),
    pytest.param([10, 20]*100 + [40]*5, id="3 imbalanced classes 1 minority class"),
])
def test_grow_prune_split_ratio(y, ratio: float):
    grow_ratio_exp = 1 - ratio
    prune_ratio_exp: float = ratio
    y = np.asarray(y)
    grow_act, prune_act = grow_prune_split(y, ratio, check_random_state(1))
    assert np.ndim(grow_act) == 1, "grow not a list of indices"
    assert np.ndim(prune_act) == 1, "prune not a list of indices"

    # total ratios are obeyed (with grow_ratio rounded up)
    assert len(grow_act) / len(y) >= grow_ratio_exp, "too few samples in growing set"
    assert len(prune_act) / len(y) <= prune_ratio_exp, "too few samples in pruning set"

    # per class ratios are obeyed
    ycv, ycc = np.unique(y, return_counts=True)
    gcv, gcc = np.unique(y[grow_act], return_counts=True)
    pcv, pcc = np.unique(y[prune_act], return_counts=True)
    # TODO ratio != {0, 1} can still yield 0 samples from very small class
    if ratio < 1:
        assert_array_equal(gcv, ycv, "not all classes in growing set")
        for c, gcci, ycci in zip(ycv, gcc, ycc):
            assert gcci / ycci >= grow_ratio_exp, f"too few samples of class {c} in growing set"
    if ratio > 0:
        assert_array_equal(pcv, ycv, "not all classes in pruning set")
        for c, pcci, ycci in zip(ycv, gcc, ycc):
            assert pcci / ycci >= prune_ratio_exp, f"too few samples of class {c} in pruning set"

    if 0 < ratio < 1:
        assert_array_equal(gcc + pcc, ycc, "not all samples from y in {growing + pruning}")


def test_TopDownSearch():
    # setup an estimator using TopDownSearch
    X = np.array([[1., 1, 99], [2., 3, 99], [3., 4, 99]] * 4)
    y = np.array([10, 20, 30] * 4)
    try:
        estimator = SimpleSeCoEstimator()
        estimator.fit(X, y, categorical_features=np.array([False, True, True]))
    except BaseException as e:
        print(e)  # ignore, we're not here to test SimpleSeCo
    assert estimator.multi_class_ == 'direct'
    assert len(estimator.get_seco_estimators()) == 1
    base = estimator.get_seco_estimators()[0]
    transform: BySizeLabelEncoder = estimator.base_estimator_.transform
    implementation = base.algorithm_config_.implementation
    assert isinstance(implementation, TopDownSearchImplementation)
    # setup TheoryContext and RuleContext
    theory_context = base.algorithm_config_.make_theory_context(
        base.categorical_mask_, base.n_features_, base.classes_, base.rng,
        X, transform.transform(y))
    rule_context = base.algorithm_config_.make_rule_context(
        theory_context, X, transform.transform(y))
    # check rule specialization
    rule = base.algorithm_config_.make_rule(base.n_features_,
                                            estimator.classes_[-1])
    refinements = list(implementation.refine_rule(rule, rule_context))
    assert_array_equal(
        np.unique(transform.inverse_transform([r.head for r in refinements])),
        [10, 20, 30])
    assert_array_equal(np.unique([r.lower[1] for r in refinements]),
                       [np.NINF, 1, 3, 4])
    assert_array_equal(np.unique([r.upper[2] for r in refinements]), [np.PINF])
    # 2 numeric splits +2 for +-inf; 3, 1 categorical matches; 3 target classes.
    assert len(refinements) == ((2+2) + 3 + 1) * 3
    # check that categorical test is not overridden
    value2 = 4
    rule2 = rule.copy(condition=(Rule.LOWER, 1, value2))
    refinements2 = list(implementation.refine_rule(rule2, rule_context))
    assert_array_equal(np.unique([r.lower[1] for r in refinements2]),
                       [value2])


def test_base_trivial(record_theory):
    """Test SimpleSeCo with a trivial test set of 2 instances."""
    categorical_mask = np.array([True, False])
    X_train = np.array([[100, 0.0],
                        [111, 1.0]])
    y_train = np.array([1, 2])
    est = SimpleSeCoEstimator().fit(X_train, y_train,
                                    categorical_features=categorical_mask)
    assert isinstance(est.base_estimator_, TargetTransformingMetaEstimator)
    base = est.base_estimator_.estimator
    assert isinstance(base, _BinarySeCoEstimator)

    record_theory(base.theory_)
    assert_array_equal(base.classes_,  [0, 1])  # indices
    assert len(base.theory_) == 1
    # first refinement wins (tie breaking)
    assert_array_equal(base.theory_[0].body, [[100, NINF], [PINF, PINF]])

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
    est = SimpleSeCoEstimator().fit(X_train, y_train,
                                    categorical_features=categorical_mask)
    assert isinstance(est.base_estimator_, TargetTransformingMetaEstimator)
    base = est.base_estimator_.estimator
    assert isinstance(base, _BinarySeCoEstimator)

    record_theory(base.theory_)

    assert_array_equal(base.classes_, [0, 1])  # indices
    assert len(base.theory_) == 2
    assert_array_equal(base.theory_[0].body, np.array([[NINF, NINF], [PINF, -1.5]]))
    assert_array_equal(base.theory_[1].body, np.array([[   0, NINF], [PINF,    0]]))

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
    assert isinstance(seco_estimator.base_estimator_,
                      TargetTransformingMetaEstimator)
    base = seco_estimator.base_estimator_.estimator
    assert isinstance(base, _BinarySeCoEstimator)
    record_theory(base.theory_)
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
    record_theory([b.theory_ for b in seco_estimator.get_seco_estimators()])
    # check rules
    if seco_estimator.multi_class_ == 'direct':
        assert len(seco_estimator.get_seco_estimators()) == 1
        base = seco_estimator.get_seco_estimators()[0]
        assert len(base.classes_) == 10
        assert len(base.theory_) == 10
        assert count_conditions(base.theory_, limit=Rule.UPPER) == 0
    else:
        for base in seco_estimator.get_seco_estimators():
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
        categorical_features=blackbox_test.categorical_features)

    theories = [base.theory_ if hasattr(base, 'theory_') else None
                for base in seco_estimator.get_seco_estimators()]
    record_theory(theories)
    print("{} theories:\n".format(len(theories)))
    for base in seco_estimator.get_seco_estimators():
        assert isinstance(base, _BinarySeCoEstimator)
        assert len(base.theory_)
        assert count_conditions(base.theory_, Rule.UPPER,
                                base.categorical_mask_) == 0, \
            "rule.UPPER set for categorical feature"
        print("{} rules:\n{}".format(len(base.theory_),
                                     base.export_text(feature_names)))
    assert_prediction_performance(seco_estimator,
                                  blackbox_test.x_train, blackbox_test.y_train,
                                  blackbox_test.get_opt("x_test"),
                                  blackbox_test.get_opt("y_test"))


# test helpers


def assert_prediction_performance(estimator, x_train, y_train, x_test, y_test):
    # check accuracy,precision on training data
    assert estimator.score(x_train, y_train) > 0.8
    assert precision_score(y_train, estimator.predict(x_train),
                           average='weighted') > 0.8
    if x_test is not None:
        # check accuracy on test data
        y_predicted = estimator.predict(x_test)
        assert accuracy_score(y_test, y_predicted) > 0.8

        from sklearn.metrics import classification_report, confusion_matrix
        print()
        print(confusion_matrix(y_test, y_predicted))
        print(classification_report(y_test, y_predicted))
