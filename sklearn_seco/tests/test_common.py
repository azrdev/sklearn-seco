"""Tests for `sklearn_seco.common`."""

import numpy as np
from numpy.core.umath import NINF, PINF
from numpy.testing import assert_array_equal

from sklearn_seco.common import match_rule, Rule, AugmentedRule
from sklearn_seco.concrete import SimpleSeCoEstimator
from .conftest import assert_array_unequal


def test_match_rule():
    """Test basic rule matching (:func:`match_rule`)."""
    categorical_mask = np.array([True, True, False, False])
    X = np.array([[1, 2, 3.0, 4.0]])

    def am(body, expected_result, X=X):
        assert_array_equal(match_rule(X, Rule(100, np.asarray(body)),
                                      categorical_mask),
                           expected_result)

    am([[NINF, NINF, NINF, NINF], [PINF, PINF, PINF, PINF]], [True])
    # categorical: upper unused
    am([[NINF, NINF, NINF, NINF], [7,    7,    PINF, PINF]], [True])
    # categorical ==
    am([[1,    2,    NINF, NINF], [PINF, PINF, PINF, PINF]], [True])
    am([[0,    2,    NINF, NINF], [PINF, PINF, PINF, PINF]], [False])
    am([[11,   2,    NINF, NINF], [PINF, PINF, PINF, PINF]], [False])
    # numerical <= upper
    am([[NINF, NINF, NINF, NINF], [PINF, PINF, 13.0, 14.0]], [True])
    am([[NINF, NINF, NINF, NINF], [PINF, PINF, -3.0, -4.0]], [False])
    am([[NINF, NINF, NINF, NINF], [PINF, PINF,  3.0, PINF]], [True])
    am([[NINF, NINF, NINF, NINF], [1,    2,    10.0, 10.0]], [True])

    # test broadcasting using 4 samples, 4 features
    X4 = np.array([[1, 2, 3.0, 4.0],
                   [1, 2, 30.0, 40.0],
                   [1, 2, 0.0,  0.0],
                   [0, 0, 2.0, 3.0]])
    am([[   1, NINF, NINF, NINF],
        [PINF, PINF,  3.0,  4.0]], [True, False, True, False], X=X4)


def test_copy_rule():
    rule = Rule.make_empty(17, 'positive')
    copy = rule.copy()
    assert rule.head == copy.head, "Rule.head modified head"
    assert_array_equal(rule.body, copy.body, "Rule.copy modified body")
    mod_copy = rule.copy()
    mod_copy.head = 'negative'
    mod_copy.body[Rule.LOWER, 11] = 42

    assert_array_unequal(rule.body, mod_copy.body)
    assert_array_equal(rule.body, copy.body,
                       "modifying copied rule changed original")


def test_copy_augmentedrule():
    rule = AugmentedRule.make_empty(17, 'positive')
    copy_unmod = rule.copy()
    assert rule.head == copy_unmod.head
    assert_array_equal(rule.body, copy_unmod.body)
    copy_mod_head = rule.copy(head='negative')
    assert rule.head != copy_mod_head.head
    assert_array_equal(rule.body, copy_mod_head.body)
    copy_mod_body = rule.copy(condition=(Rule.LOWER, 11, 42))
    assert rule.head == copy_mod_body.head
    assert_array_unequal(rule.body, copy_mod_body.body)
    assert_array_equal(rule.body, copy_unmod.body,
                       "modifying copied rule changed original")


def test_categorical_mask():
    X = [[1, 2, 3]] * 10 + [[3, 2, 1]] * 10
    y = [13] * 10 + [31] * 10
    estimator = SimpleSeCoEstimator().fit(X, y, categorical_features=None)
    assert_array_equal(estimator.get_seco_estimators()[0].categorical_mask_,
                       [False, False, False])
    estimator = SimpleSeCoEstimator().fit(X, y, categorical_features=[])
    assert_array_equal(estimator.get_seco_estimators()[0].categorical_mask_,
                       [False, False, False])
    estimator = SimpleSeCoEstimator().fit(X, y, categorical_features='all')
    assert_array_equal(estimator.get_seco_estimators()[0].categorical_mask_,
                       [True, True, True])
    estimator = SimpleSeCoEstimator() \
        .fit(X, y, categorical_features=np.array([True, False, False]))
    assert_array_equal(estimator.get_seco_estimators()[0].categorical_mask_,
                       [True, False, False])


# TODO: test _BinarySeCoEstimator.ordered_matching = True
# TODO: test match_rule without numba
