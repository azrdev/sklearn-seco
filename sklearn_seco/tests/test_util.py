"""Tests for `sklearn_seco.util`."""
import functools

import numpy as np
from numpy.testing import assert_array_equal

from sklearn_seco import util


def test_BySizeLabelEncoder():
    y_full = np.array(['extra small']
                      + ['small']*2
                      + ['medium']*10
                      + ['big']*40
                      + ['extralarge']*200)
    y_test = np.array(['big', 'extra small', 'extralarge', 'medium', 'small'])
    enc = util.BySizeLabelEncoder()
    enc.fit(y_full)
    y_trans = enc.transform(y_test)
    assert_array_equal(y_test, enc.inverse_transform(y_trans),
                       "y != inverse_transform(transform(y))")
    assert_array_equal(y_trans, [1, 4, 0, 2, 3],
                       "transformed classes not ordered by descending size")

    assert_array_equal(enc.fit(y_full).transform(y_full),
                       enc.fit_transform(y_full),
                       "fit_transform() != fit().transform()")


def test_categorical_mask():
    X = [[1, 2, 3]] * 10 + [[3, 2, 1]] * 10
    y = [13] * 10 + [31] * 10
    build_mask = functools.partial(util.build_categorical_mask, n_features=3)
    assert_array_equal(build_mask(None), [False, False, False])
    assert_array_equal(build_mask([]), [False, False, False])
    assert_array_equal(build_mask('all'), [True, True, True])
    assert_array_equal(build_mask(np.array([True, False, False])),
                       [True, False, False])
