"""Tests for `sklearn_seco.util`."""

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
