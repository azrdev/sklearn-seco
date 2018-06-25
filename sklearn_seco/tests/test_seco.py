import numpy as np
from numpy import NINF, PINF
from nose.tools import assert_is_instance
from numpy.testing import assert_array_equal, assert_equal, \
    assert_array_almost_equal
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import _yield_all_checks, \
    check_parameters_default_constructible, check_no_fit_attributes_set_in_init
from sklearn_seco.seco_base import \
    SimpleSeCoEstimator, CN2Estimator, _BinarySeCoEstimator, \
    SimpleSeCoImplementation, \
    match_rule, make_empty_rule, Rule


def test_match_rule():
    """Test :func:`match_rule`, i.e. basic rule matching."""
    categorical_mask = np.array([True, True, False, False])
    X = np.array([[1, 2, 3.0, 4.0]])

    def am(rule, expected_result, X=X):
        assert_array_equal(match_rule(X, Rule(np.asarray(rule)),
                                      categorical_mask),
                           expected_result)

    am(make_empty_rule(len(categorical_mask)), [True])
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

    # TODO: define & test NaN in X (missing values)


def test_base_easyrules():
    """Test :class:`_BinarySeCoEstimator` with some small test set for some
    trivial rules
    """
    categorical_mask = np.array([True, False])
    X_train = np.array([[0, -1.0],
                        [0, -2.0],
                        [0,  1.0],
                        [1, -1.0]])
    y_train = np.array([1, 1, 2, 2])
    est = _BinarySeCoEstimator(SimpleSeCoImplementation())
    est.fit(X_train, y_train, categorical_mask)

    assert_equal(est.target_class_, 1)
    assert_equal(len(est.theory_), 2)
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


def test_trivial_decision_border():
    """Generate two scattered classes without overlap and check the border is
    determined correctly.
    """
    random = check_random_state(None)
    X = np.array([random.normal(size=50), random.random_sample(50)]).T
    X[0:25, 1] += 1
    y = np.zeros(50)
    y[0:25] = 1
    est = SimpleSeCoEstimator()
    est.fit(X, y)
    base = est.base_estimator_
    assert_is_instance(base, _BinarySeCoEstimator)
    assert_equal(base.target_class_, 0)
    assert_equal(len(base.theory_), 1)
    assert_array_almost_equal(base.theory_[0], [[NINF, NINF], [PINF, 1.0]],
                              decimal=1)


def test_check_simple():
    """Unwrap :func:`sklearn.utils.estimator_checks.check_estimator` for
    :class:`SimpleSeCoEstimator`.
    """

    # disassembled check_estimator, to have all as separate nosetests
    Estimator = SimpleSeCoEstimator
    name = Estimator.__name__
    check_parameters_default_constructible(name, Estimator)
    check_no_fit_attributes_set_in_init(name, Estimator)

    estimator = Estimator()
    for check in _yield_all_checks(name, estimator):
        check.description = check.__name__
        yield check, name, estimator


def test_check_CN2():
    """Unwrap :func:`sklearn.utils.estimator_checks.check_estimator` for
    :class:`CN2Estimator`.
    """

    # disassembled check_estimator, to have all as separate nosetests
    Estimator = CN2Estimator
    name = Estimator.__name__
    check_parameters_default_constructible(name, Estimator)
    check_no_fit_attributes_set_in_init(name, Estimator)

    estimator = Estimator()
    for check in _yield_all_checks(name, estimator):
        check.description = check.__name__
        yield check, name, estimator


# TODO: check estimator vs. classifier in <https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/template.py>
