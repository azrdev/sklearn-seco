import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from sklearn.utils.estimator_checks import _yield_all_checks, \
    check_parameters_default_constructible, check_no_fit_attributes_set_in_init
from sklearn_seco.seco_base import \
    SimpleSeCoEstimator, CN2Estimator, _BinarySeCoEstimator, \
    SimpleSeCoImplementation, \
    match_rule


def test_match_rule():
    from numpy import NaN
    categorical_mask = np.array([True, True, False, False])
    X = np.array([[1, 2, 3.0, 4.0]])

    def assert_match(rule, expected_result, X=X):
        assert_array_equal(match_rule(X, np.array(rule), categorical_mask),
                           expected_result)

    assert_match([NaN, NaN, NaN, NaN], [True])
    assert_match([1,   2,   NaN, NaN], [True])  # categorical ==
    assert_match([0,   2,   NaN, NaN], [False])
    assert_match([11,   2,   NaN, NaN], [False])
    assert_match([NaN, NaN, 13.0, 14.0], [True])  # numerical <=
    assert_match([NaN, NaN, -3.0, -4.0], [False])
    assert_match([NaN, NaN, 3.0, NaN], [True])
    assert_match([1, 2, 10.0, 10.0], [True])

    # test broadcasting using 4 samples, 4 features
    X4 = np.array([[1, 2, 3.0, 4.0],
                   [1, 2, 30.0, 40.0],
                   [1, 2, 0.0,  0.0],
                   [0, 0, 2.0, 3.0]])
    assert_match([1, NaN, 3.0, 4.0], [True, False, True, False], X=X4)

    # TODO: define & test NaN in X


def test_base_easyrules():
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
    assert_array_equal(est.theory_[0], np.array([np.NaN, -1.5]))
    assert_array_equal(est.theory_[1], np.array([0, 0]))

    assert_array_equal(est.predict(X_train), y_train)

    X_test = np.array([[0, 14],
                       [1, -15],
                       [0, -16],
                       [0, 0],
                       [1, 1]
                       ])
    y_test = np.array([2, 1, 1, 1, 2])
    assert_array_equal(est.predict(X_test), y_test)


def test_check_simple():
    # check_estimator(SimpleSeCoEstimator)

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
    # check_estimator(CN2Estimator)

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
# TODO: failing sklearn checks:
# - check_classifiers_classes misses 1/3 classes in the prediction
# - check_classifiers_train due to low accuracy
