"""pytest fixtures for the test cases in this directory."""
from typing import Optional, Type

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_raises
from sklearn.utils import check_random_state
from sklearn.utils.testing import set_random_state

from sklearn_seco.abstract import SeCoEstimator
from sklearn_seco.common import Theory
from sklearn_seco.concrete import \
    SimpleSeCoEstimator, CN2Estimator, IrepEstimator, RipperEstimator

from .datasets import Dataset, \
    binary_slight_overlap, binary_categorical, \
    binary_mixed, xor_2d, checkerboard_2d, perfectly_correlated_multiclass, \
    sklearn_make_classification, sklearn_make_moons, artificial_disjunction, \
    staged


def assert_array_unequal(actual, expected, *args):
    """Fail iff arrays are equal. Arguments like `assert_array_equal`."""
    with assert_raises(AssertionError):
        assert_array_equal(actual, expected, *args)


# pytest plugin, to print theory on test failure
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    default = yield
    report = default.get_result()
    if report.failed and report.user_properties:
        for name, prop in report.user_properties:
            if name == 'theory':
                report.longrepr.addsection(name, str(prop))
                break
    return default


@pytest.fixture
def record_theory(record_property):
    def _record(theory: Theory):
        record_property("theory", theory)
    return _record


def count_conditions(theory: Theory,
                     limit: Optional[int] = None,
                     feature_mask: np.ndarray = None):
    """
    :return: the number of conditions (i.e. non-infinite bounds) of all rules
      in `theory`.
    :param limit: None or 'lower' or 'upper'
      If not None, only check the named part of the conditions.
    """
    return np.count_nonzero([np.isfinite(rule.body[limit, feature_mask])
                             for rule in theory])


@pytest.fixture(params=[SimpleSeCoEstimator,
                        CN2Estimator,
                        IrepEstimator,
                        RipperEstimator])
def seco_estimator_class(request) -> Type[SeCoEstimator]:
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
    est = seco_estimator_class()
    set_random_state(est)  # fix random state for tests
    return est


@pytest.fixture
def trivial_decision_border():
    """Generate normal distributed, linearly separated binary problem.

    :return: tuple(X, y)
    """
    random = check_random_state(42)
    samples = np.array([random.normal(size=50),
                        random.random_sample(50),
                        np.zeros(50)]).T
    X = samples[:, :-1]  # all but last column
    y = samples[:, -1]  # last column
    samples[0:25, 1] += 1
    y[0:25] = 1
    random.shuffle(samples)
    return Dataset(X, y)


@pytest.fixture(params=[perfectly_correlated_multiclass,
                        binary_categorical,
                        binary_mixed,
                        xor_2d,
                        checkerboard_2d,
                        binary_slight_overlap,
                        sklearn_make_classification,
                        sklearn_make_moons,
                        artificial_disjunction,
                        staged,
                        ])
def blackbox_test(request):
    return request.param()
