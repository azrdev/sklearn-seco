"""pytest fixtures for the test cases in this directory."""
from typing import Union, List

import numpy as np
import pytest
from sklearn.utils import check_random_state

from sklearn_seco.concrete import \
    SimpleSeCoEstimator, CN2Estimator, RipperEstimator


from .datasets import Dataset, \
    binary_slight_overlap, binary_categorical, \
    binary_mixed, xor_2d, checkerboard_2d


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
    def _record(theory: Union[np.ndarray, List[np.ndarray]]):
        record_property("theory", theory)
    return _record


def count_conditions(theory):
    """
    :return: the number of conditions (i.e. non-infinite bounds) of all rules
      in `theory`.
    """
    return np.count_nonzero(np.isfinite(theory))


@pytest.fixture(params=[SimpleSeCoEstimator, CN2Estimator, RipperEstimator])
def seco_estimator_class(request):
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
    return seco_estimator_class()


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


@pytest.fixture
def perfectly_correlated_multiclass():
    """Generate 10-class problem with 10 features each matching one class."""
    n = 10
    y = np.arange(1, n + 1)
    x = np.eye(n, dtype=int) * y
    return Dataset(x, y, categorical_features='all')


@pytest.fixture(params=[binary_categorical,
                        binary_mixed,
                        xor_2d,
                        checkerboard_2d,
                        binary_slight_overlap])
def blackbox_test(request):
    return request.param()
