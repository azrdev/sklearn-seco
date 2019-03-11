"""Tests for `sklearn_seco.extra`."""
from typing import Tuple

import pytest

from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.extra import trace_coverage, plot_coverage_log, Trace
from sklearn_seco.tests.datasets import xor_2d


@pytest.fixture
def trace_estimator(seco_estimator_class) -> Tuple[Trace, str]:

    trace: Trace = None

    def trace_callback(trace_: Trace):
        nonlocal trace
        assert trace is None  # only one run & estimator for binary problem
        trace = trace_

    estimator = trace_coverage(seco_estimator_class, trace_callback)()
    X, y, X_test, y_test, cm = xor_2d()
    estimator.fit(X, y, categorical_features=cm)
    # check recognition of binary problem
    base = estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    # check trace consistency
    assert isinstance(trace.last_rule_stop, bool)
    # subtract boolean, autocasts to integer 0/1
    assert len(base.theory_) == len(trace.steps) - trace.last_rule_stop

    # TODO trace levels
    return trace, seco_estimator_class.__name__


@pytest.mark.fast
def test_coverage_tracing(trace_estimator):
    """Test the `trace_coverage` mixin."""
    trace, estimator_name = trace_estimator

    # test (de)serialization
    json_str = trace.to_json()
    trace_recovered = Trace.from_json(json_str)
    assert trace == trace_recovered

    # test plotting
    figures = plot_coverage_log(trace, title="XOR 2d on " + estimator_name)
    for figure in figures:
        figure.show()
