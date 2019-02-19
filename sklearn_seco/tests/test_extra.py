"""Tests for `sklearn_seco.extra`."""

from typing import Sequence

import pytest

from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.extra import trace_coverage, plot_coverage_log, TraceEntry
from sklearn_seco.tests.datasets import xor_2d


@pytest.mark.fast
def test_coverage_tracing(seco_estimator_class):
    """Test the `trace_coverage` mixin, using SimpleSeCo and CN2."""

    trace: Sequence[TraceEntry] = None
    last_rule_stop: bool = None

    def trace_callback(log, lrs):
        nonlocal trace, last_rule_stop
        assert trace is None  # only one run & estimator for binary problem
        trace = log
        last_rule_stop = lrs

    estimator = trace_coverage(seco_estimator_class, trace_callback)()
    X, y, X_test, y_test, cm = xor_2d()
    estimator.fit(X, y, categorical_features=cm)
    # check recognition of binary problem
    base = estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    # subtract boolean, autocasts to integer 0/1
    assert len(base.theory_) == len(trace) - last_rule_stop
    # TODO trace levels

    # test plotting
    tf, rfs = plot_coverage_log(trace, last_rule_stop,
                                title=seco_estimator_class.__name__)
    tf.show()
    rfs.show()
