"""Tests for `sklearn_seco.extra`."""

from collections import namedtuple

from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.extra import trace_coverage, plot_coverage_log
from sklearn_seco.tests.datasets import binary_categorical


def test_coverage_tracing(seco_estimator_class):
    """Test the `trace_coverage` mixin, using SimpleSeCo and CN2."""

    Trace = namedtuple("Trace", ["coverage_log", "refinement_log", "last_rule_stop", "PN"])
    trace: Trace = None

    def trace_callback(coverage_log, refinement_log, last_rule_stop, PN):
        nonlocal trace
        assert trace is None  # only one run & estimator for binary problem
        trace = Trace(coverage_log, refinement_log, last_rule_stop, PN)

    estimator = trace_coverage(seco_estimator_class, trace_callback)()
    X, y, X_test, y_test, cm = binary_categorical()
    estimator.fit(X, y, categorical_features=cm)
    # check recognition of binary problem
    base = estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    # check consistency of trace
    assert isinstance(trace, Trace)
    assert isinstance(trace.last_rule_stop, bool)
    # subtract boolean, autocasts to integer 0/1
    assert len(base.theory_) == len(trace.coverage_log) - trace.last_rule_stop
    assert len(base.theory_) == len(trace.refinement_log) - trace.last_rule_stop
    # TODO trace levels

    # test plotting
    tf, rfs = plot_coverage_log(**trace._asdict(),
                                title=seco_estimator_class.__name__)
    tf.show()
    rfs.show()
