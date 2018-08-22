"""Tests for `sklearn_seco.extra`."""

import pytest
from numpy.testing import assert_array_equal

from sklearn_seco.abstract import SeCoEstimator, _BinarySeCoEstimator
from sklearn_seco.concrete import SimpleSeCoImplementation, CN2Implementation
from sklearn_seco.extra import TraceCoverage, plot_coverage_log


@pytest.mark.parametrize('implementation_class',
                         [SimpleSeCoImplementation, CN2Implementation])
def test_coverage_tracing(binary_categorical, implementation_class):
    """Test the `TraceCoverage` mixin, using SimpleSeCo and CN2."""

    class TracingImpl(TraceCoverage, implementation_class):
        pass
    tracer = TracingImpl()
    estimator = SeCoEstimator(tracer)
    X, y, X_test, y_test = binary_categorical
    estimator.fit(X, y, categorical_features='all')
    # check recognition of binary problem
    base = estimator.base_estimator_
    assert isinstance(base, _BinarySeCoEstimator)
    # check consistency of trace
    assert tracer.has_complete_trace
    assert isinstance(tracer.last_rule_stop, bool)
    assert len(base.theory_) == \
           len(tracer.coverage_log) - tracer.last_rule_stop  # bool is int 0/1
    assert len(base.theory_) == \
           len(tracer.refinement_log) - tracer.last_rule_stop
    # TODO trace levels

    # test (de)serialization
    json_str = tracer.to_json()
    trace_recovered = TraceCoverage.from_json(json_str)
    assert tracer.last_rule_stop == trace_recovered["last_rule_stop"]
    for x, y in zip(tracer.coverage_log, trace_recovered["coverage_log"]):
        assert_array_equal(x, y)
    for x, y in zip(tracer.refinement_log, trace_recovered["refinement_log"]):
        assert_array_equal(x, y)
    for x, y in zip(tracer.PN, trace_recovered["PN"]):
        assert_array_equal(x, y)

    # test plotting
    tf, rfs = plot_coverage_log(**trace_recovered,
                                title=implementation_class.__name__)
    tf.show()
    rfs.show()
