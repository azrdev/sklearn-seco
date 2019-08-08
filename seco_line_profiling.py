import warnings

from line_profiler import LineProfiler

from sklearn_seco import SimpleSeCoEstimator
from sklearn_seco.common import match_rule, RuleContext
from sklearn_seco.tests import conftest


def tcn2(dataset):
    with warnings.catch_warnings():
        from _pytest.deprecated import RemovedInPytest4Warning
        warnings.simplefilter("ignore", RemovedInPytest4Warning)
    est = SimpleSeCoEstimator()
    est.fit(dataset.x_train, dataset.y_train)
    ypred = est.predict(dataset.x_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(dataset.y_test, ypred))
    print(classification_report(dataset.y_test, ypred))

try:
    from numba import NumbaWarning
    warnings.simplefilter("error", NumbaWarning)
except ImportError:
    pass
profile = LineProfiler()
profile.add_function(match_rule)
profile.add_function(RuleContext._count_matches)
profile.add_function(RuleContext.pn)
profile.run('tcn2(conftest.sklearn_make_moons())')
profile.print_stats()
