import warnings

from line_profiler import LineProfiler

from sklearn_seco.tests.conftest import xor_2d
from sklearn_seco.common import match_rule, RuleContext
from sklearn_seco.concrete import CN2Estimator


def tcn2():
    with warnings.catch_warnings():
        from _pytest.deprecated import RemovedInPytest4Warning
        warnings.simplefilter("ignore", RemovedInPytest4Warning)
        xor = xor_2d()
    cn2 = CN2Estimator()
    cn2.fit(xor.x_train, xor.y_train)
    ypred = cn2.predict(xor.x_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(xor.y_test, ypred))
    print(classification_report(xor.y_test, ypred))


profile = LineProfiler()
profile.add_function(match_rule)
profile.add_function(RuleContext._count_matches)
profile.add_function(RuleContext.pn)
profile.run('tcn2()')
profile.print_stats()
