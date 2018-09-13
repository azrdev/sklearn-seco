import sys
import warnings

from line_profiler import LineProfiler

from sklearn_seco.tests.conftest import xor_2d
from sklearn_seco.common import match_rule, count_matches
from sklearn_seco.concrete import CN2Estimator


def tcn2():
    with warnings.catch_warnings():
        from _pytest.deprecated import RemovedInPytest4Warning
        warnings.simplefilter("ignore", RemovedInPytest4Warning)
        x, y, xtest, ytest = xor_2d()
    cn2 = CN2Estimator()
    cn2.fit(x, y)
    ypred = cn2.predict(xtest)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred))


profile = LineProfiler()
if 'm' in sys.argv[1:]:
    profile.add_function(match_rule)
if 'c' in sys.argv[1:]:
    profile.add_function(count_matches)
profile.run('tcn2()')
profile.print_stats()
