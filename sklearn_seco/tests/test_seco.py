from sklearn.utils.estimator_checks import check_estimator, _yield_all_checks
from sklearn_seco.seco_base import SimpleSeCoEstimator, CN2Estimator


def test_check_simple():
    # return check_estimator(SimpleSeCoEstimator)

    # disassembled check_estimator, to have all as separate nosetests
    name = SimpleSeCoEstimator.__name__
    estimator = SimpleSeCoEstimator()
    for check in _yield_all_checks(name, estimator):
        check.description = check.__name__
        yield check, name, estimator



def test_check_CN2():
    return check_estimator(CN2Estimator)


# TODO: check estimator vs. classifier in <https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/template.py>
