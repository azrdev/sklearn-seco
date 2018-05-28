from sklearn.utils.estimator_checks import check_estimator
from sklearn_seco.seco_base import SimpleSeCoEstimator, CN2Estimator


def test_check_simple():
    return check_estimator(SimpleSeCoEstimator)


def test_check_CN2():
    return check_estimator(CN2Estimator)


# TODO: check estimator vs. classifier in <https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/template.py>
