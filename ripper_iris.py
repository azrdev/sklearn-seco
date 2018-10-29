from sklearn.datasets import load_iris

from sklearn_seco.concrete import RipperEstimator

iris = load_iris()
est = RipperEstimator()
est.fit(iris.data, iris.target)


for base, target_class in zip(est.base_estimator_.estimators_,
                              est.base_estimator_.classes_):
    print(target_class)
    print(base.theory_)
