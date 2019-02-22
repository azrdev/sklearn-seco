from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import Bunch

from sklearn_seco.concrete import RipperEstimator


iris = load_iris()  # type: Bunch
est = RipperEstimator()
est.fit(iris.data, iris.target)

print(flush=True)
print("feature names: " + ', '.join(iris.feature_names))

print("rules")
for base, target_class in zip(est.base_estimator_.estimators_,
                              est.base_estimator_.classes_):
    print("target {}: {}".format(target_class,
                                 iris.target_names[target_class]))
    print(base.theory_)

print("\n" + '"evaluation" on training set')
pred = est.predict(iris.data)
print(confusion_matrix(iris.target, pred))
print(classification_report(iris.target, pred,
                            target_names=iris.target_names))

pass
