from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import Bunch

from sklearn_seco.abstract import _BinarySeCoEstimator
from sklearn_seco.concrete import RipperEstimator


# prepare binary iris dataset (classes setosa, other)
iris = load_iris()  # type: Bunch
target_index = 0  # setosa
iris.target[iris.target != target_index] = target_index + 1
iris.target_names = [iris.target_names[target_index], 'other']

est = RipperEstimator()
est.fit(iris.data, iris.target)

# print learning results
print(flush=True)
print("feature names: " + ', '.join(iris.feature_names))

print("rules")
if isinstance(est.base_estimator_, _BinarySeCoEstimator):
    print(est.base_estimator_.theory_)
else:
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
