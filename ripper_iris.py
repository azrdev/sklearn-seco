# script fitting RIPPER on a binarized version of the iris dataset and
# calculating usual evaluation measures on the training data

from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import Bunch

from sklearn_seco.concrete import RipperEstimator


# prepare binary iris dataset (classes setosa, other)
iris = load_iris()  # type: Bunch
target_index = 0  # setosa
# ensure target has biggest index, see `_BinarySeCoEstimator.classes_`
iris.target[iris.target == target_index] = target_index
iris.target[iris.target != target_index] = target_index - 1
iris.target_names = ['other', iris.target_names[target_index]]

est = RipperEstimator()
est.fit(iris.data, iris.target)

# print learning results
print(flush=True)
print("feature names: " + ', '.join(iris.feature_names))

print("# rules #")
if isinstance(est.base_estimator_, OneVsRestClassifier):
    for base, target_class in zip(est.base_estimator_.estimators_,
                                  est.base_estimator_.classes_):
        print("## target {}: {} ##".format(target_class,
                                           iris.target_names[target_class]))
        print(base.export_text(iris.feature_names))  # theory
else:
    for base in est.get_seco_estimators():
        print(f"## delegate with classes_ {base.classes_.tolist()} ##")
        print(base.export_text(iris.feature_names))  # theory

print("\n" + '# "evaluation" on training set #')
pred = est.predict(iris.data)
print(confusion_matrix(iris.target, pred))
print(classification_report(iris.target, pred,
                            target_names=iris.target_names))

pass
