"""
Miscellaneous things not depending on anything else from sklearn_seco.
"""

import math

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.preprocessing import LabelEncoder


def log2(x: float) -> float:
    """`log2(x) if x > 0 else 0`"""
    return math.log2(x) if x > 0 else 0


# noinspection PyAttributeOutsideInit
class BySizeLabelEncoder(BaseEstimator, TransformerMixin):
    """Encode labels with ints between 0 and n_classes-1, ordered descending by
    size.

    Attributes
    -----
    le_ : LabelEncoder
        Delegate doing the actual encoding

    cidx_by_size_ : array of shape (n_class,)
        Holds the class labels (after translation by le_), ordered by size.
        I.e. a mapping from
        LabelEncoder-translated-index => size-ordered class index.

    See Also
    -----
    :class:`sklearn.label.LabelEncoder`

    <https://github.com/scikit-learn/scikit-learn/issues/4450>
        Feature Request to have LabelEncoder handle that case, too.
    """
    def _fit(self, y):
        _, class_counts = np.unique(y, return_counts=True)
        self.bsidx_lexsorted_ = np.argsort(class_counts)[::-1]
        self.cidx_by_size_ = np.argsort(self.bsidx_lexsorted_)

    def fit(self, y):
        """Learn labels present in `y` and calculate encoding."""
        self.le_: LabelEncoder = LabelEncoder().fit(y)
        self._fit(y)
        return self

    def transform(self, y):
        """Transform `y` to normalized encoding."""
        return self.cidx_by_size_[self.le_.transform(y)]

    # noinspection PyMethodOverriding
    def fit_transform(self, y):
        """Learn labels in `y` and encode them."""
        self.le_ = LabelEncoder()
        self._fit(y)
        return self.cidx_by_size_[self.le_.fit_transform(y)]

    def inverse_transform(self, y_enc):
        """Transform labels back to original encoding."""
        return self.le_.inverse_transform(self.bsidx_lexsorted_[y_enc])


class TargetTransformingMetaEstimator(BaseEstimator, MetaEstimatorMixin):
    """A Meta-Estimator wrapping a Transformer operating on the target `y` and
    an estimator.

    NOTE: Does not support `decision_function` and `predict_proba`, because
      sklearn assumes lexicographically ordered classes which
      `BySizeLabelEncoder` explicitly breaks.

    # TODO: support get_params and stuff, maybe using `sklearn.utils.metaestimators._BaseComposition`

    See Also
    -----
    SLEP001
        "Transformers that modify their target"
        - <https://github.com/scikit-learn/enhancement_proposals/tree/master/slep001>
        - <https://github.com/scikit-learn/enhancement_proposals/pull/2>

    <https://github.com/scikit-learn/scikit-learn/issues/4143>
    """
    def __init__(self, transform, estimator):
        self.transform = transform
        self.estimator = estimator

    def fit(self, X, y):
        y_trans = self.transform.fit_transform(y)
        self.estimator.fit(X, y_trans)
        return self

    def fit_transform(self, X, y):
        y_trans = self.transform.fit_transform(y)
        return self.estimator.fit_transform(X, y_trans)

    def predict(self, X):
        return self.transform.inverse_transform(self.estimator.predict(X))

    # common things in other meta-estimators

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def _first_estimator(self):
        return self.estimator
