"""
Miscellaneous things not depending on anything else from sklearn_seco.
"""

import math

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target


def log2(x: float) -> float:
    """`log2(x) if x > 0 else 0`"""
    return math.log2(x) if x > 0 else 0


def build_categorical_mask(which_features, n_features: int
                           ) -> np.ndarray or None:
    """:return: A mask array of length `n_features` based on `which_features`.
        For its contents, see `_BaseSeCoEstimator` docs.
        Returns None if `which_features` cannot be recognized.
    """
    # which_features modeled like sklearn.preprocessing.OneHotEncoder
    categorical_mask_ = np.zeros(n_features, dtype=bool)  # default "all False"
    if which_features is None or not len(which_features):
        pass  # keep default
    elif isinstance(which_features, np.ndarray):
        categorical_mask_[np.asarray(which_features)] = True
    elif which_features == 'all':
        return np.ones(n_features, dtype=bool)
    else:
        return None
    return categorical_mask_


# noinspection PyAttributeOutsideInit
class BySizeLabelEncoder(BaseEstimator, TransformerMixin):
    """Encode labels with ints between 0 and n_classes-1, ordered descending by
    size.

    Uses LabelEncoder to do the encoding, so there are two steps:

    original labels => LabelEncoder-encoded labels => by-size assigned labels.

    Attributes
    -----
    le_ : LabelEncoder
        Delegate doing the actual encoding, we only reorder.

    cidx_by_size_ : array of shape (n_class,)
        Holds the labels ordering by size, indexed by the `le_`-transformed
        original class labels. I.e. constitutes a mapping from
        LabelEncoder-translated-index => size-ordered class index.

    bsidx_lexsorted_ : array of shape (n_class,)
        Holds the `le_`-transformed original class labels, ordered by size.
        I.e. constitutes a mapping from
        size-ordered class index => LabelEncoder-translated-index.

    See Also
    -----
    :class:`sklearn.label.LabelEncoder`

    <https://github.com/scikit-learn/scikit-learn/issues/4450>
        Feature Request to have LabelEncoder handle that case, too.
    """
    def _fit(self, y):
        target_type = type_of_target(y)
        if target_type not in {'binary', 'multiclass'}:
            raise ValueError("Unknown label type: %s not supported (of y=%s)"
                             % (target_type, y))
        _, class_counts = np.unique(y, return_counts=True)
        self.bsidx_lexsorted_ = np.argsort(- class_counts)
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
        return self.le_.inverse_transform(
            self.bsidx_lexsorted_[np.require(y_enc, dtype=int)])


class TargetTransformingMetaEstimator(BaseEstimator, MetaEstimatorMixin):
    """A Meta-Estimator wrapping a Transformer operating on the target `y` and
    an estimator.

    NOTE: Does not support `decision_function` and `predict_proba`, because
      sklearn assumes lexicographically ordered classes which
      `BySizeLabelEncoder` explicitly breaks.

    # TODO: support get_params etc, maybe using `sklearn.utils.metaestimators._BaseComposition`

    Attributes
    -----
    estimator : estimator object
        Used after `transform` on the transformed labels.

    transform : transformer object
        Used to transform the labels `y` before they're passed to `estimator`.

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

    def predict(self, X):
        return self.transform.inverse_transform(self.estimator.predict(X))

    # common things in other meta-estimators

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def _first_estimator(self):
        return self.estimator
