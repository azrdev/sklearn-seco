"""
Implementation of SeCo / Covering algorithm: Abstract base algorithm.
"""

from typing import Union

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import \
    unique_labels, check_classification_targets
from sklearn.utils.validation import check_is_fitted


# noinspection PyAttributeOutsideInit
class _BinarySeCoEstimator(BaseEstimator, ClassifierMixin):
    """Binary SeCo Classification, deferring to :class:`AbstractSecoImplementation`
    for concrete algorithm implementation.

    :param implementation: A `AbstractSecoImplementation` subclass whose
      methods define the algorithm to be run.

    :param categorical_features: None or “all” or array of indices or mask.

        Specify what features are treated as categorical, i.e. equality
        tests are used for these features, based on the set of values
        present in the training data.

        Note that numerical features may be tested in multiple (inequality)
        conditions of a rule, while multiple equality tests (for a
        categorical feature) would be useless.

        -   None (default): All features are treated as numerical & ordinal.
        -   'all': All features are treated as categorical.
        -   array of indices: Array of categorical feature indices.
        -   mask: Array of length n_features and with dtype=bool.

        You may instead transform your categorical features beforehand,
        using e.g. :class:`sklearn.preprocessing.OneHotEncoder` or
        :class:`sklearn.preprocessing.Binarizer`.
        TODO: compare performance

    :param explicit_target_class: Use as positive/target class for learning. If
        `None` (the default), use the first class from `np.unique(y)` (which is
        sorted).
    """
    def __init__(self,
                 implementation: 'AbstractSecoImplementation',
                 categorical_features: Union[None, str, np.ndarray] = None,
                 explicit_target_class=None):
        super().__init__()
        # FIXME: `implementation` shared across estimator.copy() instances
        self.implementation = implementation
        self.categorical_features = categorical_features
        self.explicit_target_class = explicit_target_class

    def fit(self, X, y):
        """Fit to data, i.e. learn theory

        :param X: Not yet covered examples.
        :param y: Classification for `X`.
        """
        X, y = check_X_y(X, y, dtype=np.floating)

        # prepare  target / labels / y

        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        if self.explicit_target_class is not None:
            self.target_class_ = self.explicit_target_class
        else:
            self.target_class_ = self.classes_[0]

        # prepare  attributes / features / X
        self.n_features_ = X.shape[1]
        # categorical_features modeled like sklearn.preprocessing.OneHotEncoder
        self.categorical_mask_ = np.zeros(self.n_features_, dtype=bool)
        if (self.categorical_features is None) or \
                not len(self.categorical_features):
            pass  # keep default "all False"
        elif isinstance(self.categorical_features, np.ndarray):
            self.categorical_mask_[
                np.asarray(self.categorical_features)] = True
        elif self.categorical_features == 'all':
            self.categorical_mask_ = np.ones(self.n_features_, dtype=bool)
        else:
            raise ValueError("categorical_features must be one of: None,"
                             " 'all', np.ndarray of dtype bool or integer,"
                             " but got {}.".format(self.categorical_features))

        # run SeCo algorithm
        self.theory_ = np.array(self.abstract_seco(X, y))
        return self

    def find_best_rule(self, context: 'RuleContext') -> 'AugmentedRule':
        """Inner loop of abstract SeCo/Covering algorithm."""

        # resolve methods once for performance
        init_rule = self.implementation.init_rule
        evaluate_rule = context.evaluate_rule
        select_candidate_rules = self.implementation.select_candidate_rules
        refine_rule = self.implementation.refine_rule
        inner_stopping_criterion = self.implementation.inner_stopping_criterion
        filter_rules = self.implementation.filter_rules

        # algorithm
        best_rule = init_rule(context)
        evaluate_rule(best_rule)
        rules: RuleQueue = [best_rule]
        while len(rules):
            for candidate in select_candidate_rules(rules, context):
                # TODO: parallelize here:
                for refinement in refine_rule(candidate, context):
                    context.evaluate_rule(refinement)
                    if not inner_stopping_criterion(refinement, context):
                        rules.append(refinement)
                        if best_rule < refinement:
                            best_rule = refinement
            rules.sort()
            rules = filter_rules(rules, context)
        return best_rule

    def abstract_seco(self, X: np.ndarray, y: np.ndarray) -> 'Theory':
        """Main loop of abstract SeCo/Covering algorithm."""

        target_class = self.target_class_
        theory_context = self.implementation.theory_context_class(
            self.implementation,
            self.categorical_mask_, self.n_features_, target_class, X, y)

        # resolve methods once for performance
        make_rule_context = self.implementation.rule_context_class
        find_best_rule = self.find_best_rule
        simplify_rule = self.implementation.simplify_rule
        rule_stopping_criterion = self.implementation.rule_stopping_criterion

        # main loop
        theory: Theory = list()
        while np.any(y == target_class):
            rule_context = make_rule_context(theory_context, X, y)
            rule = find_best_rule(rule_context)
            # TODO: ensure grow-rating is not used in pruning. use property & override in GrowPruneSplit ?
            rule = simplify_rule(rule, rule_context)
            if rule_stopping_criterion(theory, rule, rule_context):  # TODO: use pruning or growing+pruning?
                break
            # ignore the rest of theory, because it already covered
            uncovered = np.invert(rule.match(rule_context))
            X = X[uncovered]  # TODO: use mask array instead of copy?
            y = y[uncovered]
            theory.append(rule.conditions)  # throw away augmentation
        return self.implementation.post_process(theory)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['theory_', 'categorical_mask_'])
        X: np.ndarray = check_array(X)
        target_class = self.target_class_
        match_rule = self.implementation.match_rule
        result = np.repeat(self.classes_[1],  # negative class  # FIXME: not if self.explicit_target_class
                           X.shape[0])

        for rule in self.theory_:
            result = np.where(
                match_rule(X, rule, self.categorical_mask_),
                target_class,
                result)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prediction = self.predict(X) == self.target_class_
        return np.where(prediction[:, np.newaxis],
                        # TODO: use rule metric as probability
                        np.array([[1, 0]]),
                        np.array([[0, 1]]))


# noinspection PyAttributeOutsideInit
class SeCoEstimator(BaseEstimator, ClassifierMixin):
    """Wrap the base SeCo to provide class label binarization."""

    def __init__(self, implementation: 'AbstractSecoImplementation',
                 multi_class="one_vs_rest", n_jobs=1):
        self.implementation = implementation
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwargs):
        # TODO: document available kwargs or link `_BinarySeCoEstimator.fit`
        X, y = check_X_y(X, y, multi_output=False)

        self.base_estimator_ = _BinarySeCoEstimator(self.implementation,
                                                    **kwargs)

        # copied from GaussianProcessClassifier
        self.classes_ = np.unique(y)
        n_classes_ = self.classes_.size
        if n_classes_ == 1:
            raise ValueError("SeCoEstimator requires 2 or more distinct "
                             "classes. Only 1 class (%s) present."
                             % self.classes_[0])
        elif n_classes_ > 2:
            # TODO: multi_class strategy of ripper: OneVsRest, remove C_i after learning rules for it
            if self.multi_class == "one_vs_rest":
                self.base_estimator_.set_params(explicit_target_class=1)
                self.base_estimator_ = OneVsRestClassifier(self.base_estimator_,
                                                           n_jobs=self.n_jobs)
            elif self.multi_class == "one_vs_one":
                # TODO: tell _BinarySeCoEstimator about classes, i.e. which is positive
                self.base_estimator_ = OneVsOneClassifier(self.base_estimator_,
                                                          n_jobs=self.n_jobs)
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class)

        self.base_estimator_.fit(X, y)
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        check_is_fitted(self, ["classes_"])
        X = check_array(X)
        return self.base_estimator_.predict(X)


# imports needed only for type checking, place here to break circularity
from sklearn_seco.common import \
    AugmentedRule, RuleQueue, Theory, \
    AbstractSecoImplementation, RuleContext
