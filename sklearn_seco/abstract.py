"""
Implementation of SeCo / Covering algorithm: Abstract base algorithm.
"""

from typing import Union, Type

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from sklearn_seco.common import \
    AugmentedRule, RuleQueue, Theory, \
    RuleContext, SeCoAlgorithmConfiguration


# noinspection PyAttributeOutsideInit
class _BinarySeCoEstimator(BaseEstimator, ClassifierMixin):
    """Binary SeCo Classification, deferring to :var:`algorithm_config`
    for concrete algorithm implementation.

    :param algorithm_config: Type[SeCoAlgorithmConfiguration]
        Defines the SeCo variant to be run.

    :param categorical_features: None or "all" or array of indices or mask.

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
                 algorithm_config: Type['SeCoAlgorithmConfiguration'],
                 categorical_features: Union[None, str, np.ndarray] = None,
                 explicit_target_class=None):
        super().__init__()
        self.algorithm_config = algorithm_config
        self.categorical_features = categorical_features
        self.explicit_target_class = explicit_target_class

    def fit(self, X, y):
        """Fit to data, i.e. learn theory

        :param X: Not yet covered examples.
        :param y: Classification for `X`.
        """
        X, y = check_X_y(X, y, dtype=np.floating)
        assert self.algorithm_config

        # prepare  target / labels / y
        check_classification_targets(y)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        if self.explicit_target_class is not None:
            self.target_class_ = self.explicit_target_class
        else:
            self.target_class_ = self.classes_[np.argmax(class_counts)]

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
        if len(self.theory_) and np.all(~ np.isfinite(self.theory_)):
            # an empty theory is learned when the default rule is already very
            # good (i.e. the target class has very high a priori probability)
            # therefore the case of only empty rules is superfluous
            raise ValueError("Invalid theory learned")
        return self

    def find_best_rule(self, context: 'RuleContext') -> 'AugmentedRule':
        """Inner loop of abstract SeCo/Covering algorithm."""

        # resolve methods once for performance
        implementation = self.algorithm_config.Implementation
        init_rule = implementation.init_rule
        evaluate_rule = context.evaluate_rule
        select_candidate_rules = implementation.select_candidate_rules
        refine_rule = implementation.refine_rule
        inner_stopping_criterion = implementation.inner_stopping_criterion
        filter_rules = implementation.filter_rules

        # algorithm
        best_rule = init_rule(context)
        evaluate_rule(best_rule)
        rules: RuleQueue = [best_rule]
        while len(rules):
            for candidate in select_candidate_rules(rules, context):
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

        theory_context = self.algorithm_config.TheoryContextClass(
            self.algorithm_config,
            self.categorical_mask_, self.n_features_, target_class, X, y)

        # resolve methods once for performance
        implementation = self.algorithm_config.Implementation
        make_rule_context = self.algorithm_config.RuleContextClass
        find_best_rule = self.find_best_rule
        simplify_rule = implementation.simplify_rule
        rule_stopping_criterion = implementation.rule_stopping_criterion
        post_process = implementation.post_process

        # main loop
        theory: Theory = list()
        while np.any(y == target_class):
            rule_context = make_rule_context(theory_context, X, y)
            rule = find_best_rule(rule_context)
            rule = simplify_rule(rule, rule_context)
            if rule_stopping_criterion(theory, rule, rule_context):  # TODO: use pruning or growing+pruning?
                break
            uncovered = np.invert(
                rule_context.match_rule(rule, force_X_complete=True))
            X = X[uncovered]
            y = y[uncovered]
            theory.append(rule.conditions)  # throw away augmentation
        theory = post_process(theory, theory_context)

        # store growing_heuristic(training set) for decision_function
        rule_context = RuleContext(theory_context,
                                   theory_context.complete_X,
                                   theory_context.complete_y)
        self.confidence_estimates_ = [
            self.algorithm_config.Implementation.growing_heuristic(
                AugmentedRule(conditions=rule), rule_context)
            for rule in theory
        ]
        return theory

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['theory_', 'categorical_mask_'])
        X: np.ndarray = check_array(X)
        target_class = self.target_class_
        negative_class = self.classes_[self.classes_ != target_class][0]
        match_rule = self.algorithm_config.match_rule
        rule_results = \
            np.array([match_rule(X, rule, self.categorical_mask_)
                      for rule in self.theory_]
                     ) \
            .any(axis=0) \
            .astype(type(target_class))  # any of the rules matched
        # translate bool to class value
        rule_results[rule_results == True] = target_class  # noqa: E712
        rule_results[rule_results == False] = negative_class  # noqa: E712
        return rule_results

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # used by `sklearn.utils.multiclass._ovr_decision_function`
        check_is_fitted(self, ['theory_', 'categorical_mask_'])
        X: np.ndarray = check_array(X)
        match_rule = self.algorithm_config.match_rule
        proba = np.zeros(X.shape[0])
        for i, rule in enumerate(self.theory_):  # TODO: matrix multiplication?
            proba = np.where(  # TODO: later rule matches override previous ones
                match_rule(X, rule, self.categorical_mask_),
                self.confidence_estimates_[i],
                proba
            )
        return proba


# noinspection PyAttributeOutsideInit
class SeCoEstimator(BaseEstimator, ClassifierMixin):
    """Wrap the base SeCo to provide class label binarization.

    The concrete SeCo variant to run is defined by `algorithm_config`.
    """

    algorithm_config: Type[SeCoAlgorithmConfiguration]

    def __init__(self, multi_class="one_vs_rest", n_jobs=1):
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwargs):
        # TODO: document available kwargs or link `_BinarySeCoEstimator.fit`
        X, y = check_X_y(X, y, multi_output=False)

        self.base_estimator_ = _BinarySeCoEstimator(self.algorithm_config,
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
