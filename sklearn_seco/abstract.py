"""
Implementation of SeCo / Covering algorithm: Abstract base algorithm.
"""

from typing import Union, Type, List

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from sklearn_seco.common import \
    AugmentedRule, RuleQueue, Theory, \
    RuleContext, SeCoAlgorithmConfiguration, \
    rule_to_string


# noinspection PyAttributeOutsideInit
class _BinarySeCoEstimator(BaseEstimator, ClassifierMixin):
    """Binary SeCo Classification, deferring to :var:`algorithm_config_`
    for concrete algorithm implementation.

    :param algorithm_config_: SeCoAlgorithmConfiguration
        Defines the SeCo variant to be run. Instance of `class
        self.algorithm_config_class`.

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

    :param explicit_target_class: Use as positive/target class for learning. If
        `None` (the default), use the first class from `np.unique(y)` (which is
        sorted).
    """

    def __init__(self,
                 algorithm_config_class: Type['SeCoAlgorithmConfiguration'],
                 categorical_features: Union[None, str, np.ndarray] = None,
                 explicit_target_class=None):
        super().__init__()
        self.algorithm_config_class = algorithm_config_class
        self.categorical_features = categorical_features
        self.explicit_target_class = explicit_target_class

    def fit(self, X, y):
        """Fit to data, i.e. learn theory

        :param X: Not yet covered examples.
        :param y: Classification for `X`.
        """
        X, y = check_X_y(X, y, dtype=np.floating)
        self.algorithm_config_ = self.algorithm_config_class()

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
        self.theory_: Theory = self.abstract_seco(X, y)
        if len(self.theory_) and all((~ np.isfinite(rule.body).any()
                                      for rule in self.theory_)):
            # an empty theory is learned when the default rule is already very
            # good (i.e. the target class has very high a priori probability)
            # therefore the case of only empty rules is an error
            raise ValueError("Invalid theory learned")
        return self

    def find_best_rule(self, context: 'RuleContext') -> 'AugmentedRule':
        """Inner loop of abstract SeCo/Covering algorithm."""

        # resolve methods once for performance
        implementation = self.algorithm_config_.implementation
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

        theory_context = self.algorithm_config_.make_theory_context(
            self.categorical_mask_, self.n_features_, X, y)

        # resolve methods once for performance
        implementation = self.algorithm_config_.implementation
        make_rule_context = self.algorithm_config_.make_rule_context
        find_best_rule = self.find_best_rule
        simplify_rule = implementation.simplify_rule
        rule_stopping_criterion = implementation.rule_stopping_criterion
        post_process = implementation.post_process
        confidence_estimator = implementation.confidence_estimate

        # main loop
        theory: Theory = list()
        while np.any(y == target_class):
            rule_context = make_rule_context(theory_context, X, y)
            rule = find_best_rule(rule_context)
            rule = simplify_rule(rule, rule_context)
            if rule_stopping_criterion(theory, rule, rule_context):
                break
            uncovered = np.invert(
                rule_context.match_rule(rule, force_complete_data=True))
            X = X[uncovered]
            y = y[uncovered]
            theory.append(rule.raw)  # throw away augmentation
        theory = post_process(theory, theory_context)

        # store growing_heuristic(training set) for decision_function
        rule_context = make_rule_context(theory_context,
                                         theory_context.complete_X,
                                         theory_context.complete_y)
        self.confidence_estimates_ = [
            confidence_estimator(AugmentedRule(conditions=rule), rule_context)
            for rule in theory
        ]
        # TODO: ? for confidence_estimate use *uncovered by theory[i]* instead of whole X to match theory[i+1]
        # TODO: ? confidence_estimate for default rule (i.e. not any rule from theory matches). not compatible with current confidence_estimate(rule, RuleContext) interface
        return theory

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make a prediction for each sample in `X`.

        See <https://scikit-learn.org/dev/glossary.html#term-predict>
        """
        check_is_fitted(self, ['theory_', 'categorical_mask_'])
        X: np.ndarray = check_array(X)
        target_class = self.target_class_
        negative_class = self.classes_[self.classes_ != target_class][0]
        match_rule = self.algorithm_config_.match_rule
        matches = (np.array([match_rule(X, rule, self.categorical_mask_)
                             for rule in self.theory_]
                            # note: samples are columns at this point
                            )
                   .reshape((len(self.theory_), len(X)))  # if theory empty
                   .any(axis=0)  # any of the rules matched
                   )
        # translate bool to class value
        return np.where(matches, target_class, negative_class)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Predict a “soft” score for each sample, values strictly greater than
        zero indicate the positive class.

        Used by `sklearn.multiclass.OneVsRestClassifier` through
        `sklearn.utils.multiclass._ovr_decision_function`.

        See <https://scikit-learn.org/dev/glossary.html#term-decision-function>
        """
        check_is_fitted(self, ['theory_', 'categorical_mask_'])
        X: np.ndarray = check_array(X)
        match_rule = self.algorithm_config_.match_rule
        confidence = np.column_stack((
            self.confidence_estimates_ *
            np.transpose([match_rule(X, rule.body, self.categorical_mask_)
                         for rule in self.theory_])
                .reshape((len(X), len(self.theory_))),
            np.zeros((len(X), 1))  # if theory empty
        ))
        # for ordered: get leftmost nonzero value i.e. first matched rule
        # for unordered, would just do confidence.max(axis=1)
        return confidence[range(len(confidence)),
                          np.argmax(confidence > 0, axis=1)]

    def export_text(self, feature_names: List[str] = None,
                    target_class: str = None) -> str:
        """Build a text report showing the rules in the learned theory.

        See Also `sklearn.tree.export_tree`

        Parameters
        -----
        feature_names : list, optional (default=None)
            A list of length n_features containing the feature names.
            If None generic names will be used ("feature_0", "feature_1", ...).

        target_class: str, optional
            A string representation of the target class. If None, the value of
            self.target_class_ is used.
        """

        if feature_names:
            if len(feature_names) != self.n_features_:
                raise ValueError(
                    "feature_names must contain %d elements, got %d"
                    % (self.n_features_, len(feature_names)))
        else:
            feature_names = ["feature_{}".format(i)
                             for i in range(self.n_features_)]

        if not target_class:
            target_class = self.target_class_

        negative_class = self.classes_[self.classes_ != target_class][0]
        default_rule = '(true) => ' + str(negative_class)
        return '\n'.join([
            rule_to_string(rule, self.categorical_mask_, feature_names)
            for rule in self.theory_] + [default_rule])


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
