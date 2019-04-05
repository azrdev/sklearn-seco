"""
Implementation of SeCo / Covering algorithm: Abstract base algorithm.
"""
import warnings
from typing import Union, Type, List

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from sklearn_seco.common import \
    AugmentedRule, RuleQueue, Theory, \
    RuleContext, SeCoAlgorithmConfiguration


# noinspection PyAttributeOutsideInit
class _BinarySeCoEstimator(BaseEstimator, ClassifierMixin):
    """Binary SeCo Classification, deferring to :var:`algorithm_config_`
    for concrete algorithm implementation.

    Parameters
    -----
    algorithm_config_class : subclass of SeCoAlgorithmConfiguration
        Defines the SeCo variant to be run. See `algorithm_config_`.

    categorical_features : None or "all" or array of indices or mask.

        Specify what features are treated as categorical, i.e. equality
        tests are used for these features, based on the set of values
        present in the training data.

        Note that numerical features may be tested in multiple (inequality)
        conditions of a rule, while multiple equality tests (for a
        categorical feature) would be useless.

        - None (default): All features are treated as numerical & ordinal.
        - 'all': All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.

        You may instead transform your categorical features beforehand,
        using e.g. :class:`sklearn.preprocessing.OneHotEncoder` or
        :class:`sklearn.preprocessing.Binarizer`.

    explicit_target_class :
        Use as positive/target class for learning. If `None` (the default),
        use the most prevalent class. `fit` fails is this value is not in `y`.

    ordered_matching : bool
        If True (the default), learn & use the theory as an ordered rule list,
        i.e. a match of one rule shadows any possible match of later rules.
        If False, learn & use it as an unordered rule set. Particularly for
        prediction, this means that the rule with the highest confidence
        estimate will "win" among the ones covering a sample (ties are broken
        by favoring bigger classes).

    remove_false_positives : bool or None
        Relevant while rule learning in `abstract_seco`:
        If True, remove all examples covered by the just learned rule.
        If False, remove only the true positives.
        If None, set `remove_false_positives=ordered_matching`.

    Attributes
    -----
    algorithm_config_ : SeCoAlgorithmConfiguration
        Defines the SeCo variant to be run. Instance of `class
        self.algorithm_config_class`, to allow wrapping its methods in e.g.
        `functools.partialmethod`.

    is_binary_ : bool
        True iff the learned problem (the training data) was binary, False if
        multiclass (`is_binary_ = len(self.classes_) == 2`).

    classes_ : np.ndarray
        A list of class labels known to the classifier, implicitly mapping each
        label to a numerical index.

    class_counts_ : np.ndarray of the same shape as classes_
        A mapping from class index to count of examples in training data, i.e.
        the class distribution.

    classes_by_size_ : np.ndarray of the same shape as classes_
        A list of indices into `classes_`, sorted by size.

    target_class_idx_ : int, optional
        Index into `classes_` specifying the target class. If a multiclass
        problem is passed to `fit`, a multiclass theory is directly learned
        (i.e. there is no target class), and this attribute is None.

    n_features_ : int
        The number of features in (training) data `X`.

    categorical_mask_ : np.ndarray of shape (n_features_,) and dtype bool
        A mask array calculated from `categorical_features`. True entries
        denote categorical features, False entries numeric ones.

    theory_ : Theory (i.e. List[Rule])
        The learned list of `Rule`s. Interpreted according to
        `categorical_mask_` and `ordered_matching`.

    confidence_estimates_ : Sequence[float] of same length as theory_
        Represent for each rule in theory_ a confidence in its predictions.
        Used by `decision_function` as score, and therefore (if
        `ordered_matching == False`) by `predict` for voting.
    """

    def __init__(self,
                 algorithm_config_class: Type['SeCoAlgorithmConfiguration'],
                 categorical_features: Union[None, str, np.ndarray] = None,
                 explicit_target_class=None, ordered_matching: bool = True,
                 remove_false_positives: bool = None):
        super().__init__()
        self.algorithm_config_class = algorithm_config_class
        self.categorical_features = categorical_features
        self.explicit_target_class = explicit_target_class
        self.ordered_matching = ordered_matching
        self.remove_false_positives = remove_false_positives

    def fit(self, X, y):
        """Fit to data, i.e. learn theory

        :param X: Not yet covered examples.
        :param y: Classification for `X`.
        """
        X, y = check_X_y(X, y, dtype=np.floating)
        self.algorithm_config_ = self.algorithm_config_class()

        # prepare  target / labels / y
        check_classification_targets(y)
        self.classes_, self.class_counts_ = np.unique(y, return_counts=True)
        self.classes_by_size_ = np.argsort(self.class_counts_)
        self.is_binary_ = len(self.classes_) == 2
        if self.is_binary_:
            if self.explicit_target_class is None:
                # binary target: most prevalent class
                self.target_class_idx_ = np.argmax(self.class_counts_)
                # NOTE: for binary targets we always do concept learning, i.e.
                #   all rules have the same (target_class) head, and only the
                #   "fallback" default rule classifies as the other "negative"
                #   class.
            else:
                # binary target explicitly specified
                target_class_idx = np.argwhere(
                    self.explicit_target_class == self.classes_)
                if not target_class_idx.size:
                    raise ValueError("explicit_target_class {!s} not in y"
                                     .format(self.explicit_target_class))
                self.target_class_idx_ = np.take(target_class_idx, 0)
        else:  # multiclass
            assert self.algorithm_config_.direct_multiclass_support()
            if self.explicit_target_class is not None:
                # NOTE: we cannot learn a single concept on a multiclass
                #   problem: which of the other classes would be the "negative"
                #   default class? Binarize your problem beforehand if you need
                #   that behavior.
                warnings.warn("explicit_target_class set on multiclass "
                              "problem. Ignoring explicit_target_class.")

            # learning multiclass directly, no global target
            self.target_class_idx_ = None  # implicit default class = biggest

        # prepare  attributes / features / X
        self.n_features_ = X.shape[1]
        # categorical_features modeled like sklearn.preprocessing.OneHotEncoder
        self.categorical_mask_ = np.zeros(self.n_features_, dtype=bool)
        if (self.categorical_features is None
                or not len(self.categorical_features)):
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
        if len(self.theory_):
            # an empty theory is learned when the default rule is already very
            # good (i.e. the target class has very high a priori probability)
            if all(rule.body_empty() for rule in self.theory_):
                # therefore only the case of only empty rules is an error
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

        theory_context = self.algorithm_config_.make_theory_context(
            self.categorical_mask_, self.n_features_, self.classes_,
            self.class_counts_, self.classes_by_size_, self.target_class_idx_,
            X, y)
        remove_false_positives = (
            self.remove_false_positives
            if self.remove_false_positives is not None
            else self.ordered_matching)

        # resolve methods once for performance
        implementation = self.algorithm_config_.implementation
        abstract_seco_continue = implementation.abstract_seco_continue
        make_rule_context = self.algorithm_config_.make_rule_context
        find_best_rule = self.find_best_rule
        simplify_rule = implementation.simplify_rule
        rule_stopping_criterion = implementation.rule_stopping_criterion
        post_process = implementation.post_process
        confidence_estimator = implementation.confidence_estimate

        # main loop
        theory: Theory = list()
        while abstract_seco_continue(y, theory_context):
            rule_context = make_rule_context(theory_context, X, y)
            rule = find_best_rule(rule_context)
            rule = simplify_rule(rule, rule_context)
            if rule_stopping_criterion(theory, rule, rule_context):
                break
            uncovered = np.invert(
                rule_context.match_rule(rule, force_complete_data=True))
            if not remove_false_positives:
                uncovered[y != rule.head] = True  # keep false positives
            X = X[uncovered]
            y = y[uncovered]
            theory.append(rule.raw)  # throw away augmentation
        theory = post_process(theory, theory_context)

        # store growing_heuristic(training set) for decision_function
        rule_context = make_rule_context(theory_context,
                                         theory_context.complete_X,
                                         theory_context.complete_y)
        self.confidence_estimates_ = np.array([
            confidence_estimator(AugmentedRule(conditions=rule), rule_context)
            for rule in theory
        ])
        # TODO: ? for confidence_estimate use *uncovered by theory[i]* instead of whole X to match theory[i+1]
        # TODO: ? confidence_estimate for default rule (i.e. not any rule from theory matches). not compatible with current confidence_estimate(rule, RuleContext) interface
        return theory

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make a prediction for each sample in `X`.

        See <https://scikit-learn.org/dev/glossary.html#term-predict>
        """
        check_is_fitted(self, ['theory_', 'categorical_mask_',
                               'confidence_estimates_'])
        if self.is_binary_:
            cl: List = self.classes_.tolist()
            positive = cl.pop(self.target_class_idx_)
            negative = cl.pop()  # default class
            return np.where(self.decision_function(X), positive, negative)
        else:
            # multi-class case: break ties by size (default class = biggest)
            classidx_by_size_rev = self.classes_by_size_[::-1]
            classes_by_size_rev = self.classes_[classidx_by_size_rev]
            decisions = self.decision_function(X)
            return classes_by_size_rev[
                # argmax uses first to break ties, thus order by size reversed
                decisions[:, classidx_by_size_rev].argmax(axis=1)]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Predict a "soft" score for each sample.

        :return: np.ndarray
          Contains for each sample and each class a floating value representing
          confidence that the sample is of that class.

          If binary, shape is `(n_samples,)` and values strictly greater than
          zero indicate the positive class.

          If multiclass, shape is `(n_samples, n_classes)`.

        Used by `sklearn.multiclass.OneVsRestClassifier` through
        `sklearn.utils.multiclass._ovr_decision_function`.

        See <https://scikit-learn.org/dev/glossary.html#term-decision-function>
        """
        check_is_fitted(self, ['theory_', 'categorical_mask_',
                               'confidence_estimates_'])
        X: np.ndarray = check_array(X)
        match_rule = self.algorithm_config_.match_rule
        # TODO: valid if any(confidence_estimates_ < 0) ?
        confidence_by_class = np.zeros((len(X), len(self.classes_)))
        for rule, rule_confidence in zip(
                # backwards so earlier rules overwrite later ones.
                # needed if ordered theory
                reversed(self.theory_),
                reversed(self.confidence_estimates_)):
            class_index = np.take(np.argwhere(rule.head == self.classes_), 0)
            matches = match_rule(X, rule, self.categorical_mask_)
            if self.ordered_matching:
                # if ordered, get leftmost match; i.e. overwrite if match
                confidence_by_class[matches, class_index] = rule_confidence
            else:
                # if unordered, get the max confidence among matching rules
                confidence_by_class[:, class_index] = np.max(
                    confidence_by_class[:, class_index],
                    matches * rule_confidence)

        if self.is_binary_:
            # specified by sklearn: only return one column, for positive class
            assert self.target_class_idx_ is not None
            return confidence_by_class[:, self.target_class_idx_]
        return confidence_by_class

    def export_text(self, feature_names: List[str] = None,
                    class_names: List[str] = None) -> str:
        """Build a text report showing the rules in the learned theory.

        See Also `sklearn.tree.export_tree`

        Parameters
        -----
        feature_names : list, optional
            A list of length n_features containing the feature names.
            If None, generic names will be generated.

        class_names: list, optional
            A list of length n_classes containing the class names, ordered like
            `self.classes_`. If None, indices will be used.
        """

        if feature_names:
            if len(feature_names) != self.n_features_:
                raise ValueError(
                    "feature_names must contain %d elements, got %d"
                    % (self.n_features_, len(feature_names)))
        else:
            feature_names = ["feature_{}".format(i + 1)
                             for i in range(self.n_features_)]

        if self.is_binary_:
            # in binary, fallback to the negative class
            negative_idx = (set(self.classes_by_size_)
                            - {self.target_class_idx_}).pop()
            default_class = self.classes_[negative_idx]
        else:
            # in multiclass, fallback to the biggest class
            default_class = self.classes_[self.classes_by_size_[-1]]
        if class_names:
            default_class = class_names[default_class]
        default_rule = '(true) => ' + str(default_class)

        return '\n'.join([
            rule.to_string(self.categorical_mask_, feature_names, class_names)
            for rule in self.theory_] + [default_rule])


# noinspection PyAttributeOutsideInit
class SeCoEstimator(BaseEstimator, ClassifierMixin):
    """Wrap the base SeCo to provide class label binarization.

    The concrete SeCo variant to run is defined by `algorithm_config`.

    Fields
    -----
    algorithm_config : subclass of SeCoAlgorithmConfiguration
        Defines the concrete SeCo algorithm to run, see
        :class:`SeCoAlgorithmConfiguration`.

    Parameters
    -----
    multi_class : callable or str or None
        Which strategy to use for non-binary problems. Possible values:

        - None: auto-select; use 'direct' if possible
          (`algorithm_config.direct_multiclass_support()` returns True),
          'one_vs_rest' otherwise.
        - A callable: Construct
          `self.base_estimator_ = multi_class(_BinarySeCoEstimator())` and
          delegate to that estimator. Useful if you want to roll a different
          binarization strategy, e.g.
          >>> multi_class=partial(sklearn.multiclass.OutputCodeClassifier,
                                  code_size=0.7, random_state=42)
        - 'direct': Directly learn a theory of rules with different heads
          (target classes).
        - 'one_vs_rest': Use `sklearn.multiclass.OneVsRestClassifier` for class
          binarization and learn binary theories.
        - 'one_vs_one': Use `sklearn.multiclass.OneVsOneClassifier` for class
          binarization and learn binary theories.

    n_jobs : int, optional
        Passed to `OneVsRestClassifier` or `OneVsOneClassifier` if these are
        used.

    Attributes
    -----
    base_estimator_ : estimator instance
        The estimator object that all tasks are delegated to. A
        `sklearn.multiclass.OneVsRestClassifier` or
        `sklearn.multiclass.OneVsOneClassifier` if demanded by the
        `multi_class_` strategy, a `_BinarySeCoEstimator` otherwise.

    multi_class_ : callable or str
        The actual strategy used on a non-binary problem. Relevant if
        `multi_class=None` demanded auto-selection.

    classes_ : np.ndarray
        `np.unique(y)`

    """

    algorithm_config: Type[SeCoAlgorithmConfiguration]

    def __init__(self, multi_class=None, n_jobs=1):
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwargs):
        """Learn SeCo theory/theories on training data `X, y`.

        For possible parameters (`**kwargs`), refer to
        :class:`_BinarySeCoEstimator`.
        """
        X, y = check_X_y(X, y, multi_output=False)
        self.multi_class_ = self.multi_class
        self.base_estimator_ = _BinarySeCoEstimator(self.algorithm_config,
                                                    **kwargs)

        self.classes_ = np.unique(y)
        n_classes_ = self.classes_.size
        if n_classes_ == 1:
            raise ValueError("SeCoEstimator requires 2 or more distinct "
                             "classes. Only 1 class (%s) present."
                             % self.classes_[0])
        elif n_classes_ > 2:
            # TODO: multi_class strategy of ripper: OneVsRest, remove C_i after learning rules for it
            if self.multi_class_ is None:
                # default / auto-selection
                if self.algorithm_config.direct_multiclass_support():
                    self.multi_class_ = "direct"
                else:
                    self.multi_class_ = "one_vs_rest"

            if callable(self.multi_class_):
                self.base_estimator_ = self.multi_class_(self.base_estimator_)
            elif self.multi_class_ == "one_vs_rest":
                self.base_estimator_.set_params(explicit_target_class=1)
                self.base_estimator_ = OneVsRestClassifier(self.base_estimator_,
                                                           n_jobs=self.n_jobs)
            elif self.multi_class_ == "one_vs_one":
                self.base_estimator_ = OneVsOneClassifier(self.base_estimator_,
                                                          n_jobs=self.n_jobs)
            elif self.multi_class_ == "direct":
                pass
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class_)

        # TODO: maybe move explicit_target_class,categorical_features here from __init__, since they're data dependent
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

    @if_delegate_has_method('base_estimator_')
    def predict_proba(self, X):
        # noinspection PyUnresolvedReferences
        return self.base_estimator_.predict_proba(X)

    @if_delegate_has_method('base_estimator_')
    def decision_function(self, X):
        # noinspection PyUnresolvedReferences
        return self.base_estimator_.decision_function(X)
