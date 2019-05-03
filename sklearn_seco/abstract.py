"""
Implementation of SeCo / Covering algorithm: Abstract base algorithm.
"""

from typing import Union, Type, List, Sequence

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from sklearn_seco.common import \
    AugmentedRule, RuleQueue, Theory, \
    RuleContext, SeCoAlgorithmConfiguration
from sklearn_seco.util import \
    BySizeLabelEncoder, TargetTransformingMetaEstimator


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

    classes_ : np.ndarray
        A list of class labels known to the classifier, sorted
        lexicographically. The last class (`classes_[-1]`) is the positive
        class for binary problems. The first class is the default (for
        multiclass problems) resp. negative (for binary problems) class.

        To conform with sklearn assumptions, `predict` and `decision_function`
        break ties by preferring lower values in classes_, thus to prefer more
        common (in the training data) classes, make sure that class order by
        size always matches lexicographic class order. `SeCoEstimator`
        does so.

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

    See Also
    -----
    `SeCoEstimator`
        Wrapper handling multi-class strategies, use instead of this.

    `SeCoAlgorithmConfiguration`
        Defining the SeCo variant that is executed.
    """

    def __init__(self,
                 algorithm_config_class: Type['SeCoAlgorithmConfiguration'],
                 categorical_features: Union[None, str, np.ndarray] = None,
                 ordered_matching: bool = True,
                 remove_false_positives: bool = None):
        super().__init__()
        self.algorithm_config_class = algorithm_config_class
        self.categorical_features = categorical_features
        self.ordered_matching = ordered_matching
        self.remove_false_positives = remove_false_positives

    def fit(self, X, y):
        """Fit to data, i.e. learn theory

        :param X: Not yet covered examples.
        :param y: Classification labels for `X`. They are assumed to be ordered
            by size.
        """
        X, y = check_X_y(X, y, dtype=np.floating)
        self.algorithm_config_ = self.algorithm_config_class()

        # prepare  target / labels / y
        self.classes_ = unique_labels(y)

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

    def is_binary(self):
        """:return: True iff the learned problem (the training data) was
        binary, False if multi-class (`is_binary_ = len(self.classes_) == 2`).
        """
        check_is_fitted(self, 'classes_')
        return len(self.classes_) == 2

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
        sort_key = context.sort_key

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
                        if sort_key(best_rule) < sort_key(refinement):
                            best_rule = refinement
            rules.sort(key=sort_key)
            rules = filter_rules(rules, context)
        return best_rule

    def abstract_seco(self, X: np.ndarray, y: np.ndarray) -> 'Theory':
        """Main loop of abstract SeCo/Covering algorithm."""

        theory_context = self.algorithm_config_.make_theory_context(
            self.categorical_mask_, self.n_features_, self.classes_,
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
        check_is_fitted(self, ['classes_'])
        if self.is_binary():
            return np.where(self.decision_function(X),
                            self.classes_[1],  # positive
                            self.classes_[0])  # negative
        else:
            return self.classes_[np.argmax(self.decision_function(X),
                                           axis=1)]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Predict a "soft" score for each sample.

        :return: np.ndarray
          Contains for each sample and each class a floating value representing
          confidence that the sample is of that class.

          If binary, shape is `(n_samples,)` and values strictly greater than
          zero indicate the positive class.

          If multi-class, shape is `(n_samples, n_classes)`.

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

        if self.is_binary():
            # specified by sklearn: only return positive class, as 1d array
            return confidence_by_class[:, 1].ravel()
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

        # fallback to the negative (binary) resp. biggest (multi-class) class
        default_class = self.classes_[0]
        if class_names:
            default_class = class_names[default_class]
        default_rule = '(true) => ' + str(default_class)

        return '\n'.join([
            rule.to_string(self.categorical_mask_, feature_names, class_names)
            for rule in self.theory_] + [default_rule])


# noinspection PyAttributeOutsideInit
class SeCoEstimator(BaseEstimator, ClassifierMixin):
    """A classifier using rules learned with the *Separate-and-Conquer* (SeCo)
    algorithm, also known as *Covering* algorithm.

    Wraps `_BinarySeCoEstimator` to handle multi-class problems, selecting a
    multi-class strategy and making sure that _BinarySeCoEstimator always sees
    class labels where the lexicographically smallest/first is the intended
    fallback class; i.e. the biggest class in multi-class problems, or the
    negative class when learning a binary concept.

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
          >>> import sklearn.multiclass, functools
          >>> multi_class=functools.partial(
          ...     sklearn.multiclass.OutputCodeClassifier,
          ...     code_size=0.7, random_state=42)
          If you use this, be aware of class order influence on tie-breaking.
        - 'direct': Directly learn a theory of rules with different heads
          (target classes). Uses :class:`BySizeLabelEncoder` internally.
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
        The estimator object that all tasks are delegated to. One of
        `sklearn.multiclass.OneVsRestClassifier`,
        `sklearn.multiclass.OneVsOneClassifier` or
        `sklearn_seco.util.TargetTransformingMetaEstimator` if demanded by the
        `multi_class_` strategy, a `_BinarySeCoEstimator` otherwise.

    multi_class_ : callable or str
        The actual strategy used on a non-binary problem. Relevant if
        `multi_class=None` demanded auto-selection.

    classes_ : np.ndarray
        `np.unique(y)`

    See Also
    -----
    `_BinarySeCoEstimator`
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
        # FIXME: categorical_features contains class order, has to go through BySizeLabelEncoderm, too

        def wrapper_ordering_classes_by_size(estimator):
            # BySizeLabelEncoder ensures:  first class = default = biggest
            return TargetTransformingMetaEstimator(BySizeLabelEncoder(),
                                                   estimator)

        self.classes_ = np.unique(y)
        n_classes_ = self.classes_.size
        if n_classes_ == 1:
            raise ValueError("SeCoEstimator requires 2 or more distinct "
                             "classes. Only 1 class (%s) present."
                             % self.classes_[0])
        elif n_classes_ == 2:
            # BySizeLabelEncoder ensures:  first class = default = biggest
            self.base_estimator_ = wrapper_ordering_classes_by_size(
                self.base_estimator_)
        else:  # n_classes_ > 2
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
                self.base_estimator_ = OneVsRestClassifier(self.base_estimator_,
                                                           n_jobs=self.n_jobs)
            elif self.multi_class_ == "one_vs_one":
                self.base_estimator_ = OneVsOneClassifier(self.base_estimator_,
                                                          n_jobs=self.n_jobs)
            elif self.multi_class_ == "direct":
                self.base_estimator_ = wrapper_ordering_classes_by_size(
                    self.base_estimator_)
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class_)

        # TODO: param categorical_features is data dependent, but OvR/OvO don't pass **kwargs through fit(), so it has to be in _BinarySeCoEstimator.__init__
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

    def get_seco_estimators(self) -> Sequence[_BinarySeCoEstimator]:
        """
        :return: The `_BinarySeCoEstimator` instances that were trained.
            Depending on the multi-class strategy, the class labels they use
            differ in order and value.
        """
        check_is_fitted(self, 'base_estimator_')
        is_binary = len(self.classes_) == 2
        if is_binary or self.multi_class_ == "direct":
            assert isinstance(self.base_estimator_,
                              TargetTransformingMetaEstimator)
            return [self.base_estimator_.estimator]
        elif self.multi_class_ == "one_vs_rest":
            assert isinstance(self.base_estimator_, OneVsRestClassifier)
            return self.base_estimator_.estimators_
        elif self.multi_class_ == "one_vs_one":
            assert isinstance(self.base_estimator_, OneVsOneClassifier)
            return self.base_estimator_.estimators_
        else:
            assert False, "invalid state: unknown type of base_estimator_ " \
                f"({str(self.base_estimator_)})"
