"""Implementation of SeCo / Covering algorithm.

Limitations (TODO)
=====

- at most two tests per feature and rule
- no sparse input
- no missing values
- binary estimator, applies binarization to multi-class problems
- first class always assumed to be positive
- only ordered rule list, implicit default rule
- limited operator set:
    - for categorical only ==
    - for numerical only <= and >=

Assumptions
=====

- numerical features are ordinal
- first in `sklearn.utils.unique_labels()` (i.e. lowest) is positive class
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Iterable, List, NamedTuple, Dict, MutableMapping, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.utils.validation import check_is_fitted


# copied from itertools docs
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


Rule = np.ndarray
"""Represents a conjunction of conditions.

array of dtype float, shape `(2, n_features)`

- first row "lower"
    - categorical features: contains categories to be matched with "equals"
    - numerical features: contains lower bound (`rule[lower] <= X`)
- second row "upper"
    - categorical features: invalid (TODO: unequal operator ?)
    - numerical features: contains upper bound (`rule[upper] >= X`)
"""
lower = 0
upper = 1


def make_empty_rule(n_features: int) -> Rule:
    return np.repeat(np.NaN, n_features * 2).reshape((2, n_features))


# TODO: ordered list / unordered set/tree
Theory = List[Rule]
# TODO: document default rule

RatedRule = NamedTuple('RatedRule', [
    ('rating', float or Tuple[float, float]),
    ('rule', Rule)])
RuleQueue = List[RatedRule]


def match_rule(X: np.ndarray,
               rule: Rule,
               categorical_mask: np.ndarray) -> np.ndarray:
    """Apply `rule` to all samples in `X`.

    :param X: An array of shape `(n_samples, n_features)`.
    :param rule: An array of shape `(n_features,)`, holding thresholds (for
        numerical features) or categories (for categorical features), or
        `np.NaN` (to not test this feature).
    :param categorical_mask: An array of shape `(n_features,)` and type bool,
        specifying which features are categorical (True) and numerical (False)
    :return: An array of shape `(n_samples,)` and type bool, telling for each
        sample whether it matched `rule`.

    pseudocode::
        conjugate for all features:
            if feature is categorical:
                return rule[lower] is NaN  or  rule[lower] == X
            else:
                return  rule[lower] is NaN  or  rule[lower] <= X
                     && rule[upper] is NaN  or  rule[upper] >= X
    """

    def mkbuf():
        """allocate buffer with default value True (if rule=NaN)"""
        return np.ones_like(X, dtype=np.bool_)

    no_lower = np.isnan(rule[lower])
    no_upper = np.isnan(rule[upper])
    return (categorical_mask
               & (no_lower
                  | np.equal(X, rule[lower], where=~no_lower, out=mkbuf()))
            | ~categorical_mask
               & (no_lower
                  | np.less_equal(rule[lower], X, where=~no_lower, out=mkbuf()))
               & (no_upper
                  | np.greater_equal(rule[upper], X, where=~no_upper, out=mkbuf()))
            ).all(axis=1)


# TODO: cache result, esp. P,N are the same for all refinements
def count_matches(metrics: Dict[str, int] or Iterable[str],
                  rule: Rule, target_class, categorical_mask, X, y
                  ) -> Dict[str, int]:
    """Return `metrics` where for the following keys, if present the
     corresponding values are calculated:

    returns
    -------
    p : int
        The count of positive examples (== target_class) covered by `rule`
    n : int
        The count of negative examples (!= target_class) covered by `rule`
    P : int
        The count of positive examples
    N : int
        The count of negative examples
    """
    if not isinstance(metrics, MutableMapping):
        metrics = dict(zip(metrics, [None]*len(metrics)))

    # the following three are np.arrays of bool
    covered = match_rule(X, rule, categorical_mask)
    positives = y == target_class
    negatives = ~positives
    # NOTE: nonzero() is test for True
    if 'p' in metrics:
        metrics['p'] = np.count_nonzero(covered & positives)
    if 'n' in metrics:
        metrics['n'] = np.count_nonzero(covered & negatives)
    if 'P' in metrics:
        metrics['P'] = np.count_nonzero(positives)
    if 'N' in metrics:
        metrics['N'] = np.count_nonzero(negatives)
    if 'P' in metrics and 'N' in metrics:
        assert metrics['P'] + metrics['N'] == len(y) == X.shape[0]
    if 'p' in metrics and 'n' in metrics:
        assert metrics['p'] + metrics['n'] == np.count_nonzero(covered)
    return metrics


# noinspection PyAttributeOutsideInit
class SeCoBaseImplementation(ABC):
    """The callbacks needed by _BinarySeCoEstimator, subclasses represent
    concrete algorithms.

    A few members are maintained by the base class, and can be used by
    implementations:

    - `categorical_mask`: An array of shape `(n_features,)` and type bool,
      indicating if a feature is categorical (`True`) or numerical (`False`).
    - `empty_rule`: An empty rule (all fields set to `np.NaN` i.e. "no test"),
      equivalent to `True`. Be sure to always `np.copy()` this object.
    - `n_features`: The number of features in the dataset,
      equivalent to `X.shape[1]`.
    - `target_class`
    - `all_feature_values`
    """

    @lru_cache(maxsize=None)
    def all_feature_values(self, feature_index: int):
        """
        :return: All distinct values of feature (in examples) with given index,
             sorted.
        """
        # unique also sorts
        return np.unique(self._X[:, feature_index])

    def set_context(self, estimator: '_BinarySeCoEstimator', X, y):
        """New invocation of `_BinarySeCoEstimator._find_best_rule`.

        Override this hook if you need to keep state across all invocations of
        the callbacks from one find_best_rule run, e.g. (candidate) rule
        evaluations for their future refinement. Be sure to call the base
        implementation.
        """

        # actually don't change, but rewriting them is cheap
        self.categorical_mask = estimator.categorical_mask_
        self.empty_rule = make_empty_rule(estimator.n_features_)
        self.n_features = estimator.n_features_
        self.target_class = estimator.target_class_
        # depend on examples, which change each iteration
        self.all_feature_values.cache_clear()
        self._X = X

    def unset_context(self):
        """Called after the last invocation of
        `_BinarySeCoEstimator._find_best_rule`.
        """
        self.all_feature_values.cache_clear()
        self._X = None

    def rate_rule(self, rule: Rule, X, y) -> RatedRule:
        return RatedRule(self.evaluate_rule(rule, X, y), rule)

    # abstract interface

    @abstractmethod
    def init_rule(self, X, y) -> Rule:
        """Create a new rule to be refined before added to the theory."""
        pass

    # FIXME: make metric(s) of previous rule available (also in stop criteria)
    @abstractmethod
    def evaluate_rule(self, rule: Rule, X, y) -> float or Tuple[float, float]:
        """Rate rule to allow comparison & finding the best refinement
        (using operator `>`).
        """
        pass

    @abstractmethod
    def select_candidate_rules(self, rules: RuleQueue, X, y
                               ) -> Iterable[Rule]:
        """Remove and return those Rules from `rules` which should be refined.
        """
        pass

    @abstractmethod
    def refine_rule(self, rule: Rule, X, y) -> Iterable[Rule]:
        """Create all refinements from `rule`."""
        pass

    @abstractmethod
    def inner_stopping_criterion(self, rule: Rule, X, y) -> bool:
        """return `True` to stop refining `rule`."""
        pass

    @abstractmethod
    def filter_rules(self, rules: RuleQueue, X, y) -> RuleQueue:
        """After one refinement iteration, filter the candidate `rules` (may be
        empty) for the next one.
        """
        pass

    @abstractmethod
    def rule_stopping_criterion(self, theory: Theory, rule: Rule, X, y) -> bool:
        """return `True` to stop finding more rules, given `rule` was the
        best Rule found.
        """
        pass

    @abstractmethod
    def post_process(self, theory: Theory) -> Theory:
        """Modify `theory` after it has been learned."""
        pass


# noinspection PyAttributeOutsideInit
class _BinarySeCoEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, implementation: SeCoBaseImplementation):
        super().__init__()
        self.implementation = implementation

    # TODO: propagate `categorical_features` from SeCoEstimator
    def fit(self, X, y, categorical_features=None):
        """Build the decision rule list from training data `X` with labels `y`.

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
        """
        X, y = check_X_y(X, y)

        # prepare  target / labels / y

        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        self.target_class_ = self.classes_[0]
        # TODO: assuming classes_[0, 1] being positive & negative.
        # TODO: a default rule might lead to asymmetry → positive class should be configurable

        # prepare  attributes / features / X
        self.n_features_ = X.shape[1]
        # categorical_features modeled after OneHotEncoder
        if (categorical_features is None) or not len(categorical_features):
            self.categorical_mask_ = np.zeros(self.n_features_, dtype=bool)
        elif isinstance(categorical_features, np.ndarray):
            self.categorical_mask_ = np.zeros(self.n_features_, dtype=bool)
            self.categorical_mask_[np.asarray(categorical_features)] = True
        elif categorical_features == 'all':
            self.categorical_mask_ = np.ones(self.n_features_, dtype=bool)
        else:
            raise ValueError("categorical_features must be one of: None, 'all',"
                             " np.ndarray of dtype bool or integer,"
                             " but got {}.".format(categorical_features))

        # run SeCo algorithm
        self.theory_ = self._abstract_seco(X, y)
        return self

    def _find_best_rule(self, X, y) -> Rule:
        """Inner loop of abstract SeCo/Covering algorithm.

        :param X: Not yet covered examples.
        :param y: Classification for `X`.
        """

        # resolve methods once for performance
        init_rule = self.implementation.init_rule
        rate_rule = self.implementation.rate_rule
        select_candidate_rules = self.implementation.select_candidate_rules
        refine_rule = self.implementation.refine_rule
        inner_stopping_criterion = self.implementation.inner_stopping_criterion
        filter_rules = self.implementation.filter_rules

        # algorithm
        best_rule = rate_rule(init_rule(X, y), X, y)
        rules: RuleQueue = [best_rule]
        while len(rules):
            for candidate in select_candidate_rules(rules, X, y):
                for refinement in refine_rule(candidate, X, y):  # TODO: parallelize here?
                    new_rule = rate_rule(refinement, X, y)
                    if not inner_stopping_criterion(refinement, X, y):
                        rules.append(new_rule)
                        if new_rule[0] > best_rule[0]:
                            best_rule = new_rule
            rules.sort(key=lambda rr: rr.rating)
            rules = filter_rules(rules, X, y)
        return best_rule[1]

    def _abstract_seco(self, X: np.ndarray, y: np.ndarray) -> Theory:
        """Main loop of abstract SeCo/Covering algorithm.

        :return: Theory
        """

        # resolve methods once for performance
        set_context = self.implementation.set_context
        rule_stopping_criterion = self.implementation.rule_stopping_criterion
        find_best_rule = self._find_best_rule
        unset_context = self.implementation.unset_context
        post_process = self.implementation.post_process

        # TODO: split growing/pruning set for ripper
        # main loop
        target_class = self.target_class_
        theory = list()
        while np.any(y == target_class):
            set_context(self, X, y)
            rule = find_best_rule(X, y)
            if rule_stopping_criterion(theory, rule, X, y):
                break
            # ignore the rest of theory, because it already covered
            covered = match_rule(X, rule, self.categorical_mask_)
            X = X[~covered]  # TODO: use mask array instead of copy?
            y = y[~covered]
            theory.append(rule)
        unset_context()
        return post_process(theory)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['theory_', 'categorical_mask_'])
        X: np.ndarray = check_array(X)
        target_class = self.target_class_
        result = np.repeat(self.classes_[1],  # negative class
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
                        np.array([[1, 0]]),
                        np.array([[0, 1]]))


# noinspection PyAttributeOutsideInit
class SeCoEstimator(BaseEstimator, ClassifierMixin):
    """Wrap the base SeCo to provide class label binarization."""
    def __init__(self, implementation: SeCoBaseImplementation,
                 multi_class="one_vs_rest", n_jobs=1):
        self.implementation = implementation
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False)

        self.base_estimator_ = _BinarySeCoEstimator(self.implementation)

        # copied from GaussianProcessClassifier
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError("SeCoEstimator requires 2 or more distinct "
                             "classes. Only class %s present."
                             % self.classes_[0])
        elif self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(self.base_estimator_,
                                                           n_jobs=self.n_jobs)
            elif self.multi_class == "one_vs_one":
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
        check_is_fitted(self, ["classes_", "n_classes_"])
        X = check_array(X)
        return self.base_estimator_.predict(X)


class SimpleSeCoImplementation(SeCoBaseImplementation):

    def init_rule(self, X, y) -> Rule:
        return self.empty_rule

    def evaluate_rule(self, rule: Rule, X, y) -> Tuple[float, float]:
        metrics = count_matches(('p', 'n'), rule, self.target_class,
                                self.categorical_mask, X, y)
        p = metrics['p']
        n = metrics['n']
        if p+n == 0:
            return (0, 0)
        purity = p / (p+n)
        return (purity, p)  # tie-breaking by positive coverage

    def select_candidate_rules(self, rules: RuleQueue, X, y) -> Iterable[Rule]:
        last = rules.pop()
        return [last.rule]

    def refine_rule(self, rule: Rule, X, y) -> Iterable[Rule]:
        all_feature_values = self.all_feature_values
        # TODO: mark constant features for exclusion in future specializations

        for index in np.argwhere(self.categorical_mask
                                 & np.isnan(rule[lower])  # unused features
                                 ).ravel():
            # argwhere returns each index in separate list, ravel() unpacks
            for value in all_feature_values(index):
                specialization = rule.copy()
                specialization[lower, index] = value
                yield specialization

        for feature_index in np.nonzero(~self.categorical_mask)[0]:
            for value1, value2 in pairwise(all_feature_values(feature_index)):
                new_threshold = (value1 + value2) / 2
                # override is collation of lower bounds
                old_threshold = rule[lower, feature_index]
                if np.isnan(old_threshold) or new_threshold > old_threshold:
                    specialization = rule.copy()
                    specialization[lower, feature_index] = new_threshold
                    yield specialization
                # override is collation of upper bounds
                old_threshold = rule[upper, feature_index]
                if np.isnan(old_threshold) or new_threshold < old_threshold:
                    specialization = rule.copy()
                    specialization[upper, feature_index] = new_threshold
                    yield specialization

    def inner_stopping_criterion(self, rule: Rule, X, y) -> bool:
        # TODO: java NoNegativesCoveredStop:
        # n = count_matches(('n',), rule, self.target_class,
        #                   self.categorical_mask, X, y)['n']
        # return n == 0
        return False

    def filter_rules(self, rules: RuleQueue, X, y) -> RuleQueue:
        return rules[-1:]  # only the best one

    def rule_stopping_criterion(self, theory: Theory, rule: Rule, X, y) -> bool:
        # TODO: java CoverageRuleStop;
        # metrics = count_matches(('p', 'n'), rule, self.target_class,
        #                         self.categorical_mask, X, y)
        # p = metrics['p']
        # n = metrics['n']
        # return n >= p
        return False

    def post_process(self, theory: Theory) -> Theory:
        return theory


class SimpleSeCoEstimator(SeCoEstimator):
    def __init__(self, multi_class="one_vs_rest", n_jobs=1):
        super().__init__(SimpleSeCoImplementation(), multi_class, n_jobs)


class CN2Implementation(SimpleSeCoImplementation):
    def __init__(self, LRS_threshold: float):
        super().__init__()
        self.LRS_threshold = LRS_threshold

    def evaluate_rule(self, rule: Rule, X, y) -> Tuple[float, float]:
        # laplace heuristic
        metrics = count_matches(('p', 'n'), rule, self.target_class,
                                self.categorical_mask, X, y)
        p = metrics['p']
        n = metrics['n']
        LPA = (p + 1) / (p + n + 2)
        return (LPA, p)  # tie-breaking by positive coverage

    def inner_stopping_criterion(self, rule: Rule, X, y) -> bool:
        # TODO: compare LRS with Java & papers
        # metrics = count_matches(('p', 'n', 'P', 'N'), rule, self.target_class,
        #                         self.categorical_mask, X, y)
        # p = metrics['p']
        # n = metrics['n']
        # P = metrics['P']
        # N = metrics['N']
        # purity = p / (p+n)
        # impurity = n / (p+n)
        # CE = -purity * np.log(purity / (P/(P+N))) \
        #     -impurity * np.log(impurity / (N/P+N))
        # J = p * CE
        # LRS = 2*(P+N)*J
        # return LRS <= self.LRS_threshold
        return False

    def rule_stopping_criterion(self, theory: Theory, rule: Rule, X, y) -> bool:
        # return True iff rule covers no examples
        p = count_matches({'p'}, rule, self.target_class,
                          self.categorical_mask, X, y)['p']
        return p == 0


class CN2Estimator(SeCoEstimator):
    def __init__(self,
                 LRS_threshold: float = 0.9,
                 multi_class="one_vs_rest",
                 n_jobs=1):
        super().__init__(CN2Implementation(LRS_threshold), multi_class, n_jobs)
        # needed for test `check_parameters_default_constructible`:
        self.LRS_threshold = LRS_threshold
