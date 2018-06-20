"""Implementation of SeCo / Covering algorithm.

Limitations / Assumptions (TODO)
=====

- at most two tests per feature and rule
- no sparse input
- no missing values
- binary estimator, applies binarization to multi-class problems
- first class (from `sklearn.utils.unique_labels()`, i.e. the lowest class)
    always assumed to be positive (may be asymmetrical, because of default rule)
- implicit default rule
- only ordered rule list (no unordered rule set / tree)
- limited operator set:
    - for categorical only ==
    - for numerical only <= and >=
- numerical features always assumed to be ordinal
- no NaN, inf, or -inf values in data
"""

from abc import ABC, abstractmethod
from functools import lru_cache, total_ordering
from typing import Iterable, List, Tuple, NewType

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.utils.validation import check_is_fitted


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    # copied from itertools docs
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


Rule = NewType('Rule', np.ndarray)
Rule.__doc__ = """Represents a conjunction of conditions.

:type: array of dtype float, shape `(2, n_features)`

    - first row "lower"
        - categorical features: contains categories to be matched with `==`
        - numerical features: contains lower bound (`rule[LOWER] <= X`)
    - second row "upper"
        - categorical features: invalid (TODO: unequal operator ?)
        - numerical features: contains upper bound (`rule[UPPER] >= X`)

    To specify "no test", use appropriate infinity values for numerical features
    (np.NINF and np.PINF for lower and upper), and for categorical features any
    non-finite value (np.NaN, np.NINF, np.PINF).
"""

LOWER = 0
UPPER = 1


def make_empty_rule(n_features: int) -> Rule:
    """:return: A `Rule` with no conditions, it always matches."""
    # NOTE: even if user inputs dtype which knows no np.inf (like int),
    # `Rule` has always dtype float which does
    return Rule(np.vstack([np.repeat(np.NINF, n_features),  # lower
                           np.repeat(np.PINF, n_features)  # upper
                           ]))


@total_ordering
class AugmentedRule:
    """A `Rule` and associated data, like coverage (p, n) of current examples,
    a `sort_key` defining a total order between instances, and for rules forked
    from others (with `copy()`) a reference to the original rule.

    Lifetime of an AugmentedRule is from its creation (in `init_rule` or
    `refine_rule`) until its `conditions` are added to the theory in
    `abstract_seco`.

    Attributes
    -----
    conditions: Rule
        The actual antecedent / conjunction of conditions.

    lower, upper: np.ndarray
        Return only the lower/upper part of `self.conditions`.

    instance_no: int
        number of instance, to compare rules by creation order

    original: AugmentedRule or None
        Another rule this one has been forked from, using `copy()`.

    _sort_key: tuple of floats
        Define an order of rules for `RuleQueue` and finding a `best_rule`,
        using operator `<` (i.e. higher values == better rule == later in the
        queue).
        Set to the return value of `SeCoBaseImplementation.evaluate_rule`
        (by `SeCoBaseImplementation.rate_rule`).
        Accessed implicitly through `__lt__`.

    _p, _n: int
        positive and negative coverage of this rule. Access via
        `SeCoBaseImplementation.count_matches`.
    """
    __rule_counter = 0

    def __init__(self, *, conditions: Rule = None, n_features: int = None,
                 original=None):
        """Construct an `AugmentedRule` with either `n_features` or the given
        `conditions`.

        :param original: A reference to an "original" rule, saved as attribute.
          See `copy()`.
        """
        self.instance_no = AugmentedRule.__rule_counter
        AugmentedRule.__rule_counter += 1
        self.original = original

        assert (conditions is None) ^ (n_features is None)  # XOR
        if conditions is None:
            self.conditions = make_empty_rule(n_features)
        elif n_features is None:
            self.conditions = conditions
        else:
            raise ValueError("Exactly one of (conditions, n_features) "
                             "must be not None.")
        # init fields for stats
        self._p = None
        self._n = None
        self._sort_key = None

    def copy(self) -> 'AugmentedRule':
        """:return: A new `AugmentedRule` with a copy of `self.conditions`."""
        return AugmentedRule(conditions=self.conditions.copy(), original=self)

    @property
    def lower(self) -> np.ndarray:
        """The "lower" part of the rules' conditions, i.e. `rule.lower <= X`."""
        return self.conditions[LOWER]

    @property
    def upper(self) -> np.ndarray:
        """The "upper" part of the rules' conditions, i.e. `rule.upper >= X`."""
        return self.conditions[UPPER]

    def __lt__(self, other):
        if not hasattr(other, '_sort_key'):
            return NotImplemented
        return self._sort_key < other._sort_key

    def __eq__(self, other):
        if not hasattr(other, 'sort_key'):
            return NotImplemented
        return self._sort_key == other._sort_key


Theory = List[Rule]
RuleQueue = List[AugmentedRule]


def match_rule(X: np.ndarray,
               rule: Rule,
               categorical_mask: np.ndarray) -> np.ndarray:
    """Apply `rule` to all samples in `X`.

    :param X: An array of shape `(n_samples, n_features)`.
    :param rule: An array of shape `(2, n_features)`,
        holding thresholds (for numerical features),
        or categories (for categorical features),
        or `np.NaN` (to not test this feature).
    :param categorical_mask: An array of shape `(n_features,)` and type bool,
        specifying which features are categorical (True) and numerical (False)
    :return: An array of shape `(n_samples,)` and type bool, telling for each
        sample whether it matched `rule`.

    pseudocode::
        conjugate for all features:
            if feature is categorical:
                return rule[LOWER] is NaN  or  rule[LOWER] == X
            else:
                return  rule[LOWER] is NaN  or  rule[LOWER] <= X
                     && rule[UPPER] is NaN  or  rule[UPPER] >= X
    """

    lower = rule[LOWER]
    upper = rule[UPPER]

    return (categorical_mask & (~np.isfinite(lower) | np.equal(X, lower))
            | (~categorical_mask
               & np.less_equal(lower, X)
               & np.greater_equal(upper, X))
            ).all(axis=1)


def count_matches(rule: Rule, target_class, categorical_mask, X, y
                  ) -> Tuple[int, int]:
    """Return (p, n).

    returns
    -------
    p : int
        The count of positive examples (== target_class) covered by `rule`
    n : int
        The count of negative examples (!= target_class) covered by `rule`
    """

    # the following both are np.arrays of dtype bool
    covered = match_rule(X, rule, categorical_mask)
    positives = y == target_class
    # NOTE: nonzero() is test for True
    p = np.count_nonzero(covered & positives)
    n = np.count_nonzero(covered & ~positives)
    assert p + n == np.count_nonzero(covered)
    return (p, n)


# noinspection PyAttributeOutsideInit
class SeCoBaseImplementation(ABC):
    """The callbacks needed by _BinarySeCoEstimator, subclasses represent
    concrete algorithms.

    `set_context` will have been called before any other callback. After the
    abstract_seco main loop, `unset_context` is called once, where all state
    ought to be removed that is not needed after fitting (e.g. copies of
    training data X, y).

    A few members are maintained by the base class, and can be used by
    implementations:

    - `categorical_mask`: An array of shape `(n_features,)` and type bool,
      indicating if a feature is categorical (`True`) or numerical (`False`).
    - `match_rule()` and `match_rule_raw()`
    - `count_matches()`
    - `n_features`: The number of features in the dataset,
      equivalent to `X.shape[1]`.
    - `P` and `N`: The count of positive and negative examples (in self.X)
    - `target_class`
    - `all_feature_values()`
    """

    def __calculate_PN(self):
        """Calculate values of properties P, N."""
        if hasattr(self, '_P') and hasattr(self, '_N'):
            if None in (self._P, self._N):
                assert self._P is None  # always set both
                assert self._N is None
            else:
                return  # already calculated
        positives = self.y == self.target_class
        self._P = np.count_nonzero(positives)
        self._N = np.count_nonzero(~positives)
        assert self._P + self._N == len(self.y) == self.X.shape[0]

    @property
    def P(self):
        """The count of positive examples"""
        self.__calculate_PN()
        return self._P

    @property
    def N(self):
        """The count of negative examples"""
        self.__calculate_PN()
        return self._N

    @lru_cache(maxsize=None)
    def all_feature_values(self, feature_index: int):
        """
        :return: All distinct values of feature (in examples) with given index,
             sorted.
        """
        # unique also sorts
        return np.unique(self.X[:, feature_index])

    def match_rule(self, rule: AugmentedRule):
        """Apply `rule`, telling for each sample if it matched.

        :return: An array of dtype bool and shape `(n_samples,)`.
        """
        return match_rule(self.X, rule.conditions, self.categorical_mask)

    def match_rule_raw(self, rule: Rule, X):
        """

        :param rule:
        :param X: The samples to test.
        :return: An array of dtype bool and shape `(n_samples,)`.
        """
        return match_rule(X, rule, self.categorical_mask)

    def count_matches(self, rule: AugmentedRule):
        """Return (p, n).

        returns
        -------
        p : int
            The count of positive examples (== target_class) covered by `rule`.
        n : int
            The count of negative examples (!= target_class) covered by `rule`.
        """
        if None in (rule._p, rule._n):
            assert rule._p is None  # always set them both together
            assert rule._n is None
            rule._p, rule._n = count_matches(rule.conditions, self.target_class,
                                             self.categorical_mask,
                                             self.X, self.y)
        return (rule._p, rule._n)

    def set_context(self, estimator: '_BinarySeCoEstimator', X, y):
        """New invocation of `_BinarySeCoEstimator._find_best_rule`.

        Override this hook if you need to keep state across all invocations of
        the callbacks from one find_best_rule run, e.g. (candidate) rule
        evaluations for their future refinement. Be sure to call the base
        implementation.
        """

        # actually don't change, but rewriting them is cheap
        self.categorical_mask = estimator.categorical_mask_
        self.n_features = estimator.n_features_
        self.target_class = estimator.target_class_
        # depend on examples (X, y), which change each iteration
        self.all_feature_values.cache_clear()
        self.X = X
        self.y = y
        self._P = None
        self._N = None

    def unset_context(self):
        """Called after the last invocation of
        `_BinarySeCoEstimator._find_best_rule`.
        """
        self.all_feature_values.cache_clear()
        self.X = None
        self.y = None
        self._P = None
        self._N = None

    def rate_rule(self, rule: AugmentedRule) -> None:
        """Wrapper around `evaluate_rule`."""
        rule._sort_key = self.evaluate_rule(rule)

    # TODO: maybe separate callbacks for find_best_rule context into own class?
    # abstract interface

    @abstractmethod
    def init_rule(self) -> AugmentedRule:
        """Create a new rule to be refined before added to the theory."""
        pass

    @abstractmethod
    def evaluate_rule(self, rule: AugmentedRule) -> float or Tuple[float, ...]:
        """Rate rule to allow comparison & finding the best refinement.

        :return: A rule rating, or a tuple of these (later elements are used for
          tie breaking). Rules are compared using these tuples and operator `<`.
        """
        pass

    @abstractmethod
    def select_candidate_rules(self, rules: RuleQueue
                               ) -> Iterable[AugmentedRule]:
        """Remove and return those Rules from `rules` which should be refined.
        """
        pass

    @abstractmethod
    def refine_rule(self, rule: AugmentedRule) -> Iterable[AugmentedRule]:
        """Create all refinements from `rule`."""
        pass

    @abstractmethod
    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        """return `True` to stop refining `rule`."""
        pass

    @abstractmethod
    def filter_rules(self, rules: RuleQueue) -> RuleQueue:
        """After one refinement iteration, filter the candidate `rules` (may be
        empty) for the next one.
        """
        pass

    @abstractmethod
    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule
                                ) -> bool:
        """return `True` to stop finding more rules, given `rule` was the
        best Rule found.
        """
        pass

    @abstractmethod
    def post_process(self, theory: Theory) -> Theory:
        """Modify `theory` after it has been learned.

        *NOTE*: contrary to all other hooks, this is called after
        `unset_context`.
        """
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

    def _find_best_rule(self, X, y) -> AugmentedRule:
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
        best_rule = init_rule()
        rate_rule(best_rule)
        rules: RuleQueue = [best_rule]
        while len(rules):
            for candidate in select_candidate_rules(rules):
                for refinement in refine_rule(candidate):  # TODO: parallelize here?
                    rate_rule(refinement)
                    if not inner_stopping_criterion(refinement):
                        rules.append(refinement)
                        if best_rule < refinement:
                            best_rule = refinement
            rules.sort()
            rules = filter_rules(rules)
        return best_rule

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
        match_rule = self.implementation.match_rule

        # TODO: split growing/pruning set for ripper
        # main loop
        target_class = self.target_class_
        theory: Theory = list()
        while np.any(y == target_class):
            set_context(self, X, y)
            rule = find_best_rule(X, y)
            if rule_stopping_criterion(theory, rule):
                break
            # ignore the rest of theory, because it already covered
            uncovered = ~ match_rule(rule)
            X = X[uncovered]  # TODO: use mask array instead of copy?
            y = y[uncovered]
            theory.append(rule.conditions)  # throw away augmentation
        unset_context()
        return post_process(theory)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['theory_', 'categorical_mask_'])
        X: np.ndarray = check_array(X)
        target_class = self.target_class_
        match_rule = self.implementation.match_rule_raw
        result = np.repeat(self.classes_[1],  # negative class
                           X.shape[0])

        for rule in self.theory_:
            result = np.where(
                match_rule(rule, X),
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

    def init_rule(self) -> AugmentedRule:
        return AugmentedRule(n_features=self.n_features)

    def evaluate_rule(self, rule: AugmentedRule) -> Tuple[float, float, int]:
        p, n = self.count_matches(rule)
        if p + n == 0:
            return (0, p, -rule.instance_no)
        purity = p / (p + n)
        # tie-breaking by pos. coverage and rule creation order (older = better)
        return (purity, p, -rule.instance_no)

    def select_candidate_rules(self, rules: RuleQueue
                               ) -> Iterable[AugmentedRule]:
        last = rules.pop()
        return [last]

    def refine_rule(self, rule: AugmentedRule) -> Iterable[AugmentedRule]:
        all_feature_values = self.all_feature_values
        # TODO: mark constant features for exclusion in future specializations

        for index in np.argwhere(self.categorical_mask
                                 & ~np.isfinite(rule.lower)  # unused features
                                 ).ravel():
            # argwhere returns each index in separate list, ravel() unpacks
            for value in all_feature_values(index):
                specialization = rule.copy()
                specialization.lower[index] = value
                yield specialization

        for feature_index in np.nonzero(~self.categorical_mask)[0]:
            old_lower = rule.lower[feature_index]
            no_old_lower = ~np.isfinite(old_lower)
            old_upper = rule.upper[feature_index]
            no_old_upper = ~np.isfinite(old_upper)
            for value1, value2 in pairwise(all_feature_values(feature_index)):
                new_threshold = (value1 + value2) / 2
                # override is collation of lower bounds
                if no_old_lower or new_threshold > old_lower:
                    # don't test contradiction (e.g. f < 4 && f > 6)
                    if no_old_upper or new_threshold < old_upper:
                        specialization = rule.copy()
                        specialization.lower[feature_index] = new_threshold
                        yield specialization
                # override is collation of upper bounds
                if no_old_upper or new_threshold < old_upper:
                    # don't test contradiction
                    if no_old_lower or new_threshold > old_lower:
                        specialization = rule.copy()
                        specialization.upper[feature_index] = new_threshold
                        yield specialization

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        # TODO: java NoNegativesCoveredStop:
        # p, n = self.count_matches(rule)
        # return n == 0
        return False

    def filter_rules(self, rules: RuleQueue) -> RuleQueue:
        return rules[-1:]  # only the best one

    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule
                                ) -> bool:
        # TODO: java CoverageRuleStop;
        # p, n = self.count_matches(rule)
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

    def evaluate_rule(self, rule: AugmentedRule) -> Tuple[float, float, int]:
        # laplace heuristic
        p, n = self.count_matches(rule)
        LPA = (p + 1) / (p + n + 2)
        return (LPA, p, rule.instance_no)  # tie-breaking by positive coverage

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        # TODO: compare LRS with Java & papers
        # p, n = self.count_matches(rule)
        # P, N = self.P, self.N
        # purity = p / (p+n)
        # impurity = n / (p+n)
        # CE = -purity * np.log(purity / (P/(P+N))) \
        #     -impurity * np.log(impurity / (N/P+N))
        # J = p * CE
        # LRS = 2*(P+N)*J
        # return LRS <= self.LRS_threshold
        return False

    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule) -> bool:
        # return True iff rule covers no examples
        p, n = self.count_matches(rule)
        return p == 0


class CN2Estimator(SeCoEstimator):
    def __init__(self,
                 LRS_threshold: float = 0.9,
                 multi_class="one_vs_rest",
                 n_jobs=1):
        super().__init__(CN2Implementation(LRS_threshold), multi_class, n_jobs)
        # sklearn assumes all parameters are class fields
        self.LRS_threshold = LRS_threshold
