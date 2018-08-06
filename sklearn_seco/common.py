"""
Implementation of SeCo / Covering algorithm:
Common `Rule` allowing == (categorical) or <= and >= (numerical) test.
"""

from abc import ABC, abstractmethod
from functools import total_ordering
from typing import NewType, Tuple, Iterable, List, Union
import numpy as np


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
Theory = List[Rule]
RuleQueue = List['AugmentedRule']


def make_empty_rule(n_features: int) -> Rule:
    """:return: A `Rule` with no conditions, it always matches."""
    # NOTE: even if user inputs dtype which knows no np.inf (like int),
    # `Rule` has always dtype float which does
    return Rule(np.vstack([np.repeat(np.NINF, n_features),  # lower
                           np.repeat(np.PINF, n_features)  # upper
                           ]))


def rule_ancestors(rule: 'AugmentedRule') -> Iterable['AugmentedRule']:
    """:return: `rule` and all its ancestors, see `AugmentedRule.copy()`."""
    while rule:
        yield rule
        rule = rule.original


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
    """

    def __calculate_PN(self):
        """Calculate values of properties P, N."""
        # TODO: get these from abstract_seco() ?
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
        self.X = X
        self.y = y
        self._P = None
        self._N = None

    def unset_context(self):
        """Called after the last invocation of
        `_BinarySeCoEstimator._find_best_rule`.
        """
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
        raise NotImplementedError

    @abstractmethod
    def evaluate_rule(self, rule: AugmentedRule) -> Union[float, Tuple[float, ...]]:
        """Rate rule to allow comparison & finding the best refinement.

        :return: A rule rating, or a tuple of these (later elements are used for
          tie breaking). Rules are compared using these tuples and operator `<`.
        """
        raise NotImplementedError

    @abstractmethod
    def select_candidate_rules(self, rules: RuleQueue
                               ) -> Iterable[AugmentedRule]:
        """Remove and return those Rules from `rules` which should be refined.
        """
        raise NotImplementedError

    @abstractmethod
    def refine_rule(self, rule: AugmentedRule) -> Iterable[AugmentedRule]:
        """Create all refinements from `rule`."""
        raise NotImplementedError

    @abstractmethod
    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        """return `True` to stop refining `rule`, i.e. pre-pruning it."""
        raise NotImplementedError

    @abstractmethod
    def filter_rules(self, rules: RuleQueue) -> RuleQueue:
        """After one refinement iteration, filter the candidate `rules` (may be
        empty) for the next one.
        """
        raise NotImplementedError

    @abstractmethod
    def simplify_rule(self, rule: AugmentedRule) -> AugmentedRule:
        """After `find_best_rule` terminates, this hook is called and may
        implement post-pruning.
        """
        raise NotImplementedError

    @abstractmethod
    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule
                                ) -> bool:
        """return `True` to stop finding more rules, given `rule` was the
        best Rule found.
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(self, theory: Theory) -> Theory:
        """Modify `theory` after it has been learned.

        *NOTE*: contrary to all other hooks, this is called after
        `unset_context`.
        """
        raise NotImplementedError


# import only needed for type checking, place here to break circularity
from sklearn_seco.abstract import _BinarySeCoEstimator
