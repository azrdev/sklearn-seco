"""
Implementation of SeCo / Covering algorithm:
Common `Rule` allowing == (categorical) or <= and >= (numerical) test.
"""
import math
from abc import ABC, abstractmethod
from functools import total_ordering
from typing import NewType, Iterable, List, NamedTuple, SupportsFloat
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


def log2(x: SupportsFloat) -> float:
    """`log2(x) if x > 0 else 0`"""
    return math.log2(x) if x > 0 else 0.0


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


condition_trace_entry = NamedTuple('condition_trace_entry',
                                   [('boundary', int), ('index', int),
                                    ('value', float), ('old_value', float)])


@total_ordering
class AugmentedRule:
    """A `Rule` and associated data, like coverage (p, n) of current examples,
    a `_sort_key` defining a total order between instances, and for rules
    forked from others (with `copy()`) a reference to the original rule.

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

    condition_trace: list of tuple(int, int, float)
        (Only used if `enable_condition_trace` is True).
        Contains a tuple(UPPER/LOWER, feature_index, value) for each value that
        was set in conditions, like they were applied to the initially
        constructed rule to get to this rule.

    enable_condition_trace: bool
        Enable filling of the `condition_trace` field. Default `False`.

    _sort_key: tuple of floats
        Define an order of rules for `RuleQueue` and finding a `best_rule`,
        using operator `<` (i.e. higher values == better rule == later in the
        queue).
        Set by `SeCoBaseImplementation.evaluate_rule` and accessed implicitly
        through `__lt__`.

    _p, _n: int
        positive and negative coverage of this rule. Access via
        `SeCoBaseImplementation.count_matches`.
    """
    __rule_counter = 0

    def __init__(self, *,
                 conditions: Rule = None, n_features: int = None,
                 original: 'AugmentedRule' = None,
                 enable_condition_trace: bool = False):
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
        # init fields
        self.enable_condition_trace = enable_condition_trace
        self.condition_trace = []
        if original:
            # copy trace, but into a new list
            self.condition_trace[:] = original.condition_trace
        self._p = None
        self._n = None
        self._sort_key = None

    def copy(self) -> 'AugmentedRule':
        """:return: A new `AugmentedRule` with a copy of `self.conditions`."""
        return AugmentedRule(conditions=self.conditions.copy(), original=self)

    @property
    def lower(self) -> np.ndarray:
        """The "lower" part of the rules' conditions, i.e. `rule.lower <= X`.
        """
        return self.conditions[LOWER]

    @property
    def upper(self) -> np.ndarray:
        """The "upper" part of the rules' conditions, i.e. `rule.upper >= X`.
        """
        return self.conditions[UPPER]

    def set_condition(self, boundary: int, index: int, value):
        self.condition_trace.append(
            condition_trace_entry(boundary, index, value,
                                  self.conditions[boundary, index]))
        self.conditions[boundary, index] = value

    def __lt__(self, other):
        if not hasattr(other, '_sort_key'):
            return NotImplemented
        return self._sort_key < other._sort_key

    def __eq__(self, other):
        if not hasattr(other, '_sort_key'):
            return NotImplemented
        return self._sort_key == other._sort_key

    def match(self, context: 'RuleContext'):
        """TODO: doc"""
        return match_rule(context.X,
                          self.conditions,
                          context.theory_context.categorical_mask)

    def count_matches(self, context: 'RuleContext'):
        """Return (p, n).

        returns
        -------
        p : int
            The count of positive examples (== target_class) covered by `rule`,
            also called *true positives*.
        n : int
            The count of negative examples (!= target_class) covered by `rule`,
            also called *false positives*.
        """
        covered = self.match(context)
        positives = context.y == context.theory_context.target_class
        # NOTE: nonzero() is test for True
        p = np.count_nonzero(covered & positives)
        n = np.count_nonzero(covered & ~positives)
        assert p + n == np.count_nonzero(covered)
        return (p, n)


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


class TheoryContext():
    """TODO: doc"""

    def __init__(self, categorical_mask, n_features, target_class):
        self.rule_prototype_arguments = {}
        self.categorical_mask = categorical_mask
        self.n_features = n_features
        self.target_class = target_class


class RuleContext():
    """TODO: doc"""

    def __init__(self, theory_context: TheoryContext, X, y):
        self.theory_context = theory_context
        self._X = X
        self._y = y
        self._P = self._N = None

    # def __getattr__(self, item):
    #     """Proxy every attribute of TheoryContext"""
    #     return getattr(self.theory_context, item)

    # TODO: merge P,N
    def __calculate_PN(self):
        """Calculate values of properties P, N."""
        y = self.y
        target_class = self.theory_context.target_class
        assert all(x is None for x in (self._P, self._N)) or \
            all(x is not None for x in (self._P, self._N))  # always set both
        # TODO: get P,N from abstract_seco() ?
        self._P = np.count_nonzero(y == target_class)
        self._N = len(y) - self._P
        assert self._N == np.count_nonzero(y != target_class)
        assert self._P + self._N == len(y)

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

    @property
    def X(self):
        """The current training data features"""
        return self._X

    @property
    def y(self):
        """The current training data labels/classification"""
        return self._y


class SeCoBaseImplementation(ABC):
    """The callbacks needed by _BinarySeCoEstimator, subclasses represent
    concrete algorithms.

    `set_context` will have been called before any other callback. After the
    abstract_seco main loop, `unset_context` is called once, where all state
    ought to be removed that is not needed after fitting (e.g. copies of
    training data X, y).

    A few members are maintained by the base class, and can be used by
    implementations:
    # XXX: WIP remove all state from this class

    * `categorical_mask`: An array of shape `(n_features,)` and type bool,
      indicating if a feature is categorical (`True`) or numerical (`False`).
    * `match_rule()`
    * `count_matches()`
    * `n_features`: The number of features in the dataset,
      equivalent to `X.shape[1]`.
    - `P` and `N`: The count of positive and negative examples (in self.X)
    * `rule_prototype_arguments`: kwargs that should be passed to the
      constructor of `AugmentedRule` by any subclass calling it.
    * `target_class`
    * `trace_feature_order`: If True, all our `AugmentedRule` instances use
      their `condition_trace` property to log all values set to each condition.
    """

    match_rule = match_rule

    def make_theory_context(self, *args, **kwargs):
        return TheoryContext(*args, **kwargs)

    def make_rule_context(self, *args, **kwargs):
        return RuleContext(*args, **kwargs)

    def evaluate_rule(self, rule: AugmentedRule, context: RuleContext) -> None:
        """Rate rule to allow comparison & finding the best refinement."""
        p, n = rule.count_matches(context)
        rule._sort_key = (self.growing_heuristic(rule, context),
                          # default tie-breaking: by positive coverage
                          p,
                          # and rule creation order (older = better)
                          -rule.instance_no)

    # xxx WIP: separate callbacks for find_best_rule context into own class
    # TODO: make them all @classmethods ?
    # abstract interface

    @abstractmethod
    def init_rule(self, context: RuleContext) -> AugmentedRule:
        """Create a new rule to be refined before added to the theory."""
        raise NotImplementedError

    @abstractmethod
    def growing_heuristic(self, rule: AugmentedRule,
                          context: RuleContext) -> float:
        """Rate rule to allow comparison with other rules.
        Also used as confidence estimate for voting in multi-class cases.

        Rules are compared using operator `<` on their `_sort_key`, which is
        set by `evaluate_rule` to a tuple of values: First (most important) the
        result of this `growing_heuristic`, later values are used for tie
        breaking.
        """
        raise NotImplementedError

    @abstractmethod
    def select_candidate_rules(self, rules: RuleQueue, context: RuleContext
                               ) -> Iterable[AugmentedRule]:
        """Remove and return those Rules from `rules` which should be refined.
        """
        raise NotImplementedError

    @abstractmethod
    def refine_rule(self, rule: AugmentedRule, context: RuleContext
                    ) -> Iterable[AugmentedRule]:
        """Create all refinements from `rule`."""
        raise NotImplementedError

    @abstractmethod
    def inner_stopping_criterion(self, rule: AugmentedRule,
                                 context: RuleContext) -> bool:
        """return `True` to stop refining `rule`, i.e. pre-pruning it."""
        raise NotImplementedError

    @abstractmethod
    def filter_rules(self, rules: RuleQueue, context: RuleContext
                     ) -> RuleQueue:
        """After one refinement iteration, filter the candidate `rules` (may be
        empty) for the next one.
        """
        raise NotImplementedError

    @abstractmethod
    def simplify_rule(self, rule: AugmentedRule, context: RuleContext) -> AugmentedRule:
        """After `find_best_rule` terminates, this hook is called and may
        implement post-pruning.
        """
        raise NotImplementedError

    @abstractmethod
    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule,
                                context: RuleContext) -> bool:
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
