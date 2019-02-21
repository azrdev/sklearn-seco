"""
Implementation of SeCo / Covering algorithm:
Common `Rule` allowing == (categorical) or <= and >= (numerical) test.
"""

import math
from abc import ABC, abstractmethod
from functools import total_ordering
from typing import NewType, Iterable, List, Type, TypeVar, Tuple
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


def log2(x: float) -> float:
    """`log2(x) if x > 0 else 0`"""
    return math.log2(x) if x > 0 else 0


def make_empty_rule(n_features: int) -> Rule:
    """:return: A `Rule` with no conditions, it always matches."""
    # NOTE: even if user inputs a dtype which doesn't know np.inf (e.g. `int`),
    # `Rule` always has dtype float which does
    return Rule(np.vstack([np.repeat(np.NINF, n_features),  # lower
                           np.repeat(np.PINF, n_features)  # upper
                           ]))


def rule_ancestors(rule: 'AugmentedRule') -> Iterable['AugmentedRule']:
    """:return: `rule` and all its ancestors, see `AugmentedRule.copy()`."""
    while rule:
        yield rule
        rule = rule.original


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


T = TypeVar('T', bound='AugmentedRule')


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
        The actual antecedent / conjunction of conditions. Always use
        `set_condition` for write access.

    lower, upper: np.ndarray
        Return only the lower/upper part of `self.conditions`. Always use
        `set_condition` for write access.

    instance_no: int
        number of instance, to compare rules by creation order

    original: AugmentedRule or None
        Another rule this one has been forked from, using `copy()`.

    _sort_key: tuple of floats
        Defines an order of rules for `RuleQueue` and finding a `best_rule`,
        using operator `<` (i.e. higher values == better rule == later in the
        queue).
        Set by `AbstractSecoImplementation.evaluate_rule` and accessed
        implicitly through `__lt__`.

    _pn_cache: tuple(int, int)
        Cached positive and negative coverage of this rule. Always access via
        `AbstractSecoImplementation.count_matches`.
    """
    __rule_counter = 0  # TODO: shared across runs/instances

    def __init__(self, *,
                 conditions: Rule = None, n_features: int = None,
                 original: 'AugmentedRule' = None):
        """Construct an `AugmentedRule` with either `n_features` or the given
        `conditions`.

        :param original: A reference to an "original" rule, saved as attribute.
          See `copy()`.
        """
        self.instance_no = AugmentedRule.__rule_counter
        AugmentedRule.__rule_counter += 1
        self.original = original

        if conditions is None:
            assert n_features is not None
            self.conditions = make_empty_rule(n_features)
        elif n_features is None:
            assert conditions is not None
            self.conditions = conditions
        else:
            raise ValueError("Exactly one of (conditions, n_features) "
                             "must be not None.")
        # init fields
        self._pn_cache = None
        self._sort_key = None

    def copy(self: T) -> T:
        """:return: A new `AugmentedRule` with a copy of `self.conditions`."""
        cls = type(self)
        return cls(conditions=self.conditions.copy(), original=self)

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
        self.conditions[boundary, index] = value

    def __lt__(self, other):
        if not hasattr(other, '_sort_key'):
            return NotImplemented
        return self._sort_key < other._sort_key

    def __eq__(self, other):
        if not hasattr(other, '_sort_key'):
            return NotImplemented
        return self._sort_key == other._sort_key


# TODO: maybe move `abstract_seco`/`find_best_rule` into TheoryContext/RuleContext


class TheoryContext:
    """State variables while `abstract_seco` builds a theory.

    Members
    -----

    * `categorical_mask`: An array of shape `(n_features,)` and type bool,
      indicating if a feature is categorical (`True`) or numerical (`False`).
    * `n_features`: The number of features in the dataset, equivalent to
      `X.shape[1]`.
    * `target_class`: The value of the positive class in `y`.
    * `algorithm_config`: A reference to the used `SeCoAlgorithmConfiguration`.
    * `complete_X`: All training examples `X` *at start of training*.
    * `complete_y`: All training classifications `y` *at start of training*.
    """

    def __init__(self, algorithm_config: 'SeCoAlgorithmConfiguration',
                 categorical_mask, n_features, target_class,
                 X, y):
        self.categorical_mask = categorical_mask
        self.n_features = n_features
        self.target_class = target_class
        # keep reference
        self.algorithm_config = algorithm_config
        self.complete_X = X
        self.complete_y = y

    @property
    def implementation(self) -> 'AbstractSecoImplementation':
        return self.algorithm_config.implementation


class RuleContext:
    """State variables while `find_best_rule` builds a single rule.

    Methods provided are:

    - `match_rule`
    - `count_matches`
    - `evaluate_rule`

    Members
    -----
    * `X` and `y`: The current training data and -labels.
      (See also `concrete.GrowPruneSplit` which overrides these properties).
    * `PN`: (P: int, N: int) The count of positive and negative examples in X.
    """

    def __init__(self, theory_context: TheoryContext, X, y):
        self.theory_context = theory_context
        self._X = X
        self._y = y
        self._PN_cache = None

    @property
    def PN(self) -> Tuple[int, int]:
        """:return: (P, N), the count of (positive, negative) examples"""
        if self._PN_cache is None:
            # calculate
            y = self.y
            target_class = self.theory_context.target_class
            P = np.count_nonzero(y == target_class)
            N = len(y) - P
            assert N == np.count_nonzero(y != target_class)
            assert P + N == len(y)
            self._PN_cache = (P, N)
        return self._PN_cache

    @property
    def X(self):
        """The current training data features"""
        return self._X

    @property
    def y(self):
        """The current training data labels/classification"""
        return self._y

    def match_rule(self, rule: AugmentedRule, force_X_complete: bool = False
                   ) -> np.ndarray:
        """Wrap `SeCoAlgorithmConfiguration.match_rule`: Apply `rule` to the
        current context.

        :param force_X_complete: bool. If True, always apply to the full
          training data, instead of the property `self.X` (which may be
          overridden, e.g. by GrowPruneSplit).
        :return: A match array of type bool and length `len(X)`
        """
        tctx = self.theory_context
        X = self._X if force_X_complete else self.X
        return tctx.algorithm_config.match_rule(X,
                                                rule.conditions,
                                                tctx.categorical_mask)

    def count_matches(self, rule: AugmentedRule) -> Tuple[int, int]:
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
        if rule._pn_cache is None:
            covered = self.match_rule(rule)
            positives = self.y == self.theory_context.target_class
            # NOTE: nonzero() is test for True
            p = np.count_nonzero(covered & positives)
            n = np.count_nonzero(covered & ~positives)
            assert p + n == np.count_nonzero(covered)
            rule._pn_cache = (p, n)
        return rule._pn_cache

    def evaluate_rule(self, rule: AugmentedRule) -> None:
        """Rate rule to allow comparison & finding the best refinement."""
        growing_heuristic = \
            self.theory_context.implementation.growing_heuristic
        p, n = self.count_matches(rule)
        rule._sort_key = (growing_heuristic(rule, self),
                          # default tie-breaking: by positive coverage
                          p,
                          # and rule creation order (older = better)
                          -rule.instance_no)


class AbstractSecoImplementation(ABC):
    """The callbacks needed by _BinarySeCoEstimator, subclasses represent
    concrete algorithms (together with the corresponding subclasses of
    `RuleContext` etc).

    TODO: maybe @classmethod → @staticmethod ? pro: callbacks as fun-refs w/o class, con: configurability via class-fields
    TODO: Instead of using this interface, you can also pass all the functions
    to SeCoEstimator separately, without an enclosing class.
    """

    @classmethod
    @abstractmethod
    def init_rule(cls, context: RuleContext) -> AugmentedRule:
        """Create a new rule to be refined before added to the theory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def growing_heuristic(cls, rule: AugmentedRule, context: RuleContext
                          ) -> float:
        """Rate rule to allow comparison with other rules.
        Also used as confidence estimate for voting in multi-class cases.

        Rules are compared using operator `<` on their `_sort_key`, which is
        set by `evaluate_rule` to a tuple of values: First (most important) the
        result of this `growing_heuristic`, later values are used for tie
        breaking.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def select_candidate_rules(cls, rules: RuleQueue, context: RuleContext
                               ) -> Iterable[AugmentedRule]:
        """Remove and return those Rules from `rules` which should be refined.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def refine_rule(cls, rule: AugmentedRule, context: RuleContext
                    ) -> Iterable[AugmentedRule]:
        """Create all refinements from `rule`."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def inner_stopping_criterion(cls, rule: AugmentedRule,
                                 context: RuleContext) -> bool:
        """return `True` to stop refining `rule`, i.e. pre-pruning it."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def filter_rules(cls, rules: RuleQueue, context: RuleContext) -> RuleQueue:
        """After one refinement iteration, filter the candidate `rules` (may be
        empty) for the next one.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def simplify_rule(cls, rule: AugmentedRule, context: RuleContext
                      ) -> AugmentedRule:
        """After `find_best_rule` terminates, this hook is called and may
        implement post-pruning.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def rule_stopping_criterion(cls, theory: Theory, rule: AugmentedRule,
                                context: RuleContext) -> bool:
        """return `True` to stop finding more rules, given `rule` was the
        best Rule found.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def post_process(cls, theory: Theory, context: TheoryContext) -> Theory:
        """Modify `theory` after it has been learned."""
        raise NotImplementedError

    @classmethod
    def confidence_estimate(cls, rule: AugmentedRule, context: RuleContext
                            ) -> float:
        """Evaluate rule on whole training set (`context`) to estimate
        confidence in its predictions, compared to other rules in the theory.

        Default implementation uses `growing_heuristic`.
        """
        return cls.growing_heuristic(rule, context)


class SeCoAlgorithmConfiguration:
    """A concrete SeCo algorithm, defined by code and associated state objects.

    In `SeCoEstimator` an instance of this class is used, to allow wrapping the
    methods `make_rule`, `make_theory_context`, and `make_rule_context` in e.g.
    `functools.partialmethod`.

    The members `RuleClass`, `TheoryContextClass`, and `RuleContextClass`
    should only ever be overridden/subclassed, not wrapped in functions (like
    `functools.partial`), to enable other users (like `extra.trace_coverage`)
    to subclass them again.

    Members
    -----
    - `match_rule`: Callable[[np.ndarray, Rule, np.ndarray], np.ndarray].
      A method applying the Rule (2nd parameter) to the examples `X` (1st
      parameter) given the categorical_mask (3rd parameter) and returning an
      array of length `len(X)` and type bool.
    - `Implementation`: A non-abstract subclass of `AbstractSecoImplementation`
      defining all callback methods needed by `abstract_seco` and
      `find_best_rule`.
    - `RuleClass`: A subclass of `AugmentedRule` defining the attributes that
      supplement the basic `Rule` object (which is the `conditions` field).
    - `TheoryContextClass`: A subclass of `TheoryContext` managing state of an
      `abstract_seco` run.
    - `RuleContextClass`: A subclass of `RuleContext` managing state of a
      `find_best_rule` run.
    """
    match_rule = staticmethod(match_rule)
    # TODO: maybe use a Sequence of classes here and construct subclass with type(name, bases, []) in __init__
    Implementation: Type[AbstractSecoImplementation] = \
        AbstractSecoImplementation
    RuleClass = AugmentedRule
    TheoryContextClass: Type[TheoryContext] = TheoryContext
    RuleContextClass: Type[RuleContext] = RuleContext

    def __init__(self):
        self.implementation = self.Implementation()

    def make_rule(self, *args, **kwargs) -> 'RuleClass':
        """:return: An instance of `RuleClass`."""
        return self.RuleClass(*args, **kwargs)

    def make_theory_context(self, *args, **kwargs) -> 'TheoryContextClass':
        """:return: An instance of `TheoryContextClass`."""
        return self.TheoryContextClass(self, *args, **kwargs)

    def make_rule_context(self, *args, **kwargs) -> 'RuleContextClass':
        """:return: An instance of `RuleContextClass`."""
        return self.RuleContextClass(*args, **kwargs)
