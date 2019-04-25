"""
Implementation of SeCo / Covering algorithm:
Common `Rule` allowing == (categorical) or <= and >= (numerical) test.
"""

from abc import ABC, abstractmethod
from functools import total_ordering
from typing import Iterable, List, Tuple, Type, TypeVar, Dict, Any

import numpy as np


TGT = TypeVar("TGT")

Theory = List['Rule']
RuleQueue = List['AugmentedRule']


class Rule:
    """A rule mapping feature values to a target classification.

    Attributes
    -----
    head : a single value
        The classification of a sample, if it matches the body.

    body : array of dtype float, shape `(2, n_features)`
        Represents a conjunction of conditions.

        - first row "lower"
            - categorical features: contains categories to be matched with `==`
            - numerical features: contains lower bound (`rule[LOWER] <= X`)
        - second row "upper"
            - categorical features: invalid (TODO: unequal operator ?)
            - numerical features: contains upper bound (`rule[UPPER] >= X`)

        To specify "no test", use appropriate infinity values for numerical
        features (np.NINF and np.PINF for lower and upper, respectively), and
        for categorical features any non-finite value (np.NaN, np.NINF,
        np.PINF).
    """
    head: TGT
    body: np.ndarray

    def __init__(self, head: TGT, body):
        self.head = head
        self.body = body

    def copy(self) -> 'Rule':
        return type(self)(self.head, self.body.copy())

    @staticmethod
    def make_empty(n_features: int, target_class: TGT) -> 'Rule':
        """:return: A `Rule` with no conditions, it always matches."""
        # NOTE: even if user inputs a dtype which doesn't have np.inf
        # (e.g. `int`), `Rule.body` always has dtype float which has.
        return Rule(target_class,
                    np.vstack([np.repeat(np.NINF, n_features),  # lower
                               np.repeat(np.PINF, n_features)  # upper
                               ]))
    LOWER = 0
    UPPER = 1

    def body_empty(self):
        """
        :return: True iff the rule body is empty, i.e. no conditions are set.
            Such a rule matches any sample.
        """
        return not np.isfinite(self.body).any()

    def to_string(self: 'Rule',
                  categorical_mask: np.ndarray,
                  feature_names: List[str] = None,
                  class_names: List[str] = None,
                  ) -> str:
        """:return: a string representation of `self`."""

        n_features = self.body.shape[1]
        if feature_names:
            assert n_features == len(feature_names)
        else:
            feature_names = ['feature_{}'.format(i + 1)
                             for i in range(n_features)]
        classification = ' => ' + str(class_names[self.head]
                                      if class_names
                                      else self.head)
        return ' and '.join(
            '({ft} {op} {thresh:.3})'.format(
                ft=feature_names[ti[1]],
                op='==' if categorical_mask[ti[1]] else
                '>=' if ti[0] == Rule.LOWER else '<=',
                thresh=self.body[ti])
            # ti has type: Tuple[int, int]
            # where ti[0] is LOWER or UPPER and ti[1] is the feature index
            for ti in zip(*np.isfinite(self.body).nonzero())) + classification


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

    lower = rule.body[Rule.LOWER]
    upper = rule.body[Rule.UPPER]

    return (categorical_mask & (~np.isfinite(lower) | np.equal(X, lower))
            | (~categorical_mask
               & np.less_equal(lower, X)
               & np.greater_equal(upper, X))
            ).all(axis=1)


T = TypeVar('T', bound='AugmentedRule')  # needed for type signature of `copy`


@total_ordering
class AugmentedRule:
    """A `Rule` and associated data, like coverage `pn` of current examples,
    a `_sort_key` defining a total order between instances, and for rules
    forked from others (with `copy()`) a reference to the original rule.

    Lifetime of an AugmentedRule is from its creation (in `init_rule` or
    `refine_rule`) until its `_conditions` are added to the theory in
    `abstract_seco`.

    Fields
    -----
    direct_multiclass_support : bool, default True
        If any subclass does not support direct learning of multiclass
        theories, it must set this to False.

    Attributes
    -----
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
    """

    direct_multiclass_support = True
    __rule_counter = 0  # TODO: shared across runs/instances

    @classmethod
    def make_empty(cls, n_features: int, target_class: TGT) -> 'AugmentedRule':
        return cls(Rule.make_empty(n_features, target_class))

    def __init__(self, conditions: Rule, original: 'AugmentedRule' = None):
        """Construct an `AugmentedRule`, use `from_raw_rule` or `make_empty`
        instead.
        """
        self.instance_no = AugmentedRule.__rule_counter
        AugmentedRule.__rule_counter += 1
        self.original = original
        self._conditions = conditions
        # _p_cache maps force_complete_data to covered counts per class
        self._p_cache: Dict[bool, np.ndarray] = {}
        self._sort_key = None

    def copy(self: T, *, head: TGT = None,
             conditions: Tuple[int, int, Any] = None) -> T:
        """Create a modified copy of a rule.

        :param head: A new head (target_class), if not None.
        :param conditions: A tuple(boundary: int, index: int, value) or None.
            If not None, set `copy.body[boundary, index] = value`.
        :return: A copy of self with different `head` and/or `body`, if not
            None.
        """
        cls = type(self)
        copy = cls(self._conditions.copy(), original=self)  # type: T
        if head is not None:
            copy._conditions.head = self.head
        if conditions is None:
            # share the coverage counts
            copy._p_cache = self._p_cache
        else:
            boundary, index, value = conditions
            copy._conditions.body[boundary, index] = value
        return copy

    def ancestors(self: T) -> Iterable[T]:
        """:return: `self` and all its ancestors, see `copy`."""
        rule = self
        while rule:
            yield rule
            rule = rule.original

    @property
    def head(self) -> TGT:
        """The rules' target class, i.e. `rule.head`."""
        return self._conditions.head

    @head.setter
    def head(self, target_class: TGT):
        self._conditions.head = target_class

    @property
    def body(self) -> np.ndarray:
        """The rules' conditions. To change it, use `copy`."""
        return self._conditions.body

    @property
    def lower(self) -> np.ndarray:
        """The "lower" part of the rule body, i.e. `rule.lower <= X`.
        To change it, use `copy`.
        """
        return self._conditions.body[Rule.LOWER]

    @property
    def upper(self) -> np.ndarray:
        """The "upper" part of the rule body, i.e. `rule.upper >= X`.
        To change it, use `copy`.
        """
        return self._conditions.body[Rule.UPPER]

    @property
    def raw(self) -> Rule:
        """
        :return: The rule body without augmentation, as needed for the theory.
        """
        return self._conditions

    def __lt__(self, other):
        """Sort `AugmentedRule` objects by their `_sort_key`."""
        if not hasattr(other, '_sort_key'):
            return NotImplemented
        return self._sort_key < other._sort_key

    def __eq__(self, other):
        """Compare `AugmentedRule` objects by their `_sort_key`."""
        if not hasattr(other, '_sort_key'):
            return NotImplemented
        return self._sort_key == other._sort_key

    def pn(self, context: 'RuleContext', force_complete_data: bool = False):
        """:return: `context.pn(self, force_complete_data)`"""
        # TODO: move code here from RuleContext.pn, just there to avoid needing GrowPruneSplitRuleClass
        return context.pn(self, force_complete_data)


# TODO: maybe move `abstract_seco`/`find_best_rule` into TheoryContext/RuleContext


class TheoryContext:
    """State variables while `abstract_seco` builds a theory.

    The static Parameters are the same as in :class:`_BinarySeCoEstimator`,
    apart from the trailing "_" in their name:
      - `algorithm_config`
      - `categorical_mask`
      - `n_features`
      - `classes`
      - `class_counts`
      - `classes_by_size`
      - `target_class_idx`

    Other Methods & Properties provided are:
      - `implementation`
      - `target_class`

    Fields
    -----
    direct_multiclass_support : bool, default True
        If any subclass does not support direct learning of multiclass
        theories, it must set this to False *per class*.

    Attributes
    -----
    complete_X :
        All training examples `X` *at start of training*.

    complete_y :
        All training classifications `y` *at start of training*.
    """

    direct_multiclass_support = True

    def __init__(self,
                 algorithm_config: 'SeCoAlgorithmConfiguration',
                 categorical_mask: np.ndarray,
                 n_features: int,
                 classes: np.ndarray,
                 X, y):
        self.categorical_mask = categorical_mask
        self.n_features = n_features
        self.classes = classes
        self.algorithm_config = algorithm_config
        self.complete_X = X
        self.complete_y = y

        assert self.is_binary() or self.algorithm_config.direct_multiclass_support(), (
            "Trying to learn non-binary problem directly, "
            "but no direct multiclass support.")

    @property
    def implementation(self) -> 'AbstractSecoImplementation':
        return self.algorithm_config.implementation

    def is_binary(self):
        return len(self.classes) == 2


class RuleContext:
    """State variables while `find_best_rule` builds a single rule.

    Methods & Properties provided are:
      - `PN`
      - `X` and `y`
      - `match_rule`
      - `pn`
      - `evaluate_rule`

    Fields
    -----
    direct_multiclass_support : bool, default True
        If any subclass does not support direct learning of multiclass
        theories, it must set this to False.

    Members
    -----
    * `X` and `y`: The current training data and -labels.
      (See also `concrete.GrowPruneSplit` which overrides these properties).
    * `PN`: Tuple[P: int, N: int]
        The count of positive and negative examples in X, for given
        target_class. Cached.
    """

    direct_multiclass_support = True

    def __init__(self, theory_context: TheoryContext, X, y):
        self.theory_context = theory_context
        self._X = X
        self._y = y
        # _PN_cache maps force_complete_data:bool to the counts per target (P)
        self._PN_cache: Dict[bool, np.ndarray] = {}

    def PN(self, target_class: TGT, force_complete_data: bool = False
           ) -> Tuple[int, int]:
        """
        :return: (P, N), the count of (positive, negative) examples for
            `target_class`. Cached.
        :param force_complete_data:
            Iff True, always use complete dataset (e.g. if growing/pruning
            datasets would be used otherwise).
        """
        if force_complete_data not in self._PN_cache:
            y = self._y if force_complete_data else self.y
            self._PN_cache[force_complete_data] = self._count_PN(y)
        class_counts = self._PN_cache[force_complete_data].tolist()
        target_idx = np.take(
            np.argwhere(self.theory_context.classes == target_class), 0)
        P = class_counts.pop(target_idx)
        N = sum(class_counts)
        return P, N

    def _count_PN(self, y) -> np.ndarray:
        """
        :return: A mapping `target_class => P` for all classes.
            Uncached, use `PN` instead.
        """
        classes, class_counts = np.unique(y, return_counts=True)
        class_idx = np.searchsorted(self.theory_context.classes, classes)
        P = np.zeros_like(self.theory_context.classes)
        P[class_idx] = class_counts
        return P

    @property
    def X(self):
        """The current training data features"""
        return self._X

    @property
    def y(self):
        """The current training data labels/classification"""
        return self._y

    def pn(self, rule: AugmentedRule, force_complete_data: bool = False
           ) -> Tuple[int, int]:
        """Return positive and negative coverage (p, n) for `rule`. Cached.

        :param force_complete_data:
          Iff True, always use complete dataset (e.g. if growing/pruning
           datasets would be used otherwise).

        returns
        -------
        p : int
            The count of positive examples (== rule.head) covered by `rule`,
            also called *true positives*.
        n : int
            The count of negative examples (!= rule.head) covered by `rule`,
            also called *false positives*.
        """
        if force_complete_data not in rule._p_cache:
            rule._p_cache[force_complete_data] = \
                self._count_matches(rule, force_complete_data)
        covered_counts = rule._p_cache[force_complete_data].tolist()
        pos_index = np.take(
            np.argwhere(rule.head == self.theory_context.classes), 0)
        p = covered_counts.pop(pos_index)
        n = sum(covered_counts)
        return p, n

    def match_rule(self, rule: AugmentedRule, force_complete_data: bool = False
                   ) -> np.ndarray:
        """Apply `rule` to the current context (`self`), delegating to
        `SeCoAlgorithmConfiguration.match_rule`.

        :param force_complete_data: bool. If True, always apply to the full
          training data, instead of the property `self.X` (which may be
          overridden, e.g. by GrowPruneSplit).
        :return: A match array of dtype bool and length `len(X)`.
        """
        tctx = self.theory_context
        X = self._X if force_complete_data else self.X
        return tctx.algorithm_config.match_rule(X,
                                                rule.raw,
                                                tctx.categorical_mask)

    def _count_matches(self, rule: AugmentedRule,
                       force_complete_data: bool = False) -> np.ndarray:
        """:return: Example counts matched by `rule`, per class.
            Uncached, use `pn` instead.
        """
        covered = self.match_rule(rule, force_complete_data)
        y = self._y if force_complete_data else self.y
        cov_classes, cov_counts = np.unique(y[covered], return_counts=True)
        all_counts = np.zeros_like(self.theory_context.classes)
        cov_class_idx = np.searchsorted(self.theory_context.classes,
                                        cov_classes)
        all_counts[cov_class_idx] = cov_counts
        return all_counts

    def evaluate_rule(self, rule: AugmentedRule) -> None:
        """Rate rule to allow comparison & finding the best refinement."""
        growing_heuristic = \
            self.theory_context.implementation.growing_heuristic
        p, n = self.pn(rule)
        rule._sort_key = (growing_heuristic(rule, self),
                          # default tie-breaking: by positive coverage
                          p,
                          # and rule creation order (older = better)
                          -rule.instance_no)


class AbstractSecoImplementation(ABC):
    """The callbacks needed by :class:`_BinarySeCoEstimator`; Subclasses
    represent concrete algorithms (together with the corresponding subclasses
    of `RuleContext` etc, see :class:`SeCoAlgorithmConfiguration`).

    Fields
    -----
    direct_multiclass_support : bool, default True
        If any subclass does not support direct learning of multiclass
        theories, it must set this to False.

    TODO: maybe @classmethod → @staticmethod ? pro: callbacks as fun-refs w/o class, con: configurability via class-fields
    TODO: Instead of using this interface, you can also pass all the functions to SeCoEstimator separately, without an enclosing class.
    """

    direct_multiclass_support = True

    @classmethod
    def abstract_seco_continue(cls, y: np.ndarray,
                               theory_context: TheoryContext) -> bool:
        if theory_context.is_binary():
            positive = theory_context.classes[-1]
            return np.any(y == positive)
        else:
            default_class = theory_context.classes[0]
            return np.any(y != default_class)

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
        best one found.
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

    In `SeCoEstimator` an *instance* of this class is used, to allow wrapping
    the methods `make_rule`, `make_theory_context`, and `make_rule_context` in
    e.g. `functools.partialmethod`.

    The members `RuleClass`, `TheoryContextClass`, and `RuleContextClass`
    should only ever be overridden/subclassed, not wrapped in functions (like
    `functools.partial`), to enable other users (like `extra.trace_coverage`)
    to subclass them again.

    Methods provided are:
      - classmethod `direct_multiclass_support`
      - `make_rule`
      - `make_theory_context`
      - `make_rule_context`

    Fields
    -----
    `match_rule`: Callable[[np.ndarray, Rule, np.ndarray], np.ndarray].
        A method applying the Rule (2nd parameter) to the examples `X` (1st
        parameter) given the categorical_mask (3rd parameter) and returning an
        array of length `len(X)` and type bool.

    `Implementation`:
        A non-abstract subclass of `AbstractSecoImplementation` defining all
        callback methods needed by :class:`_BinarySeCoEstimator`.

    `RuleClass`:
        A subclass of `AugmentedRule` defining the attributes that supplement
        the basic `Rule` object.

    `TheoryContextClass`:
        A subclass of `TheoryContext` managing state of an `abstract_seco` run.

    `RuleContextClass`:
        A subclass of `RuleContext` managing state of a `find_best_rule` run.
    """

    match_rule = staticmethod(match_rule)
    # TODO: maybe use a Sequence of classes here and construct subclass with type(name, bases, []) in __init__
    Implementation: Type[AbstractSecoImplementation] = \
        AbstractSecoImplementation
    RuleClass = AugmentedRule
    TheoryContextClass: Type[TheoryContext] = TheoryContext
    RuleContextClass: Type[RuleContext] = RuleContext

    @classmethod
    def direct_multiclass_support(cls) -> bool:
        """
        :return: True iff the algorithm supports direct learning of multiclass
            theories, False otherwise.
        """
        return all(o.direct_multiclass_support
                   for o in (cls.Implementation,
                             cls.RuleClass,
                             cls.TheoryContextClass,
                             cls.RuleContextClass,
                             ))

    def __init__(self):
        self.implementation = self.Implementation()

    def make_rule(self, *args, **kwargs) -> 'RuleClass':
        """:return: An instance of `RuleClass`."""
        return self.RuleClass.make_empty(*args, **kwargs)

    def make_theory_context(self, *args, **kwargs) -> 'TheoryContextClass':
        """:return: An instance of `TheoryContextClass`."""
        return self.TheoryContextClass(self, *args, **kwargs)

    def make_rule_context(self, *args, **kwargs) -> 'RuleContextClass':
        """:return: An instance of `RuleContextClass`."""
        return self.RuleContextClass(*args, **kwargs)
