"""
Implementation of SeCo / Covering algorithm:
Usual building blocks of SeCo algorithms, implemented as Mixins.

For __init__ parameters, they use cooperative multi-inheritance: Each class has
to declare **kwargs and forward anything it doesn't consume using
`super().__init__(**kwargs)`. Creators of mixin-composed classes will have to
use keyword- instead of positional arguments.
"""

import functools
import math
from abc import abstractmethod, ABC
from functools import lru_cache
from typing import Iterable, NamedTuple, Tuple, Any, Callable

import numpy as np
from scipy.special import xlogy

from sklearn_seco.abstract import Theory
from sklearn_seco.common import \
    Rule, RuleQueue, AugmentedRule, T, TGT, \
    AbstractSecoImplementation, RuleContext, TheoryContext
from sklearn_seco.ripper_mdl import \
    data_description_length, relative_description_length
from sklearn_seco.util import log2, pairwise


def grow_prune_split(y,
                     split_ratio: float,
                     rng):
    """
    Split `y` into grow and pruning set according to `split_ratio` for RIPPER.

    Make sure that every class is included in the growing set, even if it has
    little instances.
    # TODO: split undefined if too few examples => pruning set will be empty
    # TODO: if is_binary() <==> concept learning, then round #negative down

    Partially adapted from `sklearn.model_selection.StratifiedShuffleSplit`.

    :param split_ratio: float between [0,1].
      Size of pruning set.
    :param y: array-like of single dimension.
      The y/class values of all instances.
    :param rng: np.random.RandomState
      RNG to perform splitting.
    :return: tuple(np.ndarray, np.ndarray), both arrays of single dimension.
      `(grow, prune)` arrays which each list the indices into `y` for the
      respective set.
    """

    classes, y_indices, class_counts = \
        np.unique(y, return_inverse=True, return_counts=True)

    grow = []
    prune = []
    for class_index, class_count in enumerate(class_counts):
        n_prune = math.floor(class_count * split_ratio)
        y_indices_shuffled = \
            rng.permutation(np.nonzero(y_indices == class_index)[0]).tolist()
        prune.extend(y_indices_shuffled[:n_prune])
        grow.extend(y_indices_shuffled[n_prune:])

    grow = rng.permutation(grow).astype(int)
    prune = rng.permutation(prune).astype(int)
    return grow, prune


class BeamSearch(AbstractSecoImplementation):
    """Mixin implementing a beam search of width `n`.

    - The default `beam_width` of 1 signifies a hill climbing search, only ever
      optimizing one candidate rule.
    - A special `beam_width` of 0 means trying all candidate rules, i.e. a
      best-first search.

    Rule selection is done in `filter_rules`, while `select_candidate_rules`
    always returns the whole queue as candidates.
    """
    beam_width: int = 1

    @classmethod
    def select_candidate_rules(cls, rules: RuleQueue, context: RuleContext
                               ) -> Iterable[AugmentedRule]:
        # pop all items from rules, retaining the reference
        # (so caller also sees the empty list)
        candidates, rules[:] = rules[:], []
        return candidates

    @classmethod
    def filter_rules(cls, rules: RuleQueue, context: RuleContext) -> RuleQueue:
        # negative to get the last items, the best.
        # special case -0 equals 0 and gets the whole queue
        return rules[-cls.beam_width:]


class TopDownSearchImplementation(AbstractSecoImplementation):
    """Mixin providing a Top-Down rule search: initializes an empty rule and
    subsequently specializes it.
    """

    @classmethod
    def init_rule(cls, context: RuleContext) -> AugmentedRule:
        tctx = context.theory_context
        head = tctx.n_classes - 1 if tctx.is_binary() else 0
        return tctx.algorithm_config.make_rule(
            n_features=tctx.n_features,
            target_class=head)

    @classmethod
    def refine_rule(cls, rule: AugmentedRule, context: 'TopDownSearchContext'
                    ) -> Iterable[AugmentedRule]:
        """Create refinements of `rule` by adding a test for each of the
        (unused) attributes, one at a time, trying all possible attribute
        values / thresholds.
        """
        assert isinstance(context, TopDownSearchContext)
        all_feature_values = context.all_feature_values
        categorical_mask = context.theory_context.categorical_mask
        # TODO: maybe mark constant features (or with p < threshold) for exclusion in future specializations

        def specialize(boundary: int, index: int, value
                       ) -> Iterable[AugmentedRule]:
            if context.theory_context.is_binary():
                # in binary case, only emit rules for target_class,
                # i.e. do concept learning. See `_BaseSeCoEstimator.fit`.
                classes = [context.theory_context.n_classes - 1]
            else:
                # TODO: maybe yield only rule for class with maximum p
                classes = range(context.theory_context.n_classes)

            for target_class in classes:
                P, N = context.PN(target_class)
                if not P:
                    continue
                yield rule.copy(head=target_class,
                                condition=(boundary, index, value))

        # categorical features
        for index in np.argwhere(categorical_mask
                                 & ~np.isfinite(rule.lower)  # unused features
                                 ).ravel():
            # argwhere returns each index in separate list, ravel() unpacks
            for value in all_feature_values(index):
                yield from specialize(Rule.LOWER, index, value)

        # numeric features
        for feature_index in np.nonzero(~categorical_mask)[0]:
            old_lower = rule.lower[feature_index]
            old_upper = rule.upper[feature_index]
            for value1, value2 in pairwise(all_feature_values(feature_index)):
                new_threshold = (value1 + value2) / 2  # TODO: JRip uses left value1
                # TODO: lower/upper coverage here are complementary: can use single match_rule invocation, like JRip?
                # TODO: maybe only split if classes of value1,value2 differ
                # override is collation of lower bounds
                if new_threshold > old_lower:
                    # don't test contradiction (e.g. f < 4 && f > 6)
                    if new_threshold < old_upper:
                        yield from specialize(Rule.LOWER, feature_index,
                                              new_threshold)
                # override is collation of upper bounds
                if new_threshold < old_upper:
                    # don't test contradiction
                    if new_threshold > old_lower:
                        yield from specialize(Rule.UPPER, feature_index,
                                              new_threshold)


class TopDownSearchContext(RuleContext):
    @lru_cache(maxsize=None)
    def all_feature_values(self, feature_index: int) -> Iterable:
        """
        :return: All distinct values of feature (in examples) with given index,
             sorted.
        """
        # use _X instead of self.X to get all possibilities, even if e.g.
        # grow/prune split is used

        # unique also sorts
        distinct_features = np.unique(self._X[:, feature_index])
        # filter missing values.  TODO: "== missing" test on nominal features
        distinct_features = distinct_features[~np.isnan(distinct_features)]
        # tolist needed because for pypy3 7.1 the ndarray is not an iterator
        return distinct_features.tolist()


class PurityHeuristic(AbstractSecoImplementation):
    """The purity as rule evaluation metric: the percentage of positive
    examples among the examples covered by the rule.
    """

    @classmethod
    def growing_heuristic(cls, rule: AugmentedRule, context: RuleContext
                          ) -> float:
        p, n = rule.pn(context)
        if p + n == 0:
            return 0
        return p / (p + n)  # purity


class LaplaceHeuristic(AbstractSecoImplementation):
    """The Laplace rule evaluation metric.

    The Laplace estimate was defined by (Clark and Boswell 1991) for CN2.
    """

    @classmethod
    def growing_heuristic(cls, rule: AugmentedRule, context: RuleContext
                          ) -> float:
        """Laplace heuristic, as defined by (Clark and Boswell 1991)."""
        p, n = rule.pn(context)
        return (p + 1) / (p + n + 2)  # laplace


def gain_heuristic(heuristic: Callable) -> Callable:
    """Decorator for `AbstractSecoImplementation.growing_heuristic` if that is a
    *gain heuristic*, i.e. computes the gain of a rule compared to
    `rule.original`.

    Since `find_best_rule` compares rules across the whole search tree, using a
    gain heuristic as-is leads to errors (e.g. a rule A with high gain is chosen
    over its refinements R even if those improve upon it a bit, when their gain
    upon A is less than the gain of A upon its base).

    As a remedy, we sum the gain along the refinement path.

    Note: Should also be usable with
      `GrowPruneSplitRuleContext.pruning_heuristic`, but that was not tested.
    """
    @functools.wraps(heuristic)
    def as_gain_heuristic(cls, rule: AugmentedRule, context: RuleContext
                          ) -> float:
        value = heuristic(cls, rule, context)
        if rule.original:
            return value + heuristic(cls, rule.original, context)
        return value
    return as_gain_heuristic


class InformationGainHeuristic(AbstractSecoImplementation):
    """Information Gain heuristic as used in RIPPER (and FOIL).

    See (Quinlan, Cameron-Jones 1995) for the FOIL definition and
    (Witten,Frank,Hall 2011 fig 6.4) for its use in JRip/RIPPER.

    Note that the papers all have `p * ( log(p/(p+n)) - log(P/(P+N)) )`,
      whereas JRip and ripper.c implement
      `p * ( log((p+1) / (p+n+1)) - log((P+1) / (P+N+1)))`.

    Note also what is used for P,N: (Frank,Hall,Witten 2011) fig 6.4 has always
      P,N (i.e. of current X,y),
      but ripper.c, JRip, and (Fürnkranz 1999) have rule.original.pn
    """

    @classmethod
    @gain_heuristic
    def growing_heuristic(cls, rule: AugmentedRule, context: RuleContext
                          ) -> float:
        p, n = rule.pn(context)
        if rule.original:
            P, N = rule.original.pn(context)
        else:
            P, N = context.PN(rule.head)
        if p == 0:
            return 0
        return p * (log2(p / (p + n)) - log2(P / (P + N)))  # info_gain


class SignificanceStoppingCriterion(AbstractSecoImplementation):
    """Rule stopping criterion using a significance test like CN2."""

    LRS_threshold: float = 0.0

    @classmethod
    def inner_stopping_criterion(cls, rule: AugmentedRule,
                                 context: RuleContext) -> bool:
        """
        *Significance test* as defined by (Clark and Niblett 1989), but used
        there for rule evaluation, here instead used as stopping criterion
        following (Clark and Boswell 1991).
        """
        p, n = rule.pn(context)
        P, N = context.PN(rule.head)
        if p == 0:
            return True
        # purity = p / (p + n)
        # impurity = n / (p + n)
        # CE = (- purity * math.log(purity / (P / (P + N)))
        #       - xlogy(impurity, impurity / (N / (P + N))))  # cross entropy
        # J = p * CE  # J-Measure
        # LRS = 2 * (P + N) * J  # likelihood ratio statistics
        e_p = (p + n) * P / (P + N)
        e_n = (p + n) * N / (P + N)
        LRS_p = xlogy(p, p / e_p) if e_p != 0 else 0  # exclude if P == 0
        LRS_n = xlogy(n, n / e_n) if e_n != 0 else 0  # exclude if N == 0
        LRS = 2 * (LRS_p + LRS_n)
        return LRS <= cls.LRS_threshold


def delayed_inner_stop(inner_stop):
    """Decorator for `AbstractSecoImplementation.inner_stopping_criterion`,
    letting it stop only when the condition meets for a rule and its
    predecessor (`rule.original`).

    Since `find_best_rule` checks the inner_stopping_criterion filter *before*
    a refinement is processed any further, this would exclude very good rules.
    This decorator is equivalent to checking the filter *afterwards*.
    This may be wrong when not using `TopDownSearch`.
    """
    @functools.wraps(inner_stop)
    def delaying_inner_stop(cls, rule: AugmentedRule, context: RuleContext):
        if not rule.original:
            return False
        return (inner_stop(cls, rule.original, context)
                and inner_stop(cls, rule, context))
    return delaying_inner_stop


class NoNegativesStop(AbstractSecoImplementation):
    """Inner stopping criterion: Abort refining when only positive examples are
    covered.
    """

    @classmethod
    @delayed_inner_stop
    def inner_stopping_criterion(cls, rule: AugmentedRule,
                                 context: RuleContext) -> bool:
        p, n = rule.pn(context)
        return n == 0


class SkipPostPruning(AbstractSecoImplementation):
    """Mixin to skip post-pruning"""
    @classmethod
    def simplify_rule(cls, rule: AugmentedRule, context: RuleContext
                      ) -> AugmentedRule:
        return rule


class SkipPostProcess(AbstractSecoImplementation):
    """Mixin to skip post processing."""
    @classmethod
    def post_process(cls, theory: Theory, context: TheoryContext) -> Theory:
        return theory


class ConditionTracingAugmentedRule(AugmentedRule):
    """A subclass of AugmentedRule recording any value set in `self.body`.

    This is needed e.g. by `RipperPostPruning` to revert back to previous
    threshold conditions, because `TopDownSearchImplementation` collapses
    thresholds upon refinement.

    Attributes
    -----
    condition_trace: List[ConditionTracingAugmentedRule.TraceEntry].
        Contains a tuple(UPPER/LOWER, feature_index, value, previous_value)
        for each value that was set in `_conditions`, in the same order they
        were applied to the initially constructed rule to get to this rule.
    """

    class TraceEntry(NamedTuple):
        boundary: int
        index: int
        value: float
        old_value: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_trace = []
        if self.original:
            assert isinstance(self.original, ConditionTracingAugmentedRule)
            # copy trace, but into a new list
            self.condition_trace[:] = self.original.condition_trace

    def copy(self: T, *, head: TGT = None,
             condition: Tuple[int, int, Any] = None) -> T:
        copy: T = super().copy(head=head, condition=condition)
        if condition is not None:
            boundary, index, value = condition
            copy.condition_trace.append(
                ConditionTracingAugmentedRule.TraceEntry(
                    boundary, index, value, self.body[boundary, index]))
        return copy


class GrowPruneSplitRuleContext(ABC, RuleContext):
    """Implement a split of the examples into growing and pruning set once per
    iteration (i.e. for each `find_best_rule` call).

    The split is stratified (i.e. class proportions are retained).
    Note that JRip (and presumably the original RIPPER) state that their
    stratification is bugged.


    This works by overriding the getters for the properties `X`, `y`, `PN`, and
    the function `pn`.
    By default, i.e. at start of each `find_best_rule` invocation, this uses the
    growing set. As soon as the algorithm wants to switch to
    the pruning set (e.g. at the head of `simplify_rule`) it has to set
    `self.growing = False`.

    # TODO: dedup setting context.growing before callbacks. use a decorator?

    :var growing: bool
      If True (the default), let `self.X` and `self.y` return the growing set,
      if False the pruning set, if None the whole training set.

    :param pruning_split_ratio: float between 0.0 and 1.0
      The relative size of the pruning set.
    """

    pruning_split_ratio: float = 1 / 3

    def __init__(self, theory_context: TheoryContext, X, y):
        super().__init__(theory_context, X, y)
        self.growing = True

        # split examples
        grow_idx, prune_idx = grow_prune_split(y, self.pruning_split_ratio,
                                               theory_context.rng)
        self._growing_X = X[grow_idx]
        self._growing_y = y[grow_idx]
        self._pruning_X = X[prune_idx]
        self._pruning_y = y[prune_idx]

    @abstractmethod
    def pruning_heuristic(self, rule: AugmentedRule, context: RuleContext
                          ) -> float:
        """Rate rule to allow finding the best refinement, while pruning."""

        # actually set this in `AbstractSecoImplementation.simplify_rule`
        self.growing = False

    def pn(self, rule: AugmentedRule, force_complete_data: bool = False):
        """Return (p, n) for `rule`, depending on `self.growing`. Cached."""

        # the logic is the same in super(), but _p_cache keys are
        # growing = True|False|None, where None means all samples
        growing = None if force_complete_data else self.growing
        # TODO: maybe calculate from cache, given total=grow+prune
        if growing not in rule._p_cache:
            rule._p_cache[growing] = self._count_matches(rule,
                                                         force_complete_data)
        covered_counts = rule._p_cache[growing].tolist()
        p = covered_counts.pop(rule.head)
        n = sum(covered_counts)
        return p, n

    def PN(self, target_class: TGT, force_complete_data: bool = False
           ) -> Tuple[int, int]:
        """:return: (P, N) depending on the value of self.growing. Cached.

        See superclass method.
        """
        growing = None if force_complete_data else self.growing
        if growing not in self._PN_cache:
            y = self._y if force_complete_data else self.y
            # TODO: maybe calculate from cache
            self._PN_cache[growing] = self._count_PN(y)

        class_counts = self._PN_cache[growing].tolist()
        P = class_counts.pop(target_class)
        N = sum(class_counts)
        return P, N

    def evaluate_rule(self, rule: AugmentedRule) -> None:
        """Rate rule like `AbstractSecoImplementation.evaluate_rule` but use
        `pruning_heuristic` while pruning.
        """
        if self.growing:
            super().evaluate_rule(rule)
        else:
            rule._pruning_heuristic = self.pruning_heuristic(rule, self)

    def sort_key(self, rule: 'AugmentedRule'):
        if self.growing:
            return super().sort_key(rule)

        assert hasattr(rule, '_pruning_heuristic')
        p, n = self.pn(rule)
        return (rule._pruning_heuristic,
                # tie-breaking: by positive coverage
                p,
                # and rule age
                -rule.instance_no)

    @property
    def X(self):
        if self.growing is None:
            return self._X
        elif self.growing:
            return self._growing_X
        else:
            return self._pruning_X

    @property
    def y(self):
        if self.growing is None:
            return self._y
        elif self.growing:
            return self._growing_y
        else:
            return self._pruning_y


class RipperPostPruning(AbstractSecoImplementation):
    """Post-Pruning as performed by RIPPER, directly after growing
    (`find_best_rule`) each rule.
    """

    @classmethod
    def simplify_rule(cls, rule: ConditionTracingAugmentedRule,
                      context: GrowPruneSplitRuleContext) -> AugmentedRule:
        """Find the best simplification of `rule` by dropping conditions and
        evaluating using `pruning_heuristic`.

        :return: An improved version of `rule`.
        """
        assert isinstance(rule, ConditionTracingAugmentedRule)
        assert isinstance(context, GrowPruneSplitRuleContext)
        evaluate_rule = context.evaluate_rule

        context.growing = False  # tell GrowPruneSplit to use pruning set
        evaluate_rule(rule)
        candidates = [rule]
        # drop any final (i.e. last added) sequence of conditions
        generalization = rule
        for boundary, index, value, old_value in reversed(rule.condition_trace):
            generalization = generalization.copy(
                condition=(boundary, index, old_value))
            evaluate_rule(generalization)
            candidates.append(generalization)

        candidates.sort(key=context.sort_key)
        best_rule = candidates.pop()

        return best_rule


class CoverageRuleStop(AbstractSecoImplementation):
    """Rule stopping criterion. Stop if best rule covers more negative than
    positive examples (`p < n`).

    NOTE: The IREP-2 criterion `p/(p+n) <= 0.5` as defined in (Fürnkranz 1994)
      and used in RIPPER (Cohen 1995) is equivalent to `p <= n`.
    """

    @classmethod
    def rule_stopping_criterion(cls, theory: Theory, rule: AugmentedRule,
                                context: RuleContext) -> bool:
        p, n = rule.pn(context)
        return p < n


class PositiveThresholdRuleStop(AbstractSecoImplementation):
    """Rule stopping criterion. Stop if best rule covers less than `threshold`
    positive examples.
    """

    positive_coverage_stop_threshold: int

    @classmethod
    def rule_stopping_criterion(cls, theory: Theory, rule: AugmentedRule,
                                context: RuleContext) -> bool:
        """Abort search if rule covers less than `threshold` positive examples.

        If threshold == 1, this corresponds to the "E is empty" condition in
        Table 3 of (Clark and Niblett 1989) used by CN2.
        """
        p, n = rule.pn(context)
        return p < cls.positive_coverage_stop_threshold


class RipperMdlRuleStopImplementation(AbstractSecoImplementation):
    """MDL (minimum description length) stopping criterion used by RIPPER.

    Abort search if the last found rule has a `description_length_` higher than
    the best so far + some margin (`description_length_surplus`), or `p=0` or
    (if `check_error_rate is True`) `n >= p`.

    The description length is only defined for binary classification tasks, so
    use of this criterion prevents direct multiclass learning.

    NOTE: Reconstructed mainly from JRip.java source, no guarantee on all
      details being correct and identical to other implementation(s).

    Fields
    =====
    - `best_description_length_`: The minimal DL found in the current search so
      far, named `minDL` in JRip.
    - `description_length_`: The DL of the current theory, named `dl` in JRip.
    """

    check_error_rate: bool = True
    description_length_surplus: int = 64

    @classmethod
    def rule_stopping_criterion(cls, theory: Theory, rule: AugmentedRule,
                                context: GrowPruneSplitRuleContext) -> bool:
        tctx = context.theory_context
        assert isinstance(tctx, RipperMdlRuleStopTheoryContext)
        assert isinstance(context, GrowPruneSplitRuleContext)

        context.growing = None
        p, n = rule.pn(context)
        P, N = context.PN(rule.head)
        tctx.theory_pn.append((p, n))

        tctx.description_length_ += relative_description_length(
            rule, tctx.expected_fp_over_err, p, n, P, N, tctx.theory_pn,
            tctx.max_n_conditions)
        tctx.best_description_length_ = min(tctx.best_description_length_,
                                            tctx.description_length_)
        if tctx.description_length_ > (tctx.best_description_length_ +
                                       cls.description_length_surplus):
            return True
        if p <= 0:
            return True
        if cls.check_error_rate and n >= p:  # error rate
            # JRip has `(n / (p + n)) >= 0.5` which is equivalent
            return True
        return False


class RipperMdlRuleStopTheoryContext(TheoryContext):
    direct_multiclass_support = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.is_binary()

        positives = np.count_nonzero(self.complete_y == self.n_classes - 1)
        # set expected fp/(err = fp+fn) rate := proportion of the class
        self.expected_fp_over_err = positives / len(self.complete_y)
        # set default DL (only data, empty theory)
        self.description_length_ = self.best_description_length_ = \
            data_description_length(
                expected_fp_over_err=self.expected_fp_over_err,
                covered=0, uncovered=len(self.complete_y), fp=0, fn=positives)
        self.max_n_conditions = sum(
            self._num_unique_values(feature)
            * (1 if self.categorical_mask[feature] else 2)
            for feature in range(self.n_features)
        )
        self.theory_pn = []

    def _num_unique_values(self, feature: int):
        """
        :return: The count of distinct values for the given feature, excluding
            missing values.

        For weka.JRip, this is implemented in `Instances.numDistinctValues()`
        and `Attribute.numValues()`.
        """
        unique_values = np.unique(self.complete_X[:, feature])
        unique_values = unique_values[~np.isnan(unique_values)]
        return len(unique_values)


# TODO: sklearn.get/set_param setting *Implementation fields?
# TODO: allow defining heuristics/metrics (and stop criteria?) as functions and pulling them in as growing_/pruning_heuristic etc without defining an extra class
