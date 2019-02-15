"""
Implementation of SeCo / Covering algorithm:
Usual building blocks & known instantiations of the abstract base algorithm.

Implemented as Mixins. For __init__ parameters, they use cooperative
multi-inheritance, each class has to declare **kwargs and forward anything it
doesn't consume using `super().__init__(**kwargs)`. Users of mixin-composed
classes will have to use keyword- instead of positional arguments.
"""

import math
from abc import abstractmethod, ABC
from functools import lru_cache
from typing import Iterable, NamedTuple, Dict, Tuple

import numpy as np
from scipy.special import xlogy
from sklearn.utils import check_random_state

from sklearn_seco.abstract import \
    Theory, SeCoEstimator
from sklearn_seco.common import \
    Rule, RuleQueue, AugmentedRule, LOWER, UPPER, log2, \
    AbstractSecoImplementation, RuleContext, TheoryContext, \
    SeCoAlgorithmConfiguration
from sklearn_seco.ripper_mdl import \
    data_description_length, relative_description_length


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    # copied from itertools docs
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def grow_prune_split(y,
                     target_class,
                     split_ratio: float,
                     rng):
    """
    Split `y` into grow and pruning set according to `split_ratio` for RIPPER.

    Make sure that `target_class` is included in the growing set, if it has
    little instances.

    Partially adapted from `sklearn.model_selection.StratifiedShuffleSplit`.

    :param split_ratio: float between [0,1].
      Size of pruning set.
    :param y: array-like of single dimension.
      The y/class values of all instances.
    :param target_class: dtype of `y`.
      The class which should always if possible be present in the growing set.
      To disable this behaviour, just pass a value which is not in `y`.
    :param rng: np.random.RandomState
      RNG to perform splitting.
    :return: tuple(np.ndarray, np.ndarray), both arrays of single dimension.
      `(grow, prune)` arrays which each list the indices into `y` for the
      respective set.
    """

    classes, y_indices, class_counts = \
        np.unique(y, return_inverse=True, return_counts=True)

    # handle target class first, to always round down its pruning count
    class_index = target_index = np.argwhere(classes == target_class)[0]
    n_prune = math.floor(class_counts[class_index] * split_ratio)
    y_indices_shuffled = \
        rng.permutation(np.nonzero(y_indices == class_index)[0]).tolist()
    prune = y_indices_shuffled[:n_prune]
    grow = y_indices_shuffled[n_prune:]

    for class_index, class_count in enumerate(class_counts):
        if class_index == target_index:
            continue  # handled separately beforehand
        n_prune = class_count * split_ratio
        # decide how to round (up/down) by looking at the previous bias
        ratio_pre = len(prune) / (len(prune) + len(grow))
        if ratio_pre > split_ratio:
            n_prune = math.floor(n_prune)
        else:
            n_prune = math.ceil(n_prune)

        y_indices_shuffled = \
            rng.permutation(np.nonzero(y_indices == class_index)[0]).tolist()
        prune.extend(y_indices_shuffled[:n_prune])
        grow.extend(y_indices_shuffled[n_prune:])

    grow = rng.permutation(grow).astype(int)
    prune = rng.permutation(prune).astype(int)
    return grow, prune


# Implementation facets


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
        return tctx.algorithm_config.make_rule(n_features=tctx.n_features)

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
        # TODO: maybe mark constant features for exclusion in future specializations

        for index in np.argwhere(categorical_mask
                                 & ~np.isfinite(rule.lower)  # unused features
                                 ).ravel():
            # argwhere returns each index in separate list, ravel() unpacks
            for value in all_feature_values(index):
                specialization = rule.copy()
                specialization.set_condition(LOWER, index, value)
                yield specialization

        for feature_index in np.nonzero(~categorical_mask)[0]:
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
                        specialization.set_condition(LOWER, feature_index,
                                                     new_threshold)
                        yield specialization
                # override is collation of upper bounds
                if no_old_upper or new_threshold < old_upper:
                    # don't test contradiction
                    if no_old_lower or new_threshold > old_lower:
                        specialization = rule.copy()
                        specialization.set_condition(UPPER, feature_index,
                                                     new_threshold)
                        yield specialization


class TopDownSearchContext(RuleContext):
    @lru_cache(maxsize=None)
    def all_feature_values(self, feature_index: int):
        """
        :return: All distinct values of feature (in examples) with given index,
             sorted.
        """
        # unique also sorts
        return np.unique(self.X[:, feature_index])


class PurityHeuristic(AbstractSecoImplementation):
    """The purity as rule evaluation metric: the percentage of positive
    examples among the examples covered by the rule.
    """

    @classmethod
    def growing_heuristic(cls, rule: AugmentedRule, context: RuleContext
                          ) -> float:
        p, n = context.count_matches(rule)
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
        p, n = context.count_matches(rule)
        return (p + 1) / (p + n + 2)  # laplace


class InformationGainHeuristic(AbstractSecoImplementation):
    """Information Gain heuristic as used in RIPPER (and FOIL).

    See (Quinlan, Cameron-Jones 1995) for the FOIL definition and
    (Witten,Frank,Hall 2011 fig 6.4) for its use in JRip/RIPPER.
    """

    @classmethod
    def growing_heuristic(cls, rule: AugmentedRule, context: RuleContext
                          ) -> float:
        p, n = context.count_matches(rule)
        P, N = context.PN
        # TODO: JRip (book fig 6.4) has P,N but (Fürnkranz 1999) has count_matches(rule.original)
        if p == 0:
            return 0
        # return p * (log2(p / (p + n)) - log2(P / (P + N)))  # info_gain
        return p * (log2(p) - log2(p + n) - log2(P) + log2(P + N))  # info_gain


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
        p, n = context.count_matches(rule)
        P, N = context.PN
        if 0 in (p, P, N):
            return True
        # purity = p / (p + n)
        # impurity = n / (p + n)
        # # cross entropy
        # CE = (- purity * math.log(purity / (P / (P + N)))
        #       - xlogy(impurity, impurity / (N / (P + N))))
        # # J-Measure
        # J = p * CE
        # # likelihood ratio statistics
        # LRS = 2 * (P + N) * J
        e_p = (p + n) * P / (P + N)
        e_n = (p + n) * N / (P + N)
        LRS = 2 * (xlogy(p, p / e_p) + xlogy(n, n / e_n))
        return LRS <= cls.LRS_threshold


class NoNegativesStop(AbstractSecoImplementation):
    """Inner stopping criterion: Abort refining when only positive examples are
    covered.

    NOTE: Since `find_best_rule` checks the filter *before* a refinement is
      processed any further, the straight forward implementation `return n ==
      0` would exclude very good rules, therefore we reject rule with `n == 0`
      whose predecessor already had `n == 0`. This may be wrong when not using
      `TopDownSearch`.
    """

    @classmethod
    def inner_stopping_criterion(cls, rule: AugmentedRule,
                                 context: RuleContext) -> bool:
        if not rule.original:
            return False
        cm = context.count_matches
        p, n = cm(rule)
        p_prev, n_prev = cm(rule.original)
        return n_prev == 0 and n == 0


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
    """A subclass of AugmentedRule recording any value set in `self.condition`.

    This is needed e.g. by `RipperPostPruning` to revert back to previous
    threshold conditions, because `TopDownSearchImplementation` collapses
    thresholds upon refinement.

    Attributes
    -----
    condition_trace: List[TraceEntry].
        Contains a tuple(UPPER/LOWER, feature_index, value, previous_value)
        for each value that was set in conditions, in the same order they were
        applied to the initially constructed rule to get to this rule.
    """

    class TraceEntry(NamedTuple):
        boundary: int
        index: int
        value: float
        old_value: float

    def __init__(self, *, conditions: Rule = None, n_features: int = None,
                 original: 'ConditionTracingAugmentedRule' = None):
        super().__init__(conditions=conditions,
                         n_features=n_features,
                         original=original)
        self.condition_trace = []
        if original:
            assert isinstance(original, ConditionTracingAugmentedRule)
            # copy trace, but into a new list
            self.condition_trace[:] = original.condition_trace

    def set_condition(self, boundary: int, index: int, value):
        self.condition_trace.append(
            self.TraceEntry(boundary, index, value,
                            self.conditions[boundary, index]))
        super().set_condition(boundary, index, value)


class GrowPruneSplitTheoryContext(TheoryContext):
    """`TheoryContext` needed for `GrowPruneSplitRuleContext`.

    TODO: find a way to render this class obsolete

    :param pruning_split_ratio: float between 0.0 and 1.0
      The relative size of the pruning set.

    :param grow_prune_random: None | int | instance of RandomState
      RNG to perform splitting. Passed to `sklearn.utils.check_random_state`.
    """
    def __init__(self, *args,
                 pruning_split_ratio: float = 1/3,
                 grow_prune_random=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pruning_split_ratio = pruning_split_ratio
        assert 0 <= pruning_split_ratio <= 1
        self.grow_prune_random = check_random_state(grow_prune_random)


class GrowPruneSplitRuleContext(ABC, RuleContext):
    """Implement a split of the examples into growing and pruning set once per
    iteration (i.e. for each `find_best_rule` call).

    The split is stratified (i.e. class proportions are retained).
    Note that JRip (and presumably the original RIPPER) state that their
    stratification is bugged.


    This works by overriding the getters for the properties `X`, `y`, `PN`, and
    the function `count_matches` (for `p, n`).
    By default this returns the growing set, i.e. at start of each
    `find_best_rule` invocation. Methods `pruning_heuristic` and
    As soon as the algorithm wants to switch to
    the pruning set (e.g. at the head of `simplify_rule`) it has to set
    `self.growing = False`.

    :var growing: bool
      If True (the default), let `self.X` and `self.y` return the growing set,
      otherwise the pruning set.
    """

    def __init__(self, theory_context: GrowPruneSplitTheoryContext, X, y):
        super().__init__(theory_context, X, y)
        assert isinstance(theory_context, GrowPruneSplitTheoryContext)
        self._PN_cache_growing = {}
        self.growing = True

        # split examples
        grow_idx, prune_idx = grow_prune_split(
            y,
            theory_context.target_class,
            theory_context.pruning_split_ratio,
            theory_context.grow_prune_random,)
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

    def count_matches(self, rule: AugmentedRule):
        """Cache values of (p,n) depending on the value of self.growing.

        To that end, `rule._pn_cache` is set to
        `Dict[growing: bool, (p: int, n: int)]` instead of `(p: int, n: int)`.
        """
        if rule._pn_cache is None or self.growing not in rule._pn_cache:
            pn_cache_old: Dict[bool, Tuple[int, int]] = rule._pn_cache or {}
            rule._pn_cache = None  # invalidate, so super (re)computes
            pn_cache_old[self.growing] = super().count_matches(rule)
            rule._pn_cache = pn_cache_old
        return rule._pn_cache[self.growing]

    @property
    def PN(self) -> Tuple[int, int]:
        """Cache values of (P, N) depending on the value of self.growing"""
        if self.growing not in self._PN_cache_growing:
            self._PN_cache = None  # invalidate, so super (re)computes
            self._PN_cache_growing[self.growing] = super().PN
        return self._PN_cache_growing[self.growing]

    def evaluate_rule(self, rule: AugmentedRule) -> None:
        """Mimic `AbstractSecoImplementation.evaluate_rule` but use
        `pruning_heuristic` while pruning.
        """
        pruning_heuristic = self.pruning_heuristic
        if self.growing:
            super().evaluate_rule(rule)
        else:
            p, n = self.count_matches(rule)
            # TODO: ensure rule._sort_key is not mixed. use incomparable types?
            rule._sort_key = (pruning_heuristic(rule, self),
                              p,
                              -rule.instance_no)

    @property
    def X(self):
        return self._growing_X if self.growing else self._pruning_X

    @property
    def y(self):
        return self._growing_y if self.growing else self._pruning_y


class RipperPostPruning(AbstractSecoImplementation):
    """Post-Pruning as performed by RIPPER, directly after growing
    (`find_best_rule`) each rule.
    """

    @classmethod
    def simplify_rule(cls, rule: ConditionTracingAugmentedRule,
                      context: GrowPruneSplitRuleContext) -> AugmentedRule:
        """Find the best simplification of `rule` by dropping conditions and
        evaluating using `pruning_heuristic`.

        NOTE: Overrides `AugmentedRule._sort_key` (i.e. the evaluation of
          the rule under the growing heuristic) with the pruning heuristic
          during the runtime of this method.

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
            generalization = generalization.copy()
            generalization.set_condition(boundary, index, old_value)
            evaluate_rule(generalization)
            candidates.append(generalization)

        candidates.sort()
        best_rule = candidates.pop()

        return best_rule


class CoverageRuleStop(AbstractSecoImplementation):
    """Rule stopping criterion. Stop if best rule doesn't cover more positive
    than negative examples (`p < n`).

    NOTE: The IREP-2 criterion `p/(p+n) <= 0.5` as defined in (Fürnkranz 1994)
      and used in RIPPER (Cohen 1995) is equivalent to `p <= n`.
    """

    @classmethod
    def rule_stopping_criterion(cls, theory: Theory, rule: AugmentedRule,
                                context: RuleContext) -> bool:
        p, n = context.count_matches(rule)
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
        p, n = context.count_matches(rule)
        return p < cls.positive_coverage_stop_threshold


class RipperMdlRuleStopImplementation(AbstractSecoImplementation):
    """MDL (minimum description length) stopping criterion used by RIPPER.

    Abort search if the last found rule has a `description_length_` higher than
    the best so far + some margin (`description_length_surplus`), or `p=0` or
    (if `check_error_rate is True`) `n >= p`.

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
                                context: RuleContext) -> bool:
        tctx = context.theory_context
        assert isinstance(tctx, RipperMdlRuleStopTheoryContext)

        p, n = context.count_matches(rule)
        P, N = context.PN
        tctx.theory_pn.append((P, N))

        tctx.description_length_ += relative_description_length(
            rule, tctx.expected_fp_over_err, p, n, P, N, tctx.theory_pn)
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
    def __init__(self, algorithm_config, categorical_mask, n_features,
                 target_class, X, y, **kwargs):
        super().__init__(algorithm_config, categorical_mask, n_features,
                         target_class, X, y, **kwargs)

        positives = np.count_nonzero(y == self.target_class)
        # set expected fp/(err = fp+fn) rate := proportion of the class
        self.expected_fp_over_err = positives / len(y)
        # set default DL (only data, empty theory)
        self.description_length_ = self.best_description_length_ = \
            data_description_length(
                expected_fp_over_err=self.expected_fp_over_err,
                covered=0, uncovered=len(y), fp=0, fn=positives)
        self.theory_pn = []


# Example Algorithm configurations


class SimpleSeCoEstimator(SeCoEstimator):
    class algorithm_config(SeCoAlgorithmConfiguration):
        RuleContextClass = TopDownSearchContext

        class Implementation(BeamSearch,
                             TopDownSearchImplementation,
                             PurityHeuristic,
                             NoNegativesStop,
                             SkipPostPruning,
                             CoverageRuleStop,
                             SkipPostProcess):
            pass


class CN2Estimator(SeCoEstimator):
    """CN2 as refined by (Clark and Boswell 1991)."""

    class algorithm_config(SeCoAlgorithmConfiguration):
        RuleContextClass = TopDownSearchContext

        class Implementation(BeamSearch,
                             TopDownSearchImplementation,
                             LaplaceHeuristic,
                             SignificanceStoppingCriterion,
                             SkipPostPruning,
                             PositiveThresholdRuleStop,
                             SkipPostProcess):
            positive_coverage_stop_threshold = 1  # → PositiveThresholdRuleStop


class RipperEstimator(SeCoEstimator):
    """Ripper as defined by (Cohen 1995).

    NOTE: The global post-optimization phase is currently not implemented
        (that would be the `post_process` method).
    """
    class algorithm_config(SeCoAlgorithmConfiguration):
        RuleClass = ConditionTracingAugmentedRule

        class Implementation(BeamSearch,
                             TopDownSearchImplementation,
                             InformationGainHeuristic,
                             RipperMdlRuleStopImplementation,
                             RipperPostPruning,
                             SkipPostProcess
                             ):
            @classmethod
            def inner_stopping_criterion(cls, rule: AugmentedRule,
                                         context: RuleContext) -> bool:
                """Laplace-based criterion. Field `accuRate` in JRip.java."""
                p, n = context.count_matches(rule)
                accuracy_rate = (p + 1) / (p + n + 1)
                return accuracy_rate >= 1

            @classmethod
            def rule_stopping_criterion(cls, theory: Theory,
                                        rule: AugmentedRule,
                                        context: RuleContext) -> bool:
                assert isinstance(context, GrowPruneSplitRuleContext)
                # TODO: dedup setting context.growing before callbacks. use a decorator?
                context.growing = False
                return super().rule_stopping_criterion(theory, rule, context)

        class RuleContextClass(TopDownSearchContext,
                               GrowPruneSplitRuleContext):
            def pruning_heuristic(self, rule: AugmentedRule,
                                  context: RuleContext
                                  ) -> float:
                """Laplace heuristic, as defined by (Clark and Boswell 1991).

                JRip documentation states:
                "The pruning metric is (p-n)/(p+n) -- but it's actually
                2p/(p+n) -1, so in this implementation we simply use p/(p+n)
                (actually (p+1)/(p+n+2), thus if p+n is 0, it's 0.5)."
                """
                p, n = context.count_matches(rule)
                return (p + 1) / (p + n + 2)  # laplace

        class TheoryContextClass(GrowPruneSplitTheoryContext,
                                 RipperMdlRuleStopTheoryContext):
            pass


class IrepEstimator(SeCoEstimator):
    """IREP as defined by (Cohen 1995), originally by (Fürnkranz, Widmer 1994).
    """

    class algorithm_config(SeCoAlgorithmConfiguration):
        RuleClass = ConditionTracingAugmentedRule

        class Implementation(BeamSearch,
                             TopDownSearchImplementation,
                             InformationGainHeuristic,
                             NoNegativesStop,
                             RipperPostPruning,
                             CoverageRuleStop,
                             SkipPostProcess):

            @classmethod
            def rule_stopping_criterion(cls, theory: Theory,
                                        rule: AugmentedRule,
                                        context: RuleContext) -> bool:
                assert isinstance(context, GrowPruneSplitRuleContext)
                context.growing = False
                return super().rule_stopping_criterion(theory, rule, context)

        class RuleContextClass(TopDownSearchContext,
                               GrowPruneSplitRuleContext):
            def pruning_heuristic(self, rule: AugmentedRule,
                                  context: GrowPruneSplitRuleContext) -> float:
                """:return: (#true positives + #true negatives) / #examples"""
                context.growing = False
                p, n = context.count_matches(rule)
                P, N = context.PN
                tn = N - n
                return (p + tn) / (P + N)

        TheoryContextClass = GrowPruneSplitTheoryContext


# TODO: sklearn.get/set_param setting *Implementation fields?
# TODO: allow defining heuristics/metrics (and stop criteria?) as functions and pulling them in as growing_/pruning_heuristic etc without defining an extra class
