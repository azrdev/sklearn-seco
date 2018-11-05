"""
Implementation of SeCo / Covering algorithm:
Usual building blocks & known instantiations of the abstract base algorithm.

Implemented as Mixins. For __init__ parameters, they use cooperative
multi-inheritance, each class has to declare **kwargs and forward anything it
doesn't consume using `super().__init__(**kwargs)`. Users of mixin-composed
classes will have to use keyword- instead of positional arguments.
"""
import math
from abc import abstractmethod
from functools import lru_cache
from typing import Tuple, Iterable, SupportsFloat

import numpy as np
from scipy.special import xlogy
from sklearn.utils import check_random_state

from sklearn_seco.abstract import Theory, SeCoEstimator, _BinarySeCoEstimator
from sklearn_seco.common import \
    RuleQueue, SeCoBaseImplementation, AugmentedRule, LOWER, UPPER


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    # copied from itertools docs
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def log2(x: SupportsFloat) -> float:
    """`log2(x) if x > 0 else 0`"""
    return math.log2(x) if x > 0 else 0.0


# Mixins providing implementation facets


class BeamSearch(SeCoBaseImplementation):
    """Mixin implementing a beam search of width `n`.

    - The default `beam_width` of 1 signifies a hill climbing search, only ever
      optimizing one candidate rule.
    - A special `beam_width` of 0 means trying all candidate rules, i.e. a
      best-first search.

    Rule selection is done in `filter_rules`, while `select_candidate_rules`
    always returns the whole queue as candidates.
    """
    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width_ = beam_width

    def select_candidate_rules(self, rules: RuleQueue
                               ) -> Iterable[AugmentedRule]:
        # pop all items from rules, retaining the reference
        # (so caller also sees the empty list)
        candidates, rules[:] = rules[:], []
        return candidates

    def filter_rules(self, rules: RuleQueue) -> RuleQueue:
        # negative to get the last items, the best.
        # special case -0 equals 0 and gets the whole queue
        return rules[-self.beam_width_:]


class TopDownSearch(SeCoBaseImplementation):
    """Mixin providing a Top-Down rule search: initializes an empty rule and
    subsequently specializes it.
    """

    def set_context(self, estimator: _BinarySeCoEstimator, X, y):
        super().set_context(estimator, X, y)
        # cached results depend on examples (X, y), which change each iteration
        self.all_feature_values.cache_clear()

    def unset_context(self):
        super().unset_context()
        self.all_feature_values.cache_clear()

    @lru_cache(maxsize=None)
    def all_feature_values(self, feature_index: int):
        """
        :return: All distinct values of feature (in examples) with given index,
             sorted.
        """
        # unique also sorts
        return np.unique(self.X[:, feature_index])

    def init_rule(self) -> AugmentedRule:
        return AugmentedRule(n_features=self.n_features)

    def refine_rule(self, rule: AugmentedRule) -> Iterable[AugmentedRule]:
        all_feature_values = self.all_feature_values
        # TODO: mark constant features for exclusion in future specializations

        for index in np.argwhere(self.categorical_mask
                                 & ~np.isfinite(rule.lower)  # unused features
                                 ).ravel():
            # argwhere returns each index in separate list, ravel() unpacks
            for value in all_feature_values(index):
                specialization = rule.copy()
                specialization.set_condition(LOWER, index, value)
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


class PurityHeuristic(SeCoBaseImplementation):
    """Mixin providing the purity as rule evaluation metric: the percentage of
    positive examples among the examples covered by the rule.
    """

    def evaluate_rule(self, rule: AugmentedRule) -> Tuple[float, float, int]:
        p, n = self.count_matches(rule)
        if p + n == 0:
            return (0, p, -rule.instance_no)
        purity = p / (p + n)
        # tie-breaking by pos. coverage and rule creation order: older = better
        return (purity, p, -rule.instance_no)


class LaplaceHeuristic(SeCoBaseImplementation):
    """Mixin implementing the Laplace rule evaluation metric.

    The Laplace estimate was defined by (Clark and Boswell 1991) for CN2.
    """
    def evaluate_rule(self, rule: AugmentedRule) -> Tuple[float, float, int]:
        """Laplace heuristic, as defined by (Clark and Boswell 1991)."""
        p, n = self.count_matches(rule)
        laplace = (p + 1) / (p + n + 2)
        return (laplace, p, -rule.instance_no)  # tie-breaking by pos. coverage


class InformationGainHeuristic(SeCoBaseImplementation):
    def evaluate_rule(self, rule: AugmentedRule):
        p, n = self.count_matches(rule)
        P, N = self.P, self.N  # TODO: maybe count_matches(rule.original) is meant here? book fig6.4 says code is correct
        # info_gain = p * (log2(p / (p + n)) - log2(P / (P + N)))
        info_gain = p * (log2(p) - log2(p + n) - log2(P) + log2(P + N)) \
            if p > 0 else 0
        # tie-breaking by positive coverage p and rule discovery order
        return (info_gain, p, -rule.instance_no)


class SignificanceStoppingCriterion(SeCoBaseImplementation):
    """Mixin using as stopping criterion for rule refinement a significance
    test like CN2.
    """
    def __init__(self, *,  LRS_threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.LRS_threshold = LRS_threshold  # FIXME: estimator.set_param not reflected here

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        """
        *Significance test* as defined by (Clark and Niblett 1989), but used
        there for rule evaluation, here instead used as stopping criterion
        following (Clark and Boswell 1991).
        """
        p, n = self.count_matches(rule)
        P, N = self.P, self.N
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
        return LRS <= self.LRS_threshold


class NoPostPruning(SeCoBaseImplementation):
    """Mixin to skip post-pruning"""
    def simplify_rule(self, rule: AugmentedRule) -> AugmentedRule:
        return rule


class NoPostProcess(SeCoBaseImplementation):
    """Mixin to skip post processing."""
    def post_process(self, theory: Theory) -> Theory:
        return theory


class GrowPruneSplit(SeCoBaseImplementation):
    """Implement a split of the examples into growing and pruning set once per
    iteration (i.e. for each `find_best_rule` call).

    This works by overriding the getters for the properties `X` and `y`.
    By default, and after each reset of the training data (i.e. `set_context`)
    this returns the growing set. As soon as the algorithm wants to switch to
    the pruning set (e.g. at the head of `simplify_rule`) it has to set
    `self.growing = False`.

    :param pruning_split_ratio: float between 0.0 and 1.0
      The relative size of the pruning set.

    :param grow_prune_random: None | int | instance of RandomState
      RNG to perform splitting. Passed to `sklearn.utils.check_random_state`.

    :var growing: bool
      If True (the default), let `self.X` and `self.y` return the growing set,
      otherwise the pruning set.
    """
    def __init__(self, *,
                 pruning_split_ratio: float = 0.25,
                 grow_prune_random=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pruning_split_ratio = pruning_split_ratio
        self.growing = True
        self._grow_prune_random = check_random_state(grow_prune_random)

    def set_context(self, estimator: '_BinarySeCoEstimator', X, y):
        super().set_context(estimator, X, y)

        n_samples = len(y)
        shuffled = self._grow_prune_random.permutation(n_samples)
        split_point = int(n_samples * self.pruning_split_ratio)
        # FIXME: stratify, i.e. preserve imbalanced class distribution
        self._growing_X = X[shuffled][:split_point]
        self._growing_y = y[shuffled][:split_point]
        self._pruning_X = X[shuffled][split_point:]
        self._pruning_y = y[shuffled][split_point:]
        self.growing = True

    @property
    def X(self):
        return self._growing_X if self.growing else self._pruning_X

    @property
    def y(self):
        return self._growing_y if self.growing else self._pruning_y


class RipperPostPruning(GrowPruneSplit):
    @abstractmethod
    def pruning_evaluation(self, rule: AugmentedRule) -> Tuple[float, ...]:
        """Rate rule to allow finding the best refinement, while pruning."""

    def simplify_rule(self, rule: AugmentedRule) -> AugmentedRule:
        """TODO: doc

        NOTE: Overrides `AugmentedRule._sort_key` (i.e. the evaluation of
          the rule under the growing heuristic) with the pruning heuristic
          during the runtime of this method.

        :return: An improved version of `rule`.
        """
        self.growing = False  # tell GrowPruneSplit to use pruning set

        candidates = [rule]
        # dropping all final (i.e. last added) sets of conditions
        generalization = rule
        for boundary, index, value, old_value in reversed(rule.condition_trace):
            generalization = generalization.copy()
            generalization.set_condition(boundary, index, old_value)
            generalization._sort_key = self.pruning_evaluation(generalization)
            candidates.append(generalization)
        candidates.sort()

        best_rule = candidates.pop()
        self.rate_rule(best_rule)  # restore rating by grow-heuristic
        return best_rule


# Example Algorithm configurations


class SimpleSeCoImplementation(BeamSearch,
                               TopDownSearch,
                               PurityHeuristic,
                               NoPostPruning,
                               NoPostProcess):

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        p, n = self.count_matches(rule)
        return False

    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule
                                ) -> bool:
        p, n = self.count_matches(rule)
        return n >= p


class SimpleSeCoEstimator(SeCoEstimator):
    def __init__(self, multi_class="one_vs_rest", n_jobs=1):
        super().__init__(SimpleSeCoImplementation(), multi_class, n_jobs)


class CN2Implementation(BeamSearch,
                        TopDownSearch,
                        LaplaceHeuristic,
                        SignificanceStoppingCriterion,
                        NoPostPruning,
                        NoPostProcess):
    """CN2 as refined by (Clark and Boswell 1991)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule
                                ) -> bool:
        """Abort search if rule covers no positive examples.

        This corresponds to the "E is empty" condition in Table 3 of
        (Clark and Niblett 1989).
        """
        p, n = self.count_matches(rule)
        return p == 0


class CN2Estimator(SeCoEstimator):
    """Estimator using :class:`CN2Implementation`."""
    def __init__(self,
                 LRS_threshold: float = 0.0,
                 multi_class="one_vs_rest",
                 n_jobs=1):
        super().__init__(CN2Implementation(LRS_threshold=LRS_threshold),
                         multi_class, n_jobs)
        # sklearn assumes all parameters are class fields, so copy this here
        self.LRS_threshold = LRS_threshold


class RipperImplementation(BeamSearch,
                           TopDownSearch,
                           InformationGainHeuristic,
                           RipperPostPruning,  # already pulls GrowPruneSplit
                           NoPostProcess
                           ):
    """Ripper as defined by (Cohen 1995).

    NOTE: The global post-optimization phase is currently not implemented
        (that would be the `post_process` method).

    fields
    =====
    - `best_description_length`: The minimal DL found in the current search so
      far, `minDL` in weka/JRip.
    - `description_length_`: The DL of the current theory, `dl` in weka/JRip.
    """

    def __init__(self,
                 check_error_rate: bool = True,
                 description_length_surplus: int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.check_error_rate = check_error_rate
        self.description_length_surplus = description_length_surplus
        # init fields
        self.description_length_ = None
        self.best_description_length = None
        self.expected_fp_over_err = None
        self.theory_pn = None

    def pruning_evaluation(self, rule: AugmentedRule
                           ) -> Tuple[float, float, int]:
        """Laplace heuristic, as defined by (Clark and Boswell 1991).

        JRip documentation states:
        "The pruning metric is (p-n)/(p+n) -- but it's actually 2p/(p+n) -1, so
        in this implementation we simply use p/(p+n) (actually (p+1)/(p+n+2),
        thus if p+n is 0, it's 0.5)."
        """
        p, n = self.count_matches(rule)
        laplace = (p + 1) / (p + n + 2)
        # tie-breaking by positive coverage p and rule discovery order
        return (laplace, p, -rule.instance_no)

    def set_context(self, estimator: _BinarySeCoEstimator, X, y):
        super().set_context(estimator, X, y)

        if self.description_length_ is None:
            positives = np.count_nonzero(y == self.target_class)
            # set expected fp/(err = fp+fn) rate := proportion of the class
            self.expected_fp_over_err = positives / len(y)
            # set default DL (only data, empty theory)
            self.description_length_ = \
                self.best_description_length = \
                self.data_description_length(covered=0, uncovered=len(y),
                                             fp=0, fn=positives)
        self.theory_pn = []

    def unset_context(self):
        super().unset_context()
        self.description_length_ = None
        self.best_description_length = None
        self.expected_fp_over_err = None
        self.theory_pn = None

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        """TODO: accuRate from JRip"""
        p, n = self.count_matches(rule)
        accuracy_rate = (p + 1) / (p + n + 1)
        return accuracy_rate >= 1

    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule
                                ) -> bool:
        """TODO: MDL-based, taken from JRip"""
        self.theory_pn.append((self.P, self.N))

        self.description_length_ += \
            self.relative_description_length(theory, rule)
        self.best_description_length = min(self.best_description_length,
                                           self.description_length_)
        if self.description_length_ > (self.best_description_length +
                                       self.description_length_surplus):
            return True
        p, n = self.count_matches(rule)
        if p <= 0:
            return True
        if self.check_error_rate and (n / (p + n)) >= 0.5:  # error rate
            return True
        return False

    def relative_description_length(self, theory: Theory, rule: AugmentedRule):
        """TODO: from JRip

        note JRip has the index parameter which is only !=last_rule in global optimization
        """
        minDataDLIfExists = self.minDataDLIfExists(theory, rule)
        rule_DL = self.rule_description_length(rule)
        minDataDLIfDeleted = self.minDataDLIfDeleted(theory, rule)
        return minDataDLIfExists + rule_DL - minDataDLIfDeleted

    @staticmethod
    def subset_description_length(n, k, p):
        return -k * log2(p) - (n - k) * log2(1 - p)

    def rule_description_length(self, rule: AugmentedRule):
        n_conditions = np.count_nonzero(rule.conditions)  # no. of conditions
        max_n_conditions = rule.conditions.size  # possible no. of conditions
        # TODO: JRip counts all thresholds (RuleStats.numAllConditions())

        kbits = log2(n_conditions)  # no. of bits to send `n_conditions`
        if n_conditions > 1:
            kbits += 2 * log2(kbits)
        rule_dl = kbits + self.subset_description_length(
            max_n_conditions, n_conditions, n_conditions / max_n_conditions)
        return rule_dl * 0.5  # redundancy factor

    def data_description_length(self, covered, uncovered, fp, fn):
        """XXX"""
        S = self.subset_description_length
        total_bits = log2(covered + uncovered + 1)
        if covered > uncovered:
            assert covered > 0
            expected_error = self.expected_fp_over_err * (fp + fn)
            covered_bits = S(covered, fp, expected_error / covered)
            uncovered_bits = S(uncovered, fn, fn / uncovered) \
                if uncovered > 0 else 0
        else:
            assert uncovered > 0
            expected_error = (1 - self.expected_fp_over_err) * (fp + fn)
            covered_bits = S(covered, fp, fp / covered) \
                if covered > 0 else 0
            uncovered_bits = S(uncovered, fn, expected_error / uncovered)
        return total_bits + covered_bits + uncovered_bits

    def minDataDLIfExists(self, theory: Theory, rule: AugmentedRule):
        p, n = self.count_matches(rule)
        P, N = self.P, self.N
        return self.data_description_length(
            covered=sum(th_p + th_n for th_p, th_n in self.theory_pn),  # of theory
            uncovered=P + N - p - n,  # of rule
            fp=sum(th_n for th_p, th_n in self.theory_pn),  # of theory
            fn=N - n,  # of rule
        )

    def minDataDLIfDeleted(self, theory: Theory, rule: AugmentedRule):
        p, n = self.count_matches(rule)
        P, N = self.P, self.N
        # covered stats cumulate over theory
        coverage = sum(th_p + th_n for th_p, th_n in self.theory_pn)
        fp = sum(th_n for th_p, th_n in self.theory_pn)
        # uncovered stats are those of the last rule
        if theory:
            uncoverage = P + N - p - n
            fn = N - n
        else:
            # we're at the first rule
            uncoverage = P + N  # == coverage + uncoverage
            fn = p + N - n  # tp + fn
        return self.data_description_length(
            covered=coverage, uncovered=uncoverage, fp=fp, fn=fn)


class RipperEstimator(SeCoEstimator):
    def __init__(self,
                 multi_class="one_vs_rest",
                 n_jobs=1):
        super().__init__(RipperImplementation(), multi_class, n_jobs)


class IrepImplementation(BeamSearch,
                         TopDownSearch,
                         InformationGainHeuristic,
                         RipperPostPruning,  # already pulls GrowPruneSplit
                         NoPostProcess):
    """IREP as defined by (Cohen 1995), originally by (Fürnkranz, Widmer 1994).
    """

    def pruning_evaluation(self, rule: AugmentedRule) -> Tuple[float, ...]:
        p, n = self.count_matches(rule)
        P, N = self.P, self.N
        v = (p + N - n) / (P + N)
        return (v, p, -rule.instance_no)

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        """[refine] until the rule covers no negative examples"""
        p, n = self.count_matches(rule)
        return n == 0

    def rule_stopping_criterion(self, theory: Theory,
                                rule: AugmentedRule) -> bool:
        p, n = self.count_matches(rule)
        return p / (p + n) < 0.5


class IrepEstimator(SeCoEstimator):
    def __init__(self, multi_class="one_vs_rest", n_jobs=1):
        super().__init__(IrepImplementation(), multi_class, n_jobs)


# TODO: don't require definition of 2 classes, add *Estimator factory method in SeCoBaseImplementation
# TODO: dedup stopping criteria
