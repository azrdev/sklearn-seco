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
from typing import Tuple, Iterable

import numpy as np
from scipy.special import xlogy
from sklearn.utils import check_random_state

from sklearn_seco.abstract import Theory, SeCoEstimator, _BinarySeCoEstimator
from sklearn_seco.common import \
    RuleQueue, SeCoBaseImplementation, AugmentedRule, LOWER, UPPER, log2
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

    grow = rng.permutation(grow)
    prune = rng.permutation(prune)
    return grow, prune


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
        return AugmentedRule(n_features=self.n_features,
                             **self.rule_prototype_arguments)

    def refine_rule(self, rule: AugmentedRule) -> Iterable[AugmentedRule]:
        """Create refinements of `rule` by adding a test for each of the
        (unused) attributes, one at a time, trying all possible attribute
        values / thresholds.
        """
        all_feature_values = self.all_feature_values
        categorical_mask = self.categorical_mask
        # TODO: mark constant features for exclusion in future specializations

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


class PurityHeuristic(SeCoBaseImplementation):
    """Mixin providing the purity as rule evaluation metric: the percentage of
    positive examples among the examples covered by the rule.
    """

    def growing_heuristic(self, rule: AugmentedRule) -> float:
        p, n = self.count_matches(rule)
        if p + n == 0:
            return 0
        return p / (p + n)  # purity


class LaplaceHeuristic(SeCoBaseImplementation):
    """Mixin implementing the Laplace rule evaluation metric.

    The Laplace estimate was defined by (Clark and Boswell 1991) for CN2.
    """
    def growing_heuristic(self, rule: AugmentedRule) -> float:
        """Laplace heuristic, as defined by (Clark and Boswell 1991)."""
        p, n = self.count_matches(rule)
        return (p + 1) / (p + n + 2)  # laplace


class InformationGainHeuristic(SeCoBaseImplementation):
    def growing_heuristic(self, rule: AugmentedRule) -> float:
        p, n = self.count_matches(rule)
        P, N = self.P, self.N  # TODO: maybe count_matches(rule.original) is meant here? book fig6.4 says code is correct
        if p == 0:
            return 0
        # return p * (log2(p / (p + n)) - log2(P / (P + N)))  # info_gain
        return p * (log2(p) - log2(p + n) - log2(P) + log2(P + N))  # info_gain


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

    The split is stratified (i.e. class proportions are retained).
    Note that JRip (and presumably the original RIPPER) state their
    stratification is bugged.

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
        self._grow_prune_random = check_random_state(grow_prune_random)
        self.__growing = True

    def set_context(self, estimator: '_BinarySeCoEstimator', X, y):
        super().set_context(estimator, X, y)
        assert 0 <= self.pruning_split_ratio <= 1

        grow_idx, prune_idx = grow_prune_split(y,
                                               self.target_class,
                                               self.pruning_split_ratio,
                                               self._grow_prune_random,)
        self._growing_X = X[grow_idx]
        self._growing_y = y[grow_idx]
        self._pruning_X = X[prune_idx]
        self._pruning_y = y[prune_idx]

        self.growing = True

    def evaluate_rule(self, rule: AugmentedRule) -> None:
        """Copy `SeCoBaseImplementation.evaluate_rule` but use
        `pruning_heuristic` while pruning.
        """
        if self.growing:
            super().evaluate_rule(rule)
        else:
            p, n = self.count_matches(rule)
            rule._sort_key = (self.pruning_heuristic(rule),
                              p,
                              -rule.instance_no)

    @abstractmethod
    def pruning_heuristic(self, rule: AugmentedRule) -> float:
        """Rate rule to allow finding the best refinement, while pruning."""

    @property
    def growing(self):
        return self.__growing

    @growing.setter
    def growing(self, value):
        if self.__growing != value:
            self._P = None
            self._N = None
            # TODO: ensure rule._sort_key is not mixed. use two incomparable tuple-types?
        self.__growing = value

    @property
    def X(self):
        return self._growing_X if self.growing else self._pruning_X

    @property
    def y(self):
        return self._growing_y if self.growing else self._pruning_y


class RipperPostPruning(GrowPruneSplit):
    """Post-Pruning as performed by RIPPER, directly after growing
    (`find_best_rule`) each rule.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rule_prototype_arguments["enable_condition_trace"] = True

    def simplify_rule(self, rule: AugmentedRule) -> AugmentedRule:
        """Find the best simplification of `rule` by dropping conditions and
        evaluating using `pruning_heuristic`.

        NOTE: Overrides `AugmentedRule._sort_key` (i.e. the evaluation of
          the rule under the growing heuristic) with the pruning heuristic
          during the runtime of this method.

        :return: An improved version of `rule`.
        """
        evaluate_rule = self.evaluate_rule

        self.growing = False  # tell GrowPruneSplit to use pruning set
        evaluate_rule(rule)
        candidates = [rule]
        # drop all final (i.e. last added) sets of conditions
        generalization = rule
        for boundary, index, value, old_value in reversed(rule.condition_trace):
            generalization = generalization.copy()
            generalization.set_condition(boundary, index, old_value)
            evaluate_rule(generalization)
            candidates.append(generalization)

        candidates.sort()
        best_rule = candidates.pop()

        # restore rating by grow-heuristic
        self.growing = True
        evaluate_rule(best_rule)
        return best_rule


# Example Algorithm configurations


class SimpleSeCoImplementation(BeamSearch,
                               TopDownSearch,
                               PurityHeuristic,
                               NoPostPruning,
                               NoPostProcess):

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        # p, n = self.count_matches(rule)
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


class RipperMdlStop(SeCoBaseImplementation):
    """MDL (minimum description length) stopping criterion used by RIPPER.

    Abort search if the last found rule has a `description_length_` higher than
    the best so far + some margin (`description_length_surplus`), or `p=0` or
    (if `check_error_rate is True`) `n >= p`.

    NOTE: Reconstructed mainly from JRip.java source, no guarantee on all
      details being corect an identical to other implementation(s).
    """

    def __init__(self, *,
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

    def set_context(self, estimator: _BinarySeCoEstimator, X, y):
        super().set_context(estimator, X, y)

        if self.description_length_ is None:
            # initialize at start of abstract_seco
            positives = np.count_nonzero(y == self.target_class)
            # set expected fp/(err = fp+fn) rate := proportion of the class
            self.expected_fp_over_err = positives / len(y)
            # set default DL (only data, empty theory)
            self.description_length_ = self.best_description_length = \
                data_description_length(
                    expected_fp_over_err=self.expected_fp_over_err,
                    covered=0, uncovered=len(y), fp=0, fn=positives)
        self.theory_pn = []

    def unset_context(self):
        super().unset_context()
        self.description_length_ = None
        self.best_description_length = None
        self.expected_fp_over_err = None
        self.theory_pn = None

    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule
                                ) -> bool:
        p, n = self.count_matches(rule)
        P, N = self.P, self.N
        self.theory_pn.append((P, N))

        self.description_length_ += relative_description_length(
            rule, self.expected_fp_over_err, p, n, P, N, self.theory_pn)
        self.best_description_length = min(self.best_description_length,
                                           self.description_length_)
        if self.description_length_ > (self.best_description_length +
                                       self.description_length_surplus):
            return True
        if p <= 0:
            return True
        if self.check_error_rate and (n / (p + n)) >= 0.5:  # error rate  # XXX: eq. n >= p
            return True
        return False


class RipperImplementation(BeamSearch,
                           TopDownSearch,
                           InformationGainHeuristic,
                           RipperMdlStop,
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

    def pruning_heuristic(self, rule: AugmentedRule
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

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        """Laplace-based criterion. Field `accuRate` in JRip.java."""
        p, n = self.count_matches(rule)
        accuracy_rate = (p + 1) / (p + n + 1)
        return accuracy_rate >= 1


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

    def pruning_heuristic(self, rule: AugmentedRule) -> Tuple[float, ...]:
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
        return p / (p + n) < 0.5  # TODO: eq. p < n


class IrepEstimator(SeCoEstimator):
    def __init__(self, multi_class="one_vs_rest", n_jobs=1):
        super().__init__(IrepImplementation(), multi_class, n_jobs)


# TODO: don't require definition of 2 classes, add *Estimator factory method in SeCoBaseImplementation
# TODO: dedup stopping criteria
