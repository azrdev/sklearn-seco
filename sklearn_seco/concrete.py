"""
Implementation of SeCo / Covering algorithm:
Usual building blocks & known instantiations of the abstract base algorithm.
"""

from typing import Tuple, Iterable
import numpy as np
from sklearn_seco.abstract import RuleQueue, Theory, SeCoEstimator
from sklearn_seco.common import SeCoBaseImplementation, AugmentedRule


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
        p, n = self.count_matches(rule)
        return n == 0

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
    """CN2 as refined by (Clark and Boswell 1991)."""
    def __init__(self, LRS_threshold: float):
        super().__init__()
        self.LRS_threshold = LRS_threshold

    def evaluate_rule(self, rule: AugmentedRule) -> Tuple[float, float, int]:
        """Laplace heuristic, as defined by (Clark and Boswell 1991)."""
        p, n = self.count_matches(rule)
        laplace = (p + 1) / (p + n + 2)
        return (laplace, p, rule.instance_no)  # tie-breaking by pos. coverage

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        """*Significance test* as defined by (Clark and Niblett 1989), but used
        there for rule evaluation, instead used as stopping criterion following
        (Clark and Boswell 1991).
        """
        p, n = self.count_matches(rule)
        P, N = self.P, self.N
        if 0 in (p, n, P, N):
            return True
        purity = p / (p + n)
        impurity = n / (p + n)
        CE = (- purity * np.log(purity / (P / (P + N)))  # cross entropy
              - impurity * np.log(impurity / (N / P + N)))
        J = p * CE  # J-Measure
        LRS = 2 * (P + N) * J  # likelihood ratio statistics
        return LRS <= self.LRS_threshold

    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule) -> bool:
        """abort search if rule covers no examples"""
        p, n = self.count_matches(rule)
        return p == 0


class CN2Estimator(SeCoEstimator):
    """Estimator using :class:`CN2Implementation`."""
    def __init__(self,
                 LRS_threshold: float = 0.9,
                 multi_class="one_vs_rest",
                 n_jobs=1):
        super().__init__(CN2Implementation(LRS_threshold), multi_class, n_jobs)
        # sklearn assumes all parameters are class fields, so copy this here
        self.LRS_threshold = LRS_threshold  # FIXME: set_param not reflected in self.implementation


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    # copied from itertools docs
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
