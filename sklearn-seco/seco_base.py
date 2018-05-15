"""Implementation of SeCo / Covering algorithm"""
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Iterable, FrozenSet, List, NamedTuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import unique_labels, check_classification_targets

Condition = NamedTuple('Condition', [
    ('attribute_index', int),
    ('threshold', float)])
Rule = NamedTuple('Rule', [
    ('conditions', FrozenSet[Condition]),
    ('classification', int)])
Theory = List[Rule]

RatedRule = NamedTuple('RatedRule', [
    ('rating', float),
    ('rule', Rule)])
RuleQueue = List[RatedRule]


def match_rule(rule: Rule, sample) -> bool:
    def check_condition(attribute_index, value, sample) -> bool:
        # FIXME: numeric attributes (use <= instead of ==)
        return sample[attribute_index] == value

    return all(check_condition(c.attribute_index, c.threshold, sample)
               for c in rule.conditions)


class SeCoEstimator(BaseEstimator, ClassifierMixin, ABC):

    # abstract interface

    @abstractmethod
    def init_rule(self, examples) -> Rule:
        """Create a new rule to be refined before added to the theory."""
        pass

    # TODO: make metric(s) of previous rule available
    # TODO: API for tie-breaking rules
    @abstractmethod
    def evaluate_rule(self, rule: Rule, examples) -> float:
        """Rate rule to allow comparison & finding the best refinement
        (using operator `>`).
        """
        pass

    @abstractmethod
    def select_candidate_rules(self, rules: RuleQueue,
                               examples) -> Iterable[Rule or RatedRule]:
        """Remove and return those Rules from `rules` which should be refined.
        """
        pass

    @abstractmethod
    def refine_rule(self, rule: Rule, examples) -> Iterable[Rule]:
        """Create all refinements from `rule`."""
        pass

    @abstractmethod
    def inner_stopping_criterion(self, rule: Rule, examples) -> bool:
        """return `True` to stop refining `rule`."""
        pass

    @abstractmethod
    def filter_rules(self, rules: RuleQueue, examples) -> RuleQueue:
        """After one refinement iteration, filter the candidate `rules` (may be
        empty) for the next one.
        """
        pass

    @abstractmethod
    def rule_stopping_criterion(self, theory: Theory, rule: Rule,
                                examples) -> bool:
        """return `True` to stop finding more rules, given `rule` was the
        best Rule found.
        """
        pass

    @abstractmethod
    def post_process(self, theory: Theory) -> Theory:
        """Modify `theory` after it has been learned."""
        pass

    # base implementation

    def fit(self, X, y, target_class=None):
        """Build the decision rule list from training data `X` with labels `y`.
        """

        # FIXME: nominal features not possible in sklearn, only numerical. use LabelEncoder
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = unique_labels(y)

        # TODO: class binarization (sklearn.multiclass.OneVsOne / OneVsRest)
        if not target_class or target_class not in self.classes_:
            self.target_class_ = self.classes_[0]

        # TODO: optimize
        # build index of features and their possible values
        self.all_features = defaultdict(lambda: set())  # { feature index -> Set[values] }
        for instance in X:
            for i, value in enumerate(instance):
                self.all_features[i].add(value)

        # run SeCo algorithm
        self.theory_ = self._abstract_seco(
            examples=np.concatenate((X, y[:, np.newaxis]), axis=1),  # [X|y]
            target_class=self.target_class_,
        )
        self.all_features = None
        return self

    def _find_best_rule(self, examples) -> Rule:
        """Inner loop of abstract SeCo/Covering algorithm.

        :param examples: not yet covered examples with classification: [X|y]
        """

        # self. methods, as variables for performance
        init_rule = self.init_rule
        evaluate_rule = self.evaluate_rule
        select_candidate_rules = self.select_candidate_rules
        refine_rule = self.refine_rule
        inner_stopping_criterion = self.inner_stopping_criterion
        filter_rules = self.filter_rules

        # algorithm
        best_rule = init_rule(examples)
        best_rule = RatedRule(evaluate_rule(best_rule, examples), best_rule)
        rules: RuleQueue = [best_rule]
        while len(rules):
            for candidate in select_candidate_rules(rules, examples):
                for refinement in refine_rule(candidate, examples):
                    new_rule = RatedRule(evaluate_rule(refinement, examples),
                                         refinement)
                    if not inner_stopping_criterion(refinement, examples):
                        rules.append(new_rule)
                        rules.sort()
                        if new_rule[0] > best_rule[0]:
                            best_rule = new_rule
            rules = filter_rules(rules, examples)
        return best_rule[1]

    def _abstract_seco(self, examples: np.ndarray, target_class: int) -> Theory:
        """Main loop of abstract SeCo/Covering algorithm.

        :param examples: np.ndarray with features and classification: [X|y]
        :param target_class: "positive" class, all others are considered
         negative
        :return: Theory
        """

        # self. methods, as variables for performance
        rule_stopping_criterion = self.rule_stopping_criterion
        find_best_rule = self._find_best_rule
        post_process = self.post_process

        # TODO: split growing/pruning set for ripper
        # main loop
        theory = list()
        while np.any(examples[:, -1] == target_class):
            rule = find_best_rule(examples)
            if rule_stopping_criterion(theory, rule, examples):
                break
            # ignore the rest of theory, because it already covered
            covered = np.apply_along_axis(lambda row: match_rule(rule, row),
                                          1, examples)  # array of bool
            examples = examples[~covered]  # TODO: use mask array instead of copy?
            theory.append(rule)
        return post_process(theory)

    def predict(self, X):
        X = check_array(X)
        n_samples = int(X.shape[0])
        ret = np.zeros(n_samples)

        # TODO: use numpy instead of python for performance
        for sample_index, sample in enumerate(X):
            for rule in self.theory_:  # TODO: ordered list / unordered set/tree
                if match_rule(rule, sample):
                    ret[sample_index] = rule.classification  # FIXME: class index or name?
                    break
            # TODO: default rule
        return ret

    # TODO: cache result, esp. P,N are the same for all refinements
    # TODO: only calculate values when needed
    def count_matches(self, rule: Rule, examples) -> Tuple[int, int, int, int]:
        """Return (p, n, P, N) where:

        returns
        -------
        p : int
            The count of positive examples covered by `rule`
        n : int
            The count of negative examples covered by `rule`
        P : int
            The count of positive examples
        N : int
            The count of negative examples
        """
        target_class = self.target_class_
        y = examples[:, -1]

        def matcher(row):
            return match_rule(rule, row)
        # the following three are np.arrays of bool
        covered = np.apply_along_axis(matcher, 1, examples)
        positives = y == target_class
        negatives = ~positives
        # NOTE: nonzero() is test for True
        p = np.count_nonzero(covered & positives)
        n = np.count_nonzero(covered & negatives)
        P = np.count_nonzero(positives)
        N = np.count_nonzero(negatives)
        assert P+N == len(y) == examples.shape[0]
        assert p+n == np.count_nonzero(np.apply_along_axis(matcher, 1, examples))
        return (p, n, P, N)


class SimpleSeCo(SeCoEstimator):
    def init_rule(self, examples) -> Rule:
        return Rule(frozenset(), self.target_class_)  # TODO: target_class_ only known after fit()

    def evaluate_rule(self, rule: Rule, examples) -> float:
        p, n, P, N = self.count_matches(rule, examples)
        if p+n == 0:
            return 0
        purity = p / (p+n)
        return purity

    def select_candidate_rules(self, rules: RuleQueue,
                               examples) -> Iterable[Rule]:
        last = rules.pop()
        return [last[1]]

    def refine_rule(self, rule: Rule, examples) -> Iterable[Rule]:
        used_features = set((cond[0] for cond in rule.conditions))
        for feature in self.all_features.keys() ^ used_features:
            for value in self.all_features[feature]:
                # TODO: simplify rule copy/construction
                specialization = Condition(feature, value)
                new_conditions = rule.conditions.union([specialization])
                yield Rule(new_conditions, rule.classification)

    def inner_stopping_criterion(self, rule: Rule, examples) -> bool:
        return False  # TODO: java NoNegativesCoveredStop

    def filter_rules(self, rules: RuleQueue, examples) -> RuleQueue:
        return rules[-1:]  # only the best one

    def rule_stopping_criterion(self, theory: Theory, rule: Rule,
                                examples) -> bool:
        return False  # TODO: java CoverageRuleStop

    def post_process(self, theory: Theory) -> Theory:
        return theory


# TODO: CN2 incomplete
class CN2(SeCoEstimator):
    def __init__(self, LRS_threshold):
        self.LRS_threshold = LRS_threshold

    def init_rule(self, examples) -> Rule:
        return Rule(frozenset(), self.target_class_)  # TODO: target_class_ only known after fit()

    def evaluate_rule(self, rule: Rule, examples) -> float:
        # laplace heuristic
        p, n, P, N = self.count_matches(rule, examples)
        LPA = (p + 1) / (p + n + 2)
        return LPA

    def select_candidate_rules(self, rules: RuleQueue,
                               examples) -> Iterable[Rule]:
        last = rules.pop()
        return [last.rule]

    def refine_rule(self, rule: Rule, examples) -> Iterable[Rule]:
        conditions = rule[0]
        classification = rule[1]
        used_features = set(cond[0] for cond in conditions)
        for feature in self.all_features.keys() ^ used_features:
            for value in self.all_features[feature]:
                # TODO: simplify rule copy/construction
                new_conditions = conditions.union([(feature, value)])
                yield Rule(new_conditions, classification)

    def inner_stopping_criterion(self, rule: Rule, examples) -> bool:
        # TODO: return LRS(rule) <= self.LRS_threshold
        return False

    def filter_rules(self, rules: RuleQueue, examples) -> RuleQueue:
        return rules[-1:]  # only the best one

    def rule_stopping_criterion(self, theory: Theory, rule: Rule,
                                examples) -> bool:
        # return True iff rule covers no examples
        p, n, P, N = self.count_matches(rule, examples)
        return p == 0

    def post_process(self, theory: Theory) -> Theory:
        return theory


if __name__ == "__main__":
    check_estimator(SimpleSeCo)
    check_estimator(CN2)
