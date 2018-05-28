"""Implementation of SeCo / Covering algorithm"""
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple, Iterable, FrozenSet, List, NamedTuple,  Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.utils.validation import check_is_fitted

Condition = NamedTuple('Condition', [
    ('attribute_index', int),
    ('threshold', float)])  # TODO: use dict[int, threshold] instead?
Rule = FrozenSet[Condition]
Theory = List[Rule]

RatedRule = NamedTuple('RatedRule', [
    ('rating', float),
    ('rule', Rule)])
RuleQueue = List[RatedRule]


def match_rule(rule: Rule, sample) -> bool:
    def check_condition(attribute_index, value) -> bool:
        # FIXME: numeric attributes (use <= instead of ==)
        return sample[attribute_index] == value

    return all(sample[c.attribute_index] == c.threshold
               #check_condition(c.attribute_index, c.threshold)
               for c in rule)


# TODO: cache result, esp. P,N are the same for all refinements
# TODO: only calculate values when needed
def count_matches(rule: Rule, target_class, examples) -> Tuple[int, int, int, int]:
    """Return (p, n, P, N) where:

    returns
    -------
    p : int
        The count of positive examples (== target_class) covered by `rule`
    n : int
        The count of negative examples (!= target_class) covered by `rule`
    P : int
        The count of positive examples
    N : int
        The count of negative examples
    """
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


class SeCoImplementation(ABC):
    """The callbacks needed by _BinarySeCoEstimator"""

    @abstractmethod
    def set_context(self, classes_: np.ndarray,
                    all_feature_values: Callable[[int], Iterable[Any]]):
        """New invocation of `_BinarySeCoEstimator._find_best_rule`.

        :param all_feature_values: A callable returning for each feature index
         (in examples) a generator of all values of that feature.
        """
        pass

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


class _BinarySeCoEstimator(BaseEstimator):
    def __init__(self, implementation: SeCoImplementation):
        super().__init__()
        self.implementation = implementation

    def fit(self, X, y):
        """Build the decision rule list from training data `X` with labels `y`.
        """

        # FIXME: nominal features not possible in sklearn, only numerical. use LabelEncoder
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        # TODO: assuming classes_[0, 1] being positive & negative.
        # TODO: a default rule might lead to asymmetry → positive class should be configurable

        # run SeCo algorithm
        self.theory_ = self._abstract_seco(
            examples=np.concatenate((X, y[:, np.newaxis]), axis=1),  # [X|y]
        )
        return self

    def _find_best_rule(self, examples) -> Rule:
        """Inner loop of abstract SeCo/Covering algorithm.

        :param examples: not yet covered examples with classification: [X|y]
        """

        # resolve methods once for performance
        init_rule = self.implementation.init_rule
        evaluate_rule = self.implementation.evaluate_rule
        select_candidate_rules = self.implementation.select_candidate_rules
        refine_rule = self.implementation.refine_rule
        inner_stopping_criterion = self.implementation.inner_stopping_criterion
        filter_rules = self.implementation.filter_rules

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

    def _abstract_seco(self, examples: np.ndarray) -> Theory:
        """Main loop of abstract SeCo/Covering algorithm.

        :param examples: np.ndarray with features and classification: [X|y]
        :return: Theory
        """

        # resolve methods once for performance
        rule_stopping_criterion = self.implementation.rule_stopping_criterion
        find_best_rule = self._find_best_rule
        post_process = self.implementation.post_process

        # TODO: split growing/pruning set for ripper
        # main loop
        theory = list()
        while np.any(examples[:, -1] == self.classes_[0]):

            # TODO: optimize
            @lru_cache(maxsize=None)
            def all_values(feature_index: int):
                """:return: all possible values of feature with given index"""
                return np.unique(examples[:, feature_index])
            # depends on examples, which change each iteration
            all_values.cache_clear()

            self.implementation.set_context(self.classes_,
                                            all_values)
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
        pos = self.classes_[0]
        neg = self.classes_[1]

        # TODO: optimize?
        def predict_sample(sample):
            for rule in self.theory_:  # TODO: ordered list / unordered set/tree
                if match_rule(rule, sample):
                    return pos
            return neg  # TODO: default rule
        return np.apply_along_axis(predict_sample, 1, X)


    def predict_proba(self, X):
        X = check_array(X)

        # TODO: optimize?
        def predict_sample(sample):
            for rule in self.theory_:  # TODO: ordered list / unordered set/tree
                if match_rule(rule, sample):
                    return [1, 0]  # XXX: check/fix order of classes/position of target
            return [0, 1]  # TODO: default rule
        return np.apply_along_axis(predict_sample, 1, X)


class SeCoEstimator(BaseEstimator, ClassifierMixin, ABC):
    """Wrap the base SeCo implementation to provide class label binarization."""
    def __init__(self, implementation: SeCoImplementation,
                 multi_class="one_vs_rest", n_jobs=1):
        self.implementation = implementation
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False)

        self.base_estimator_ = _BinarySeCoEstimator(self.implementation)

        # copied from GaussianProcessClassifier
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError("SeCoEstimator requires 2 or more distinct "
                             "classes. Only class %s present."
                             % self.classes_[0])
        elif self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(self.base_estimator_,
                                                           n_jobs=self.n_jobs)
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = OneVsOneClassifier(self.base_estimator_,
                                                          n_jobs=self.n_jobs)
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class)

        self.base_estimator_.fit(X, y)
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        check_is_fitted(self, ["classes_", "n_classes_"])
        X = check_array(X)
        return self.base_estimator_.predict(X)


class SimpleSeCoImplementation(SeCoImplementation):
    def set_context(self, classes_: np.ndarray, all_feature_values):
        self.target_class = classes_[0]  # TODO: order implicit?
        self.all_feature_values = all_feature_values

    def init_rule(self, examples) -> Rule:
        return frozenset()

    def evaluate_rule(self, rule: Rule, examples) -> float:
        p, n, P, N = count_matches(rule, self.target_class, examples)
        if p+n == 0:
            return 0
        purity = p / (p+n)
        return purity

    def select_candidate_rules(self, rules: RuleQueue,
                               examples) -> Iterable[Rule]:
        last = rules.pop()
        return [last.rule]

    def refine_rule(self, rule: Rule, examples) -> Iterable[Rule]:
        used_features = frozenset((cond.attribute_index for cond in rule))
        all_features = range(examples.shape[1] - 1)
        for feature in used_features.symmetric_difference(all_features):
            for value in self.all_feature_values(feature):
                # TODO: simplify rule copy/construction
                specialization = Condition(feature, value)
                yield rule.union([specialization])

    def inner_stopping_criterion(self, rule: Rule, examples) -> bool:
        return False  # TODO: java NoNegativesCoveredStop

    def filter_rules(self, rules: RuleQueue, examples) -> RuleQueue:
        return rules[-1:]  # only the best one

    def rule_stopping_criterion(self, theory: Theory, rule: Rule,
                                examples) -> bool:
        return False  # TODO: java CoverageRuleStop

    def post_process(self, theory: Theory) -> Theory:
        return theory


class SimpleSeCoEstimator(SeCoEstimator):
    def __init__(self, multi_class="one_vs_rest", n_jobs=1):
        super().__init__(SimpleSeCoImplementation(), multi_class, n_jobs)


# TODO: CN2 incomplete
# TODO: dedup CN2, SimpleSeCo
class CN2Implementation(SeCoImplementation):
    def __init__(self, LRS_threshold: float):
        self.LRS_threshold = LRS_threshold

    def set_context(self, classes_: np.ndarray, all_feature_values):
        self.target_class = classes_[0]  # TODO: order implicit?
        self.all_feature_values = all_feature_values

    def init_rule(self, examples) -> Rule:
        return frozenset()

    def evaluate_rule(self, rule: Rule, examples) -> float:
        # laplace heuristic
        p, n, P, N = count_matches(rule, self.target_class, examples)
        LPA = (p + 1) / (p + n + 2)
        return LPA

    def select_candidate_rules(self, rules: RuleQueue,
                               examples) -> Iterable[Rule]:
        last = rules.pop()
        return [last.rule]

    def refine_rule(self, rule: Rule, examples) -> Iterable[Rule]:
        used_features = frozenset(cond.attribute_index for cond in rule)
        all_features = range(examples.shape[1] - 1)
        for feature in used_features.symmetric_difference(all_features):
            for value in self.all_feature_values(feature):
                # TODO: simplify rule copy/construction
                specialization = Condition(feature, value)
                yield rule.union([specialization])

    def inner_stopping_criterion(self, rule: Rule, examples) -> bool:
        # TODO: return LRS(rule) <= self.LRS_threshold
        return False

    def filter_rules(self, rules: RuleQueue, examples) -> RuleQueue:
        return rules[-1:]  # only the best one

    def rule_stopping_criterion(self, theory: Theory, rule: Rule,
                                examples) -> bool:
        # return True iff rule covers no examples
        p, n, P, N = count_matches(rule, self.target_class, examples)
        return p == 0

    def post_process(self, theory: Theory) -> Theory:
        return theory


class CN2Estimator(SeCoEstimator):
    def __init__(self, LRS_threshold: float=1.0, multi_class="one_vs_rest", n_jobs=1):
        super().__init__(CN2Implementation(LRS_threshold), multi_class, n_jobs)


if __name__ == "__main__":
    check_estimator(SimpleSeCoEstimator)
    check_estimator(CN2Estimator)
