"""
Implementation of SeCo / Covering algorithm:
Known instantiations / example configurations of the abstract base algorithm.
"""

from sklearn_seco.abstract import SeCoEstimator
from sklearn_seco.common import \
    SeCoAlgorithmConfiguration, AugmentedRule, RuleContext, Theory
from sklearn_seco.concrete import TopDownSearchContext, BeamSearch, \
    TopDownSearchImplementation, PurityHeuristic, NoNegativesStop, \
    SkipPostPruning, CoverageRuleStop, SkipPostProcess, LaplaceHeuristic, \
    SignificanceStoppingCriterion, PositiveThresholdRuleStop, \
    ConditionTracingAugmentedRule, RipperMdlRuleStopTheoryContext, \
    InformationGainHeuristic, RipperMdlRuleStopImplementation, \
    RipperPostPruning, delayed_inner_stop, GrowPruneSplitRuleContext

# TODO: create own mixins from the methods still implemented here


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
            beam_width = 3  # → BeamSearch


class RipperEstimator(SeCoEstimator):
    """Ripper as defined by (Cohen 1995).

    NOTE: The global post-optimization phase is currently not implemented
        (that would be the `post_process` method).
    """
    class algorithm_config(SeCoAlgorithmConfiguration):
        RuleClass = ConditionTracingAugmentedRule
        TheoryContextClass = RipperMdlRuleStopTheoryContext

        class Implementation(BeamSearch,
                             TopDownSearchImplementation,
                             InformationGainHeuristic,
                             RipperMdlRuleStopImplementation,
                             RipperPostPruning,
                             SkipPostProcess
                             ):
            @classmethod
            @delayed_inner_stop
            def inner_stopping_criterion(cls, rule: AugmentedRule,
                                         context: RuleContext) -> bool:
                """Laplace-based criterion. Field `accuRate` in JRip.java."""
                p, n = rule.pn(context)
                accuracy_rate = (p + 1) / (p + n + 1)
                return accuracy_rate >= 1

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
                p, n = rule.pn(context)
                return (p + 1) / (p + n + 2)  # laplace


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
                p, n = rule.pn(context)
                P, N = context.PN(rule.head)
                if P + N == 0:
                    return 0
                tn = N - n
                return (p + tn) / (P + N)
