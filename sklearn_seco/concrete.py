"""
Implementation of SeCo / Covering algorithm:
Usual building blocks & known instantiations of the abstract base algorithm.

Implemented as Mixins. For __init__ parameters, they use cooperative
multi-inheritance, each class has to declare **kwargs and forward anything it
doesn't consume using `super().__init__(**kwargs)`. Users of mixin-composed
classes will have to use keyword- instead of positional arguments.
"""

import math
import warnings
from functools import lru_cache
from itertools import zip_longest
from typing import Tuple, Iterable, Optional, Union, Sequence

import numpy as np
from matplotlib.figure import Figure
from sklearn.exceptions import NotFittedError

from sklearn_seco.abstract import Theory, SeCoEstimator
from sklearn_seco.common import \
    RuleQueue, SeCoBaseImplementation, AugmentedRule, rule_ancestors


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    # copied from itertools docs
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# Mixins providing implementation facettes


class BeamSearch(SeCoBaseImplementation):
    """Mixin implementing a beam search of width `n`.

    - The default `beam_width` of 1 signifies a hill climbing search, only ever
      optimizing one candidate rule.
    - A special `beam_width` of 0 means trying all candidate rules, i.e. a
      best-first search.

    Rule selection is done in `filter_rules`, while `select_candidate_rules`
    always returns the whole queue as candidates.
    """
    def __init__(self, beam_width: int = 1, **kwargs):
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

    def set_context(self, estimator, X, y):
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


class SignificanceStoppingCriterion(SeCoBaseImplementation):
    """Mixin using as stopping criterion for rule refinement a significance
    test like CN2.
    """
    def __init__(self, LRS_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.LRS_threshold = LRS_threshold  # FIXME: estimator.set_param not reflected here

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        """*Significance test* as defined by (Clark and Niblett 1989), but used
        there for rule evaluation, here instead used as stopping criterion
        following (Clark and Boswell 1991).
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


class NoPostPruning(SeCoBaseImplementation):
    """Mixin to skip post-pruning"""
    def simplify_rule(self, rule: AugmentedRule) -> AugmentedRule:
        return rule


class NoPostProcess(SeCoBaseImplementation):
    """Mixin to skip post processing."""
    def post_process(self, theory: Theory) -> Theory:
        return theory


class TraceCoverage(SeCoBaseImplementation):
    """Mixin tracing (p,n) while building the theory, able to plot these.

    NOTE: Always make sure to place this before all other Mixins!
      Otherwise the methods `inner_stopping_criterion` and
      `rule_stopping_criterion` won't work, because other classes are not
      required to cooperate on these methods (i.e. call `super()`).
      TODO: fix uncooperative multi-inheritance for tracing. use @decorator?

    - `trace_level`
      specifies detail level to trace:
        - `theory` only traces each of the `best_rule` in the theory
        - `ancestors` traces each `best_rule` and the refinement steps used
          to find it (starting with the `init_rule` return value).
        - `refinements` traces every rule generated by `refine_rule`.

    - `coverage_log`: list of np.array with shape (n_ancestors, 2)
       This is the log of `best_rule` and (if requested) their `ancestors`.
       For each `best_rule` +1 it keeps an array with (n,p) for that best rule
       and its ancestors, in reverse order of creation.
       The last `best_rule` is not part

    - `refinement_log`: list of np.array with shape (n_refinements, 3)
       This is the log of all `refinements`. For each `best_rule` (i.e. iteration
       in `abstract_seco`) +1 (which corresponds to the attempts to find
       another rule, aborted by `rule_stopping_criterion`), it keeps an array
       with (n, p, stop) for each refinement, where `stop` is the boolean
       result of `inner_stopping_criterion(refinement)`.

    - `last_rule_stop`: boolean result of `rule_stopping_criterion` on the last
       found `best_rule`. If False, it is part of the theory (and the rule
       search ended because all positive examples were covered), if True it is
       not part of the theory (the search ended because
       `rule_stopping_criterion` was True).

    - `NP`: np.array of shape (n_best_rules, 2)
       This is the log of the (N, P) values for each iteration of
       `abstract_seco`.
    """

    def __init__(self, trace_level='refinements', **kwargs):
        super().__init__(**kwargs)
        self.trace_level = trace_level
        self.has_complete_trace = False
        self.coverage_log = []
        self.refinement_log = []
        self.NP = []
        self.last_rule_stop = None

    def set_context(self, estimator: '_BinarySeCoEstimator', X, y):
        # note we're called before each find_best_rule
        super().set_context(estimator, X, y)
        # if we're in a new abstract_seco run: delete the old results
        if self.has_complete_trace:
            self.has_complete_trace = False
            self.coverage_log = []
            self.refinement_log = []  # type while tracing: list of
                                      # (list or np.ndarray) of
                                      # np.array [n,p,stop]
            self.last_rule_stop = None
            self.NP = []  # type while tracing: list of tuple(int, int)
        self.NP.append((self.N, self.P))

        if self.trace_level == 'refinements':
            self.refinement_log.append([])

    def inner_stopping_criterion(self, refinement: AugmentedRule) -> bool:
        stop = super().inner_stopping_criterion(refinement)
        p, n = self.count_matches(refinement)
        self.refinement_log[-1].append(np.array((n, p, stop)))
        return stop

    def rule_stopping_criterion(self, theory: Theory, rule: AugmentedRule
                                ) -> bool:
        self.last_rule_stop = super().rule_stopping_criterion(theory, rule)
        # note usage of (x=n, y=p) instead of (p,n) due to plot coordinates

        def rnp(rule):
            p, n = self.count_matches(rule)
            return n, p

        if self.trace_level == 'best_rules':
            self.coverage_log.append( np.array([rnp(rule)]) )
        else:  # elif trace_level in ('ancestors', 'refinements'):
            self.coverage_log.append(
                np.array([rnp(r) for r in rule_ancestors(rule)]) )

        if self.trace_level == 'refinements':
            self.refinement_log[-1] = np.array(self.refinement_log[-1])

        return self.last_rule_stop

    def unset_context(self):
        super().unset_context()
        # end of rule search, theory is complete
        self.has_complete_trace = True
        self.NP = np.array(self.NP)

    @property
    def n_rules(self):
        return len(self.coverage_log)

    # TODO: (de)serialize log to plot separately

    def plot_coverage_log(
            self,
            title: Optional[str] = None,
            draw_refinements: Union[str, bool] = 'nonzero',
            theory_figure: Optional[Figure] = None,
            rules_figure: Union[Figure, Sequence[Figure], None] = None,
            rules_use_subfigures: bool = True,
    ) -> Tuple[Figure, Union[Figure, Sequence[Figure]]]:
        """TODO: doc

        :param title: string or None. If not None, set figure titles and use
          this value as prefix.
        :param theory_figure: If None, use `plt.figure()` to create a figure
          for the theory plot, otherwise use this parameter.
        :param rules_figure: If `None`, use `plt.figure()` to create a figure
          (using subfigures) or a list of figures for the rules plot(s),
          otherwise use this parameter.
          If `rules_use_subfigure`, `rules_figure` has to be None or a list of
          figures of same length as `coverage_log`.
        :param rules_use_subfigures: If True, the rules plots are placed as
          subfigures in a common figure, otherwise they're drawn as separate
          figures.
        :param draw_refinements: If `True`, draw all refinements, if `False`
          don't. If `'nonzero'` (the default) only draw those with `n > 0`.
        :return: `(theory_figure, rules_figure)` where rules_figure is a figure
          or a list of figure, depending on `rules_use_subfigure`.
        """
        # TODO use structured ndarrays & plot(xlabel, ylabel, data) variant
        if not self.has_complete_trace:
            raise NotFittedError("No trace collected yet.")
        if not self.n_rules:
            # issue a warning, user can decide handling. See module `warnings`
            warnings.warn("Empty coverage_log collected, useless plot.")
        if draw_refinements and len(self.refinement_log) < self.n_rules:
            warnings.warn("draw_refinements=True requested, but no "
                          "refinement_log collected. "
                          "Using draw_refinements=False.")
            draw_refinements = False

        NP0 = self.NP[0]
        rnd_style = dict(color='grey', alpha=0.5, linestyle='dotted')
        refinements_style = dict(marker='.', markersize=1, linestyle='',
                                 zorder=-1,)

        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoLocator

        if theory_figure is None:
            theory_figure = plt.figure()
        theory_axes = theory_figure.gca(xlabel='n', ylabel='p',
                                        xlim=(0, NP0[0]), ylim=(0, NP0[1]))
        # draw "random theory" reference marker
        theory_axes.plot([0, NP0[0]], [0, NP0[1]], **rnd_style)

        if rules_use_subfigures:
            if rules_figure is None:
                rules_figure = plt.figure(figsize=(10.24, 10.24),
                                          tight_layout=True)
            # else assume rules_figure is already a figure
            if title is not None:
                rules_figure.suptitle("%s: Rules" % title, y=0.02)
            # TODO: axis labels when using subfigures
            subfigure_grid = [math.ceil(np.sqrt(self.n_rules))] * 2
            rule_axes = [rules_figure.add_subplot(*subfigure_grid, rule_idx +1)
                         for rule_idx in range(self.n_rules)]
        else:
            if rules_figure is None:
                rules_figure = [plt.figure() for _ in range(self.n_rules)]
            elif isinstance(rules_figure, Sequence):
                if len(rules_figure) != self.n_rules:
                    raise ValueError("rules_figure is a list, but length ({}) "
                                     "differs from self.n_rules ({})."
                                     .format(len(rules_figure), self.n_rules))
            else:
                raise ValueError("got rules_use_subfigures=False, thus "
                                 "rules_figure has to be None or a list of "
                                 "figures, instead got " + str(rules_figure))
            # assume rules_figure is a list of figures
            rule_axes = [f.gca() for f in rules_figure]

        previous_rule = np.array((0, 0))  # equals (N, P) for some trace
        for rule_idx, (rule_trace, refinements) in \
                enumerate(zip_longest(self.coverage_log, self.refinement_log)):
            NP = self.NP[rule_idx]
            if draw_refinements == 'nonzero':
                refts_mask = refinements[:, 1] != 0
            elif draw_refinements:
                refts_mask = slice(None)  # all
            mark_stop = self.last_rule_stop and (rule_idx == self.n_rules -1)

            # this rule in theory plot
            rule = rule_trace[0] + previous_rule
            theory_line = theory_axes.plot(
                rule[0], rule[1], 'x' if mark_stop else '.',
                label="{i:{i_width}}: ({p:4}, {n:4})"
                      .format(n=rule[0], p=rule[1], i=rule_idx,
                              i_width=math.ceil(np.log10(self.n_rules))))
            rule_color = theory_line[0].get_color()
            if draw_refinements:
                # draw refinements in theory plot
                theory_axes.plot(refinements[refts_mask,0] + previous_rule[0],
                                 refinements[refts_mask,1] + previous_rule[1],
                                 color=rule_color, alpha=0.3,
                                 **refinements_style)
            # draw arrows between best_rules
            theory_axes.annotate("", xytext=previous_rule, xy=rule,
                                 arrowprops={'arrowstyle': "->"})
            previous_rule = rule

            # TODO: move ancestor plot to separate method
            # subplot with ancestors of current rule
            rule_axis = rule_axes[rule_idx]
            if mark_stop:
                rule_axis.set_title('(Rule #%d) Candidate' % rule_idx)
            else:
                rule_axis.set_title('Rule #%d' % rule_idx)
            # draw "random theory" reference marker
            rule_axis.plot([0, NP0[0]], [0, NP0[1]], **rnd_style)
            # draw rule_trace
            rule_axis.plot(rule_trace[:, 0], rule_trace[:, 1], 'o-',
                           color=rule_color)
            if draw_refinements:
                # draw refinements as scattered dots
                rule_axis.plot(refinements[refts_mask, 0],
                               refinements[refts_mask, 1],
                               color='black', alpha=0.7, **refinements_style)

            # draw x and y axes through (0,0) and hide for negative values
            for spine_type, spine in rule_axis.spines.items():
                spine.set_position('zero')
                horizontal = spine_type in {'bottom', 'top'}
                spine.set_bounds(0, NP[0] if horizontal else NP[1])

            class PositiveTicks(AutoLocator):
                def tick_values(self, vmin, vmax):
                    orig = super().tick_values(vmin, vmax)
                    return orig[orig >= 0]

            rule_axis.xaxis.set_major_locator(PositiveTicks())
            rule_axis.yaxis.set_major_locator(PositiveTicks())
            rule_axis.locator_params(integer=True)

            # set reference frame (N,P), but move (0,0) so it looks comparable
            rule_axis.set_xbound(NP[0] - NP0[0], NP[0])
            rule_axis.set_ybound(NP[1] - NP0[1], NP[1])

        if title is not None:
            theory_axes.set_title("%s: Theory" % title)
        theory_figure.legend(title="rule: (p,n)", loc='center right')

        return theory_figure, rules_figure


# Example Algorithm configurations


class SimpleSeCoImplementation(BeamSearch,
                               TopDownSearch,
                               PurityHeuristic,
                               NoPostPruning,
                               NoPostProcess):

    def inner_stopping_criterion(self, rule: AugmentedRule) -> bool:
        p, n = self.count_matches(rule)
        return n == 0

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
        """abort search if rule covers no examples"""
        p, n = self.count_matches(rule)
        return p == 0


class CN2Estimator(SeCoEstimator):
    """Estimator using :class:`CN2Implementation`."""
    def __init__(self,
                 LRS_threshold: float = 0.9,
                 multi_class="one_vs_rest",
                 n_jobs=1):
        super().__init__(CN2Implementation(LRS_threshold=LRS_threshold),
                         multi_class, n_jobs)
        # sklearn assumes all parameters are class fields, so copy this here
        self.LRS_threshold = LRS_threshold


# TODO: don't require definition of 2 classes, add *Estimator factory method in SeCoBaseImplementation
