"""
Implementation of SeCo / Covering algorithm:
Helpers in addition to the algorithms in `concrete.py`.
"""

import math
import warnings
from itertools import zip_longest
from typing import Optional, Union, Sequence, Tuple, Type, Callable

import numpy as np
from matplotlib.figure import Figure  # needed only for type hints

from sklearn_seco.abstract import SeCoEstimator
from sklearn_seco.common import \
    AugmentedRule, Theory, rule_ancestors, \
    AbstractSecoImplementation, RuleContext, TheoryContext


LogTraceCallback = Callable[[Sequence[np.ndarray],  # coverage_log
                             Sequence[np.ndarray],  # refinement_log
                             bool,  # last_rule_stop
                             np.ndarray],  # PN
                            None]


def trace_coverage(est_cls: Type[SeCoEstimator],
                   log_coverage_trace_callback: LogTraceCallback,
                   ) -> Type['SeCoEstimator']:
    """Decorator for `SeCoEstimator` that adds tracing of (p,n) while building
    the theory. Traces can be plotted with `plot_coverage_log`.

    Accesses the internal details of the implementation to trace all
    intermediate rule objects as well as the ones which become part of the
    final theory. Just before that one is constructed (before the
    `post_process` hook), the collected trace is submitted to the
    `log_coverage_trace_callback` function.
    For non-binary problems, a `SeCoEstimator` spawns more than one
    `_BinarySeCoEstimator`, each of which collects the trace & submits it
    separately. Parallel execution of these has to be synchronized manually,
    this has not been tested.

    The trace gets the values P,N,p,n from the `RuleContext` and
    `AugmentedRule` objects; if `GrowPruneSplit` is used, this usually means
    that these are only computed on the growing set.
    TODO: get P,N,p,n for grow+prune, not only growing set.

    The trace consists of the following fields:

    - `coverage_log`: list of np.array with shape (n_ancestors, 2)
       This is the log of `best_rule` and their `ancestors`.

       For each `best_rule` +1 it keeps an array with (p, n) for that best rule
       and its ancestors, in reverse order of creation. If `last_rule_stop` is
       False, the last `best_rule` has been rejected by the
       `rule_stopping_criterion` and is not part of the theory.

       If `TraceCoverageTheoryContext.trace_level` is 'theory', no ancestors
       are traced, so only the `best_rule`s are contained.

    - `refinement_log`: list of np.array with shape (n_refinements, 3)
       This is the log of all `refinements`. For each `best_rule` (i.e.
       iteration in `abstract_seco`) +1 (which corresponds to the attempts to
       find another rule, aborted by `rule_stopping_criterion`), it keeps an
       array with (p, n, stop) for each refinement, where `stop` is the boolean
       result of `inner_stopping_criterion(refinement)`.

       If `TraceCoverageTheoryContext.trace_level` is not 'refinements', this
       is empty.

    - `last_rule_stop`: boolean result of `rule_stopping_criterion` on the last
       found `best_rule`. If False, it is part of the theory (and the rule
       search ended because all positive examples were covered), if True it is
       not part of the theory (the search ended because
       `rule_stopping_criterion` was True).

    - `PN`: np.array of shape (n_best_rules, 2)
       This is the log of the (P, N) values for each iteration of
       `abstract_seco`.

    # TODO: implement tracing of estimator instances (not only classes)

    Usage
    =====
    Define the callback function to receive the trace & display it:

    >>> def callback(coverage_log, refinement_log, last_rule_stop, PN):
    ...     theory_figure, rules_figure = plot_coverage_log(
    ...         coverage_log, refinement_log, last_rule_stop, PN)
    ...     theory_figure.show()
    ...     rules_figure.show()

    Use with decorator syntax:

    >>> @trace_coverage
    ... class MySeCoEstimator(SeCoEstimator, callback):
    ...     ...

    or call directly:

    >>> MyTracedSeCo = trace_coverage(MySeCoEstimator, callback)
    """

    original_post_process = \
        est_cls.algorithm_config.Implementation.post_process

    class TracedEstimator(est_cls):
        class algorithm_config(est_cls.algorithm_config):
            class Implementation(TraceCoverageImplementation,
                                 est_cls.algorithm_config.Implementation):

                @classmethod
                def post_process(cls, theory: Theory,
                                 context: 'TraceCoverageTheoryContext'):
                    assert isinstance(context, TraceCoverageTheoryContext)
                    # transfer collected trace from the `context` object which
                    # is in local scope
                    log_coverage_trace_callback(
                        context.coverage_log, context.refinement_log,
                        context.last_rule_stop, np.array(context.PN))
                    return original_post_process(theory, context)

            class TheoryContextClass(
                    TraceCoverageTheoryContext,
                    est_cls.algorithm_config.TheoryContextClass):
                pass

            class RuleContextClass(TraceCoverageRuleContext,
                                   est_cls.algorithm_config.RuleContextClass):
                pass

    return TracedEstimator


class TraceCoverageTheoryContext(TheoryContext):
    """Tracing `TheoryContext`.

    Fields
    -----

    - `trace_level` specifies detail level to trace:
        - `theory` only traces each of the `best_rule` part of the theory
        - `ancestors` traces each `best_rule` and the refinement steps used
          to find it (starting with the `init_rule` return value).
        - `refinements` traces every rule generated by `refine_rule`.
    """

    trace_level = 'refinements'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_log = []
        self.refinement_log = []
        self.last_rule_stop = None
        self.PN = []


class TraceCoverageRuleContext(RuleContext):
    """Tracing `RuleContext`."""
    def __init__(self, theory_context: TraceCoverageTheoryContext, X, y):
        super().__init__(theory_context, X, y)
        assert isinstance(theory_context, TraceCoverageTheoryContext)
        theory_context.PN.append((self.PN))
        if theory_context.trace_level == 'refinements':
            theory_context.refinement_log.append([])


class TraceCoverageImplementation(AbstractSecoImplementation):
    """Tracing `AbstractSecoImplementation` mixin.

    Note: This class always has to come first in the mro/inheritance tree,
    because it overrides some methods not to be overridden. Use the
    `trace_coverage` decorator to ensure that.
    """

    @classmethod
    def inner_stopping_criterion(cls, refinement: AugmentedRule,
                                 context: RuleContext) -> bool:
        stop = super().inner_stopping_criterion(refinement, context)
        p, n = context.count_matches(refinement)
        assert isinstance(context.theory_context, TraceCoverageTheoryContext)
        context.theory_context.refinement_log[-1]. \
            append(np.array((p, n, stop)))
        return stop

    @classmethod
    def rule_stopping_criterion(cls, theory: Theory, rule: AugmentedRule,
                                context: RuleContext) -> bool:
        tctx: TraceCoverageTheoryContext = context.theory_context
        assert isinstance(tctx, TraceCoverageTheoryContext)
        tctx.last_rule_stop = super().rule_stopping_criterion(theory, rule,
                                                              context)

        def pn(rule):
            return context.count_matches(rule)

        if tctx.trace_level == 'best_rules':
            tctx.coverage_log.append(np.array([pn(rule)]))
        else:  # elif trace_level in ('ancestors', 'refinements'):
            tctx.coverage_log.append(
                np.array([pn(r) for r in rule_ancestors(rule)]))

        if tctx.trace_level == 'refinements':
            tctx.refinement_log[-1] = np.array(tctx.refinement_log[-1])

        return tctx.last_rule_stop


def plot_coverage_log(
        coverage_log, refinement_log, last_rule_stop, PN,
        *,
        title: Optional[str] = None,
        draw_refinements: Union[str, bool] = 'nonzero',
        theory_figure: Optional[Figure] = None,
        rules_figure: Union[Figure, Sequence[Figure], None] = None,
        rules_use_subfigures: bool = True,
) -> Tuple[Figure, Union[Figure, Sequence[Figure]]]:
    """Plot the traced (p, n) of a theory.

    For parameters `coverage_log`, `refinement_log`, `last_rule_stop`, and
    `PN`, see documentation of `trace_coverage`.

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

    P, N = 0, 1  # readable indexes, not values!

    n_rules = len(coverage_log)
    if not n_rules:
        # issue a warning, user can decide handling. See module `warnings`
        warnings.warn("Empty coverage_log collected, useless plot.")
    if draw_refinements and len(refinement_log) < n_rules:
        warnings.warn("draw_refinements=True requested, but no refinement_log "
                      "collected. Using draw_refinements=False.")
        draw_refinements = False

    PN0 = PN[0]
    rnd_style = dict(color='grey', alpha=0.5, linestyle='dotted')
    refinements_style = dict(marker='.', markersize=1, linestyle='',
                             zorder=-1,)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator

    if theory_figure is None:
        theory_figure = plt.figure()
    theory_axes = theory_figure.gca(xlabel='n', ylabel='p',
                                    xlim=(0, PN0[N]), ylim=(0, PN0[P]))
    theory_axes.locator_params(integer=True)
    # draw "random theory" reference marker
    theory_axes.plot([0, PN0[N]], [0, PN0[P]], **rnd_style)

    if rules_use_subfigures:
        if rules_figure is None:
            rules_figure = plt.figure(figsize=(10.24, 10.24),
                                      tight_layout=True)
        # else assume rules_figure is already a figure
        if title is not None:
            rules_figure.suptitle("%s: Rules" % title, y=0.02)
        # TODO: axis labels when using subfigures
        subfigure_grid = [math.ceil(np.sqrt(n_rules))] * 2
        rule_axes = [rules_figure.add_subplot(*subfigure_grid, rule_idx + 1)
                     for rule_idx in range(n_rules)]
    else:
        if rules_figure is None:
            rules_figure = [plt.figure() for _ in range(n_rules)]
        elif isinstance(rules_figure, Sequence):
            if len(rules_figure) != n_rules:
                raise ValueError("rules_figure is a list, but length ({}) "
                                 "differs from n_rules ({})."
                                 .format(len(rules_figure), n_rules))
        else:
            raise ValueError("got rules_use_subfigures=False, thus "
                             "rules_figure has to be None or a list of "
                             "figures, instead got " + str(rules_figure))
        # assume rules_figure is a list of figures
        rule_axes = [f.gca() for f in rules_figure]

    previous_rule = np.array((0, 0))  # contains (P, N) for some trace
    for rule_idx, (rule_trace, refinements) in \
            enumerate(zip_longest(coverage_log, refinement_log)):
        PNi = PN[rule_idx]
        # TODO: visualize whether refinement was rejected due to inner_stop
        if draw_refinements == 'nonzero':
            refts_mask = refinements[:, P] != 0
        elif draw_refinements:
            refts_mask = slice(None)  # all
        mark_stop = last_rule_stop and (rule_idx == n_rules - 1)

        # this rule in theory plot
        rule = rule_trace[0] + previous_rule
        theory_line = theory_axes.plot(
            rule[N], rule[P], 'x' if mark_stop else '.',
            label="{i:{i_width}}: ({p:4}, {n:4})"
                  .format(n=rule[N], p=rule[P], i=rule_idx,
                          i_width=math.ceil(np.log10(n_rules))))
        rule_color = theory_line[0].get_color()
        if draw_refinements:
            # draw refinements in theory plot
            theory_axes.plot(refinements[refts_mask, N] + previous_rule[N],
                             refinements[refts_mask, P] + previous_rule[P],
                             color=rule_color, alpha=0.3,
                             **refinements_style)
        # draw arrows between best_rules
        # NOTE: invert coordinates because we save (p,n) and plot (x=n,y=p)
        theory_axes.annotate("", xytext=previous_rule[::-1], xy=rule[::-1],
                             arrowprops={'arrowstyle': "->"})
        previous_rule = rule

        # subplot with ancestors of current rule
        rule_axis = rule_axes[rule_idx]
        if mark_stop:
            rule_title_template = '(Rule #%d) Candidate'
        else:
            rule_title_template = 'Rule #%d'
        if not rules_use_subfigures and title is not None:
            # add title before each of the figures
            rule_title_template = "%s: %s" % (title, rule_title_template)
        rule_axis.set_title(rule_title_template % rule_idx)
        # draw "random theory" reference marker
        rule_axis.plot([0, PN0[N]], [0, PN0[P]], **rnd_style)
        # draw rule_trace
        rule_axis.plot(rule_trace[:, N], rule_trace[:, P], 'o-',
                       color=rule_color)
        if draw_refinements:
            # draw refinements as scattered dots
            rule_axis.plot(refinements[refts_mask, N],
                           refinements[refts_mask, P],
                           color='black', alpha=0.7, **refinements_style)

        # draw x and y axes through (0,0) and hide for negative values
        for spine_type, spine in rule_axis.spines.items():
            spine.set_position('zero')
            horizontal = spine_type in {'bottom', 'top'}
            spine.set_bounds(0, PNi[N] if horizontal else PNi[P])

        class PositiveTicks(AutoLocator):
            def tick_values(self, vmin, vmax):
                orig = super().tick_values(vmin, vmax)
                return orig[orig >= 0]

        rule_axis.xaxis.set_major_locator(PositiveTicks())
        rule_axis.yaxis.set_major_locator(PositiveTicks())
        rule_axis.locator_params(integer=True)

        # set reference frame (N,P), but move (0,0) so it looks comparable
        rule_axis.set_xbound(PNi[N] - PN0[N], PNi[N])
        rule_axis.set_ybound(PNi[P] - PN0[P], PNi[P])

    if title is not None:
        theory_axes.set_title("%s: Theory" % title)
    theory_figure.legend(title="rule: (p,n)", loc='center right')

    return theory_figure, rules_figure
