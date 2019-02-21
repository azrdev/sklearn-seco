"""
Implementation of SeCo / Covering algorithm:
Helpers in addition to the algorithms in `concrete.py`.
"""

import json
import math
import warnings
from typing import Optional, Union, Sequence, Tuple, Type, Callable, \
    NamedTuple, MutableSequence, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
from matplotlib.ticker import AutoLocator
from matplotlib.figure import Figure  # needed only for type hints
from matplotlib.patches import Polygon

from sklearn_seco.abstract import SeCoEstimator
from sklearn_seco.common import \
    AugmentedRule, Theory, rule_ancestors, \
    AbstractSecoImplementation, RuleContext, TheoryContext
from sklearn_seco.concrete import GrowPruneSplitRuleContext


class Trace:
    """Trace of a seco algorithm run, i.e. `abstract_seco` invocation.

    Attributes
    -----
    - `steps`: Sequence[TraceEntry]
      Each TraceEntry represents one step in `abstract_seco`, and thus one rule
      in the learned theory. The last step/rule is only part of the theory iff
      last_rule_stop is False.

    - `last_rule_stop`: boolean
      result of `rule_stopping_criterion` on the last found `best_rule`. If
      False, it is part of the theory (and the rule search ended because all
      positive examples were covered), if True it is not part of the theory
      (the search ended because `rule_stopping_criterion` was True).

    - `P_total`: int
      Count of positive examples in the whole dataset (differs from
      `steps[0].P` if grow-prune-splitting is used).

    - `N_total`: int
      Count of negative examples in the whole dataset (differs from
      `steps[0].N` if grow-prune-splitting is used).
    """

    _JSON_DUMP_DESCRIPTION = "sklearn_seco.extra.trace_coverage dump"
    _JSON_DUMP_VERSION = 2

    @staticmethod
    def _json_encoder(obj):
        """Serialize `obj` when used as `json.JSONEncoder.default` method."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Trace.TraceEntry):
            return obj.__dict__
        raise TypeError

    steps: MutableSequence['TraceEntry']
    last_rule_stop: bool
    P_total: int
    N_total: int

    def __init__(self, P, N):
        self.steps = []
        self.last_rule_stop = None
        self.P_total = P
        self.N_total = N

    class TraceEntry:
        """Trace of one seco algorithm step, i.e. `find_best_rule` invocation.

        Attributes
        -----
        `ancestors`: list of np.array with shape (n_ancestors, 2)
           This is the log of the learned `best_rule` and its `ancestors`.

           An array with (p, n) for the rule and each ancestor, in reverse order of
           creation.

           If `TraceCoverageTheoryContext.trace_level` is 'theory', no ancestors
           are traced, so only the `best_rule` is contained.

        `refinements`: np.array or list of np.array with shape (n_refinements, 3)
           This is the log of all `refinements`, an array with (p, n, stop) for
           each refinement, where `stop` is the boolean result of
           `inner_stopping_criterion(refinement)`.

           Empty if `TraceCoverageTheoryContext.trace_level` is not 'refinements'.

        P: int
          Count of positive examples

        N: int
          Count of negative examples
        """
        AncestorEntry = Union[NamedTuple('Coverage', [('p', int), ('n', int)]),
                              np.ndarray]
        RefinementEntry = Union[NamedTuple('Refinements', [('p', int),
                                                           ('n', int),
                                                           ('stop', bool)]),
                                np.ndarray]

        def __init__(self, ancestors: Sequence[AncestorEntry],
                     refinements: MutableSequence[RefinementEntry], P, N):
            self.ancestors = ancestors
            self.refinements = refinements
            self.P = P
            self.N = N

        def __eq__(self, other):
            return all((hasattr(other, 'ancestors'),
                        np.all(np.equal(self.ancestors, other.ancestors)),
                        hasattr(other, 'refinements'),
                        np.all(np.equal(self.refinements, other.refinements)),
                        hasattr(other, 'P'),
                        self.P == other.P,
                        hasattr(other, 'N'),
                        self.N == other.N,
                        ))

        ancestors: Sequence[AncestorEntry]
        refinements: MutableSequence[RefinementEntry]
        P: int
        N: int

    def __eq__(self, other):
        return all((hasattr(other, 'steps'),
                    self.steps == other.steps,
                    hasattr(other, 'last_rule_stop'),
                    self.last_rule_stop == other.last_rule_stop,
                    hasattr(other, 'P_total'),
                    self.P_total == other.P_total,
                    hasattr(other, 'N_total'),
                    self.N_total == other.N_total))

    def to_json(self):
        """:return: A string containing a JSON representation of the trace."""
        return json.dumps({
            "description": Trace._JSON_DUMP_DESCRIPTION,
            "version": Trace._JSON_DUMP_VERSION,
            "steps": self.steps,
            "last_rule_stop": self.last_rule_stop,
            "P_total": self.P_total,
            "N_total": self.N_total,
        }, allow_nan=False, default=Trace._json_encoder)

    @staticmethod
    def from_json(dump) -> 'Trace':
        """
        :param dump: A file-like object or string containing JSON.
        :return : A dict representing the trace dumped previously with
            `to_json`. To plot, pass it to :func:`plot_coverage_log`.
        """
        loader = json.loads if isinstance(dump, str) else json.load
        dec = loader(dump)

        if dec["description"] != Trace._JSON_DUMP_DESCRIPTION:
            raise ValueError("No/invalid coverage trace json: %s" % repr(dec))
        if dec["version"] != Trace._JSON_DUMP_VERSION:
            raise ValueError("Unsupported coverage trace version: %s"
                             % dec["version"])
        # convert back to numpy arrays
        trace = Trace(dec['P_total'], dec['N_total'])
        trace.last_rule_stop = dec['last_rule_stop']
        trace.steps = [Trace.TraceEntry(step['ancestors'],
                                        step['refinements'],
                                        step['P'], step['N'])
                       for step in dec['steps']]
        return trace

    def plot_coverage_log(self, **kwargs):
        """Plot the trace, see :func:`plot_coverage_log`."""
        return plot_coverage_log(self, **kwargs)


LogTraceCallback = Callable[[Trace], None]


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
    TODO: maybe get P,N,p,n for grow+prune, not only growing set.

    # TODO: implement tracing of estimator instances (not only classes)

    Usage
    =====
    Define the callback function to receive the trace & display it:

    >>> def callback(coverage_log, last_rule_stop):
    ...     theory_figure, rules_figure = plot_coverage_log(
    ...         coverage_log, last_rule_stop)
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
                    log_coverage_trace_callback(context.trace)
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
        # TODO: dedup P,N calculation with RuleContext.PN
        P = np.count_nonzero(self.complete_y == self.target_class)
        N = len(self.complete_y) - P
        self.trace: Trace = Trace(P, N)


class TraceCoverageRuleContext(RuleContext):
    """Tracing `RuleContext`."""
    def __init__(self, theory_context: TheoryContext, X, y):
        super().__init__(theory_context, X, y)
        P, N = self.PN  # always growing  if GrowPruneSplitRuleContext is used
        self.current_trace_entry = Trace.TraceEntry([], [], P, N)


class TraceCoverageImplementation(AbstractSecoImplementation):
    """Tracing `AbstractSecoImplementation` mixin.

    Note: This class always has to come first in the mro/inheritance tree,
    because it overrides some methods not to be overridden. Use the
    `trace_coverage` decorator to ensure that.
    """

    @classmethod
    def inner_stopping_criterion(cls, refinement: AugmentedRule,
                                 context: TraceCoverageRuleContext) -> bool:
        stop = super().inner_stopping_criterion(refinement, context)
        assert isinstance(context, TraceCoverageRuleContext)
        p, n = context.count_matches(refinement)
        context.current_trace_entry.refinements.append(np.array((p, n, stop)))
        return stop

    @classmethod
    def rule_stopping_criterion(cls, theory: Theory, rule: AugmentedRule,
                                context: TraceCoverageRuleContext) -> bool:
        last_rule_stop = super().rule_stopping_criterion(theory, rule, context)

        tctx = context.theory_context
        assert isinstance(tctx, TraceCoverageTheoryContext)
        assert isinstance(context, TraceCoverageRuleContext)
        current_entry = context.current_trace_entry
        if isinstance(context, GrowPruneSplitRuleContext):
            context.growing = True
        if tctx.trace_level == 'best_rules':
            current_entry.ancestors = np.array([context.count_matches(rule)])
        else:  # elif trace_level in ('ancestors', 'refinements'):
            current_entry.ancestors = np.array(
                [context.count_matches(r) for r in rule_ancestors(rule)])

        if tctx.trace_level == 'refinements':
            current_entry.refinements = np.array(current_entry.refinements)

        tctx.trace.steps.append(current_entry)
        tctx.trace.last_rule_stop = last_rule_stop
        return last_rule_stop


def _draw_outer_border(ax: axes, xmin, xmax, ymin, ymax, **kwargs):
    """
    Fill the right upper corner of a diagram, everything beyond (xmin, ymin).

    The following graphic shows the filled regions with dots:

          ↑
    ymax  +..............
          |..............
    ymin  +..............
          |       .......
          |       .......
          +-------+-----+-→
                xmin   xmax
    """
    verts = [(xmin, ymin), (xmin, 0), (xmax, 0),
             (xmax, ymax), (0, ymax), (0, ymin)]
    return ax.add_patch(Polygon(np.array(verts), **kwargs))


def plot_coverage_log(
        trace: Trace,
        *,
        title: Optional[str] = None,
        draw_refinements: Union[str, bool] = 'nonzero',
        theory_figure: Optional[Figure] = None,
        rules_figure: Union[Figure, Sequence[Figure], None] = None,
        rules_use_subfigures: bool = True,
) -> Tuple[Figure, Union[Figure, Sequence[Figure]]]:
    """Plot the traced (p, n) of a theory.

    :param trace: collected `Trace`, see also `trace_coverage`.
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

    n_rules = len(trace.steps)
    if not n_rules:
        # issue a warning, user can decide handling. See module `warnings`
        warnings.warn("Empty trace collected, useless plot.")
    if draw_refinements:
        for index, le in enumerate(trace.steps):
            if len(le.refinements) < n_rules:
                warnings.warn(
                    "draw_refinements=True requested, but refinement log "
                    "for rule #{} incomplete. Using draw_refinements=False."
                    .format(index))
                draw_refinements = False
                break

    P0 = trace.steps[0].P
    N0 = trace.steps[0].N
    rnd_style = dict(color='grey', alpha=0.5, linestyle='dotted')
    refinements_style = dict(marker='.', markersize=1, linestyle='',
                             zorder=-1,)

    if theory_figure is None:
        theory_figure = plt.figure()
    theory_axes = theory_figure.gca(xlabel='n', ylabel='p',
                                    xlim=(0, trace.N_total),
                                    ylim=(0, trace.P_total))  # type: axes.Axes
    theory_axes.locator_params(integer=True)
    # draw "random theory" reference marker
    theory_axes.plot([0, N0], [0, P0], **rnd_style)
    # mark difference between PN_total and PN (if growing set size > total X)
    max_PN = np.sum([step.ancestors[0] for step in trace.steps], axis=0)
    # theory_axes.axhspan(max(P0, max_PN[P]), trace.P_total, color='grey', alpha=0.3)
    # theory_axes.axvspan(max(N0, max_PN[N]), trace.N_total, color='grey', alpha=0.3)
    _draw_outer_border(theory_axes, max(N0, max_PN[N]), trace.N_total,
                       max(P0, max_PN[P]), trace.P_total,
                       color='grey', alpha=0.3)

    if rules_use_subfigures:
        if rules_figure is None:
            rules_figure = plt.figure(figsize=(10.24, 10.24),
                                      tight_layout=True)
        # else assume rules_figure is already a figure
        if title is not None:
            rules_figure.suptitle("%s: Rules" % title, y=0.02)
        # TODO: axis labels when using subfigures
        subfigure_cols = math.ceil(np.sqrt(n_rules))
        subfigure_rows = math.ceil(n_rules / subfigure_cols)
        rule_axes = [rules_figure.add_subplot(subfigure_rows, subfigure_cols,
                                              rule_idx + 1)
                     for rule_idx in range(n_rules)]  # type: List[axes.Axes]
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
    for rule_idx, trace_entry in enumerate(trace.steps):
        rule_trace = trace_entry.ancestors
        refinements = trace_entry.refinements
        Pi = trace_entry.P
        Ni = trace_entry.N
        # TODO: visualize whether refinement was rejected due to inner_stop
        refts_mask = slice(None)  # all
        if draw_refinements == 'nonzero':
            refts_mask: slice = refinements[:, P] != 0
        mark_stop = trace.last_rule_stop and (rule_idx == n_rules - 1)

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
        rule_axis.plot([0, trace.N_total], [0, trace.P_total], **rnd_style)
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
            spine.set_bounds(0, Ni if horizontal else Pi)

        class PositiveTicks(AutoLocator):
            def tick_values(self, vmin, vmax):
                orig = super().tick_values(vmin, vmax)
                return orig[orig >= 0]

        rule_axis.xaxis.set_major_locator(PositiveTicks())
        rule_axis.yaxis.set_major_locator(PositiveTicks())
        rule_axis.locator_params(integer=True)

        # set reference frame (N,P), but move (0,0) so it looks comparable
        rule_axis.set_xbound(Ni - N0, Ni)
        rule_axis.set_ybound(Pi - P0, Pi)

    if title is not None:
        theory_axes.set_title("%s: Theory" % title)
    theory_figure.legend(title="rule: (p,n)", loc='center right')

    return theory_figure, rules_figure
