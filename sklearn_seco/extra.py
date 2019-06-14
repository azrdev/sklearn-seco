"""
Implementation of SeCo / Covering algorithm:
Helpers in addition to the algorithms in `concrete.py`.
"""

import json
import math
import warnings
from typing import Optional, Union, Sequence, Tuple, Type, Callable, \
    NamedTuple, MutableSequence, List, Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
from matplotlib.ticker import AutoLocator
from matplotlib.figure import Figure  # needed only for type hints
from matplotlib.patches import Polygon
from typing.io import IO

from sklearn_seco.abstract import SeCoEstimator
from sklearn_seco.common import \
    AugmentedRule, Theory, \
    AbstractSecoImplementation, RuleContext, TheoryContext
from sklearn_seco.concrete import GrowPruneSplitRuleContext


class Trace:
    """Trace of a seco algorithm run, i.e. `abstract_seco` invocation.

    Attributes
    -----
    - `steps`: Sequence[Trace.Step]
      Each item represents one step in `abstract_seco`, and thus one rule
      in the learned theory. The last step/rule is only part of the theory iff
      last_rule_stop is False.

    - `last_rule_stop`: boolean
      result of `rule_stopping_criterion` on the last found `best_rule`. If
      False, it is part of the theory (and the rule search ended because all
      positive examples were covered), if True it is not part of the theory
      (the search ended because `rule_stopping_criterion` was True).
    """

    _JSON_DUMP_DESCRIPTION = "sklearn_seco.extra.trace_coverage dump"
    _JSON_DUMP_VERSION = 2

    steps: MutableSequence['Trace.Step']
    last_rule_stop: bool
    use_pruning: bool

    def __init__(self, use_pruning):
        self.steps = []
        self.last_rule_stop = None
        self.use_pruning = use_pruning

    def append_step(self,
                    context: RuleContext,
                    ancestors: Iterable[AugmentedRule],
                    refinements: Sequence['Trace.Step.Part.Refinement']):
        if self.use_pruning:
            assert isinstance(context, GrowPruneSplitRuleContext)
            self.steps.append(Trace.Step.pruning_step(context,
                                                      ancestors,
                                                      refinements))
        else:
            self.steps.append(Trace.Step.nonpruning_step(context,
                                                         ancestors,
                                                         refinements))

    class Step:
        """Trace of one seco algorithm step, i.e. `find_best_rule` invocation.

        Consists of three `Part`s storing similar information for the whole
        step, its growing, and pruning phase, respectively. If the algorithm
        doesn't employ pruning, only the `total` part of the trace is used.
        """

        total: 'Trace.Step.Part'
        growing: 'Trace.Step.Part'

        @staticmethod
        def nonpruning_step(context: RuleContext,
                            ancestors: Iterable[AugmentedRule],
                            refinements: Sequence['Trace.Step.Part.Refinement']
                            ) -> 'Trace.Step':
            ancestors = list(ancestors)
            PN = context.PN(ancestors[0].head, force_complete_data=True)
            ancestors_counts = np.array(
                [rule.pn(context, force_complete_data=True)
                 for rule in ancestors])
            step = Trace.Step()
            step.total = Trace.Step.Part(ancestors_counts, refinements, *PN)
            return step

        @staticmethod
        def pruning_step(context: GrowPruneSplitRuleContext,
                         ancestors: Iterable[AugmentedRule],
                         refinements: Sequence['Trace.Step.Part.Refinement']
                         ) -> 'Trace.Step':
            ancestors = list(ancestors)
            step = Trace.Step.nonpruning_step(context, ancestors, [])

            context.growing = True
            ancestors_counts_growing = np.array([rule.pn(context)
                                                 for rule in ancestors])
            PN_growing = context.PN(ancestors[0].head)
            step.growing = Trace.Step.Part(ancestors_counts_growing,
                                           refinements, *PN_growing)

            return step

        def __eq__(self, other):
            if type(other) is type(self):
                return self.__dict__ == other.__dict__
            return NotImplemented

        @staticmethod
        def from_json(dec: Dict):
            step = Trace.Step()
            step.total = Trace.Step.Part.from_json(dec['total'])
            if 'growing' in dec:
                step.growing = Trace.Step.Part.from_json(dec['growing'])
            if 'pruning' in dec:
                step.pruning = Trace.Step.Part.from_json(dec['pruning'])
            return step

        class Part:
            """Trace of one part of a seco algorithm `Step`.

            Attributes
            -----
            `ancestors`: list of np.array with shape (n_ancestors, 2)
               This is the log of the learned `best_rule` and its `ancestors`.

               An array with (p, n) for the rule and each ancestor, in reverse
               order of creation.

               If `TraceCoverageTheoryContext.trace_level` is 'theory', no
               ancestors are traced, so only the `best_rule` is contained.

            `refinements`: np.array or list of np.array with shape (n_refinements, 3)
               This is the log of all `refinements`, an array with (p, n, stop)
               for each refinement, where `stop` is the boolean result of
               `inner_stopping_criterion(refinement)`.

               Empty if `TraceCoverageTheoryContext.trace_level` is not
               'refinements'.

            P: int
              Count of positive examples.

            N: int
              Count of negative examples.
            """
            Ancestor = NamedTuple('Ancestor', [('p', int), ('n', int)])
            Refinement = NamedTuple('Refinement', [('p', int),
                                                   ('n', int),
                                                   ('stop', bool)])

            ancestors: Union[Sequence[Ancestor], np.ndarray]
            refinements: Union[MutableSequence[Refinement], np.ndarray]
            P: int
            N: int

            def __init__(self, ancestors: Sequence[Ancestor],
                         refinements, P, N):
                self.ancestors = ancestors
                self.refinements = np.asanyarray(refinements)
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

            @staticmethod
            def from_json(dec: Dict) -> 'Trace.Step.Part':
                return Trace.Step.Part(dec['ancestors'], dec['refinements'],
                                       dec['P'], dec['N'])

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def plot_coverage_log(self, **kwargs):
        """Plot the trace, see :func:`plot_coverage_log`."""
        return plot_coverage_log(self, **kwargs)

    @staticmethod
    def _json_encoder(obj):
        """Serialize `obj` when used as `json.JSONEncoder.default` method."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (Trace.Step, Trace.Step.Part)):
            return obj.__dict__
        raise TypeError

    def to_json(self):
        """:return: A string containing a JSON representation of the trace."""
        return json.dumps({
            "description": Trace._JSON_DUMP_DESCRIPTION,
            "version": Trace._JSON_DUMP_VERSION,
            "steps": self.steps,
            "last_rule_stop": self.last_rule_stop,
            "use_pruning": self.use_pruning,
        }, allow_nan=False, default=Trace._json_encoder)

    @staticmethod
    def from_json(dump: Union[str, IO]) -> 'Trace':
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
        trace = Trace(dec['use_pruning'])
        trace.steps = [Trace.Step.from_json(step)
                       for step in dec['steps']]
        trace.last_rule_stop = dec['last_rule_stop']
        trace.use_pruning = dec['use_pruning']
        return trace


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
    `_BaseSeCoEstimator`, each of which collects the trace & submits it
    separately. Parallel execution of these has to be synchronized manually,
    this has not been tested.

    The trace gets the values P,N,p,n from the `RuleContext` and
    `AugmentedRule` objects; if `GrowPruneSplit` is used, this usually means
    that these are only computed on the growing set.

    # TODO: implement tracing of estimator instances (not only classes)

    Usage
    =====
    Define the callback function to receive the trace & display it:

    >>> def callback(trace):
    ...     theory_figure, rules_figure = plot_coverage_log(trace)
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
        use_pruning = issubclass(self.algorithm_config.RuleContextClass,
                                 GrowPruneSplitRuleContext)
        self.trace: Trace = Trace(use_pruning)


class TraceCoverageRuleContext(RuleContext):
    """Tracing `RuleContext`."""
    def __init__(self, theory_context: TheoryContext, X, y):
        super().__init__(theory_context, X, y)
        self.growing_refinements = []


class TraceCoverageImplementation(AbstractSecoImplementation):
    """Tracing `AbstractSecoImplementation` mixin.

    Note: This class always has to come first in the mro/inheritance tree,
    because it overrides some methods not to be overridden. Use the
    `trace_coverage` decorator to ensure that.
    """

    direct_multiclass_support = False

    @classmethod
    def inner_stopping_criterion(cls, refinement: AugmentedRule,
                                 context: TraceCoverageRuleContext) -> bool:
        stop = super().inner_stopping_criterion(refinement, context)
        assert isinstance(context, TraceCoverageRuleContext)
        p, n = refinement.pn(context)
        context.growing_refinements.append(np.array((p, n, stop)))
        return stop

    @classmethod
    def rule_stopping_criterion(cls, theory: Theory, rule: AugmentedRule,
                                context: TraceCoverageRuleContext) -> bool:
        last_rule_stop = super().rule_stopping_criterion(theory, rule, context)

        tctx = context.theory_context
        assert isinstance(tctx, TraceCoverageTheoryContext)
        assert isinstance(context, TraceCoverageRuleContext)

        if tctx.trace_level == 'best_rules':
            ancestors = [rule]
        else:  # elif trace_level in ('ancestors', 'refinements'):
            ancestors = rule.ancestors()

        tctx.trace.append_step(context, ancestors, context.growing_refinements)
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
        rules_move_axes: bool = True,
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
    :param rules_move_axes: If `True`, the rules plots have their axis (or
      the (0,0) point) moved so it reflects the (P, N) left over by the
      previous rule.
    :return: `(theory_figure, rules_figure)` where rules_figure is a figure
      or a list of figure, depending on `rules_use_subfigure`.
    """

    P, N = 0, 1  # readable indexes, not values!

    n_rules = len(trace.steps)
    if not n_rules:
        # issue a warning, user can decide handling. See module `warnings`
        warnings.warn("Empty trace collected, useless plot.")
    if draw_refinements:
        if trace.use_pruning:
            warnings.warn("draw_refinements requested, but algorithm uses "
                          "pruning, so ignore for theory plot")
        parts = ["growing"] if trace.use_pruning else ["total"]
        for index, step in enumerate(trace.steps):
            for part_name in parts:
                part = getattr(step, part_name)
                if part.refinements.ndim != 2:
                    warnings.warn(
                        "draw_refinements=True requested, but refinement log "
                        "for rule[{}].{} incomplete (shape {}). Using "
                        "draw_refinements=False."
                        .format(index, part_name, part.refinements.shape))
                    draw_refinements = False
                    break

    P0_total = P0_growing = trace.steps[0].total.P
    N0_total = N0_growing = trace.steps[0].total.N
    if trace.use_pruning:
        P0_growing = trace.steps[0].growing.P
        N0_growing = trace.steps[0].growing.N
    rnd_style = dict(color='grey', alpha=0.5, linestyle='dotted')
    refinements_style = dict(marker='.', markersize=1, linestyle='',
                             zorder=-2,)

    if theory_figure is None:
        theory_figure = plt.figure()
    theory_axes = theory_figure.gca(xlabel='n', ylabel='p',
                                    xlim=(0, N0_total),
                                    ylim=(0, P0_total))  # type: axes.Axes
    theory_axes.locator_params(integer=True)
    # draw "random theory" reference marker
    theory_axes.plot([0, N0_total], [0, P0_total], **rnd_style)

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
    for rule_idx, trace_step in enumerate(trace.steps):
        # TODO: visualize whether refinement was rejected due to inner_stop
        mark_stop = trace.last_rule_stop and (rule_idx == n_rules - 1)

        # *** this rule in theory plot ***
        rule = trace_step.total.ancestors[0] + previous_rule
        theory_line = theory_axes.plot(
            rule[N], rule[P], 'x' if mark_stop else '.',
            label="{i:{i_width}}: ({p:4}, {n:4})"
                  .format(n=rule[N], p=rule[P], i=rule_idx,
                          i_width=math.ceil(np.log10(n_rules))))
        rule_color = theory_line[0].get_color()
        if draw_refinements and not trace.use_pruning:
            # draw refinements in theory plot
            refinements = trace_step.total.refinements
            mask: slice = refinements[:, P] != 0 \
                if draw_refinements == 'nonzero' else slice(None)
            theory_axes.plot(refinements[mask, N] + previous_rule[N],
                             refinements[mask, P] + previous_rule[P],
                             color=rule_color, alpha=0.3,
                             **refinements_style)
        # draw arrows between best_rules
        # NOTE: invert coordinates because we save (p,n) and plot (x=n,y=p)
        theory_axes.annotate("", xytext=previous_rule[::-1], xy=rule[::-1],
                             arrowprops={'arrowstyle': "->"}, zorder=-1)
        previous_rule = rule

        # *** growing subplot with ancestors of current rule ***
        rule_axis = rule_axes[rule_idx]
        grow_part: Trace.Step.Part = trace_step.growing \
            if trace.use_pruning else trace_step.total
        if mark_stop:
            rule_title_template = 'growing (Rule #%d) Candidate'
        else:
            rule_title_template = 'growing Rule #%d'
        if not rules_use_subfigures and title is not None:
            # add title before each of the figures
            rule_title_template = "%s: %s" % (title, rule_title_template)
        rule_axis.set_title(rule_title_template % rule_idx)
        # draw "random theory" reference marker
        rule_axis.plot([0, grow_part.N], [0, grow_part.P], **rnd_style)
        # draw rule_trace
        rule_axis.plot(grow_part.ancestors[:, N], grow_part.ancestors[:, P],
                       'o-', color=rule_color)
        if draw_refinements:
            # draw refinements as scattered dots
            mask: slice = (grow_part.refinements[:, P] != 0
                           if draw_refinements == 'nonzero'
                           else slice(None))
            rule_axis.plot(grow_part.refinements[mask, N],
                           grow_part.refinements[mask, P],
                           color='black', alpha=0.7, **refinements_style)

        # draw x and y axes through (0,0) and hide for negative values
        for spine_type, spine in rule_axis.spines.items():
            spine.set_position('zero')
            horizontal = spine_type in {'bottom', 'top'}
            spine.set_bounds(0, grow_part.N if horizontal else grow_part.P)

        class PositiveTicks(AutoLocator):
            def tick_values(self, vmin, vmax):
                orig = super().tick_values(vmin, vmax)
                return orig[orig >= 0]

        rule_axis.xaxis.set_major_locator(PositiveTicks())
        rule_axis.yaxis.set_major_locator(PositiveTicks())
        rule_axis.locator_params(integer=True)

        _draw_outer_border(rule_axis,
                           grow_part.N, N0_growing, grow_part.P, P0_growing,
                           color='grey', alpha=0.1)
        if rules_move_axes:
            # set reference frame (N,P), but move (0,0) so it looks comparable
            rule_axis.set_xbound(grow_part.N - N0_growing, grow_part.N)
            rule_axis.set_ybound(grow_part.P - P0_growing, grow_part.P)
        else:
            rule_axis.set_xbound(0, N0_growing)
            rule_axis.set_ybound(0, P0_growing)

    if title is not None:
        theory_axes.set_title("%s: Theory" % title)
    theory_figure.legend(title="rule: (p,n)", loc='center right')

    return theory_figure, rules_figure
