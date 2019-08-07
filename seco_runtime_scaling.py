"""
Measure & plot runtime of our SeCo algorithm(s) with various sample and feature
counts.
"""

import sys
import time
import timeit
from itertools import chain
from types import MappingProxyType
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

import sklearn_seco.concrete


def time_seco(estimator: str, dataset_args: str) -> Optional[Sequence[float]]:
    setup = ';\n'.join((
        "import sklearn_seco; from sklearn_seco.tests import datasets",
        "dataset = sklearn_seco.tests.datasets.sklearn_make_classification(%s)"
            % dataset_args,
        "estimator = sklearn_seco.concrete.%s()" % estimator))
    stmt = "estimator.fit(dataset.x_train, dataset.y_train," \
           "  categorical_features=dataset.categorical_features)"
    timer = timeit.Timer(stmt, setup)
    try:
        ti_number, raw_autorange_timing = timer.autorange()
        raw_timings = timer.repeat(number=ti_number) + [raw_autorange_timing]
    except ValueError:
        return None
    return sorted(timing / ti_number for timing in raw_timings)


def n_sample_gen(max=np.inf) -> Iterable[int]:
    mg = 1
    while mg * 50 < max:
        for t in (10, 20, 50):
            yield t * mg
        mg *= 10


def timing_for_param(estimator: str, categorical: bool,
                     max_samples: int = np.inf) -> Iterable:
    extra_args = ['categorize=True'] if categorical else []
    for n_samples in n_sample_gen(max_samples):
        for n_features in chain(range(2, 16), range(20, 70, 16)):
            argstr = "n_samples=%d, n_features=%d, %s" \
                     % (n_samples, n_features, ','.join(extra_args))
            timings = time_seco(estimator, argstr)
            if timings:
                yield n_samples, n_features, timings


def plot_timings(timings, title=None, figure=None):
    from matplotlib.ticker import LogLocator, LogFormatter
    if figure is None:
        figure: plt.Figure = plt.figure()
    axes = figure.gca(xlabel='n_features', ylabel='time[s]')
    if title is not None:
        axes.set_title(title)
    n_samples = timings.T[0]
    n_features = timings.T[1]
    tm_min = timings.T[2]
    for n in np.unique(n_samples):
        mask = n_samples == n
        axes.loglog(n_features[mask], tm_min[mask], '.-', label=str(n))
    axes.legend(title='n_samples')
    axes.grid(True)
    figure.tight_layout()
    # show more ticks
    axes.xaxis.set_major_locator(LogLocator(subs='all'))
    axes.xaxis.set_major_formatter(LogFormatter(minor_thresholds=(100, 99)))
    axes.yaxis.set_major_locator(LogLocator(subs='all'))
    return figure


def log(message):
    print("%s %s" % (time.strftime('%Y-%m-%dT%H:%M:%S%z'), message),
          file=sys.stderr)


if __name__ == "__main__":
    categorical = 'c' in sys.argv[1:]

    estimator = sklearn_seco.concrete.RipperEstimator.__name__
    log("start timing of %s" % estimator)
    print("n_samples, n_features, timings...")
    all_timings = []
    try:
        for n_samples, n_features, timings in timing_for_param(estimator,
                                                               categorical):
            onelist = [n_samples, n_features] + timings
            all_timings.append(onelist)
            print('[' + ",".join([str(x) for x in onelist]) + '],')
    except KeyboardInterrupt:
        pass
    log("stop timing of %s, got %d timings" % (estimator, len(all_timings)))
    if all_timings:
        log("plotting")
        plot_timings(np.array(all_timings), 'runtime of %s' % estimator).show()

    input('Press any key to exit.')
