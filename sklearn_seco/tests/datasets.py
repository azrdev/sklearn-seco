"""Artificial dataset (generator functions) for the sklearn_seco unittests."""
# TODO: maybe replace these with fixtures used with indirect=True

import itertools
from os.path import dirname
from typing import Union

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.externals import _arff as arff
from sklearn.utils import check_random_state, Bunch

from sklearn_seco.util import build_categorical_mask


class Dataset(Bunch):
    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray = None,
                 y_test: np.ndarray = None,
                 categorical_features: Union[None, str, np.ndarray] = None,
                 **kwargs):
        if x_test is not None:
            kwargs["x_test"] = x_test
        if y_test is not None:
            kwargs["y_test"] = y_test
        super().__init__(x_train=x_train, y_train=y_train,
                         categorical_features=categorical_features,
                         **kwargs)

    def get_opt(self, key):
        """:return: `self.get(key, default=None)`"""
        return self.get(key, None)

    def to_arff(self,
                name: str,
                add_test_data: bool = False,
                description: str = None) -> str:
        """Convert the dataset to wekas ARFF format

        :param name: str.
            The relation name
        :param add_test_data: bool.
            Iff true, include `x_test` and `y_test` in addition to `x_train`
            and `y_train`.
        :param description: str, optional.
            The description, inserted as comment into the file.
        :return: str.
            The ARFF file contents, usable with `open(...).write()`.
        """
        n_features = self.x_train.shape[1]
        categorical = build_categorical_mask(self.categorical_features,
                                             n_features)
        feature_names = self.get_opt('feature_names') or \
                        ['ft_{i}_{cat}'
                         .format(i=i, cat='cat' if categorical[i] else 'num')
                         for i in range(n_features)]
        data = self.x_train
        classes = self.y_train
        if add_test_data:
            data = np.vstack((data, self.x_test))
            classes = np.vstack((classes, self.y_test))
        # convert -0.0 to 0.0 to avoid having -0.0 samples but not @attribute
        if np.issubdtype(data.dtype, np.floating):
            data += 0
        if np.issubdtype(classes.dtype, np.floating):
            classes += 0

        # TODO: general transformers for each column, to convert floats to e.g. ints for categorical

        def distinct_values(values):
            return [str(v) for v in np.unique(values)]
        value_sets = {i: distinct_values(data[:, i])
                      for i in range(n_features)
                      if categorical[i]}
        obj = {'relation': name,
               'description': description,
               'attributes': [(feature_names[i],
                               value_sets[i] if categorical[i] else 'NUMERIC')
                              for i in range(n_features)] +
                             [('class', distinct_values(classes))],
               'data': [data_line.tolist() + [str(class_)]
                        for data_line, class_ in zip(data, classes)]}
        return arff.dumps(obj)


# datasets


# TODO: ripper learns 0 rules (for at least one class in ovr)
def perfectly_correlated_multiclass(n_features=10):
    """Generate multiclass problem with n features each matching one class."""

    # revert, so default class = lexicographically bigger results in good rules
    y = np.arange(1, n_features + 1)[::-1]
    x = np.eye(n_features, dtype=int) * y
    return Dataset(x, y, categorical_features='all')


def xor_2d(n_samples=400, random=None):
    """Generate numeric low-noise 2D binary XOR problem"""
    if random is None:
        random = check_random_state(11)
    n = n_samples // 4
    centers = itertools.product([0, 4], [0, 4])
    t = np.vstack([np.hstack((random.normal(loc=(x, y), size=(n, 2)),
                              [[99 + (x == y)]] * n))
                   for x, y in centers])
    random.shuffle(t)
    split = len(t) // 3 * 2
    x_train = t[:split, :-1]
    y_train = t[:split, -1]
    x_test = t[split:, :-1]
    y_test = t[split:, -1]
    return Dataset(x_train, y_train, x_test, y_test)


def checkerboard_2d(n_samples=100_000, binary=True, categorical=True,
                    random=None):
    """
    Generate huge 2D multi-class problem with interleaved class clusters (similar to XOR)

    :param binary: bool.
        If True, data has 2 classes of unequal size, otherwise 9 of equal size.
    :param categorical: bool.
        If True, features are few distinct integers, otherwise a float
        distribution.
    """
    if random is None:
        random = check_random_state(1)
    centers = itertools.product([0, 7, 15], [20, 30, 39])
    n = (n_samples * 2) // 10
    t = np.vstack([
        np.hstack((random.normal(loc=(x, y), size=(n, 2)), [[cls]] * n))
        for cls, (x, y) in enumerate(centers)])
    random.shuffle(t)
    split = len(t) // 3 * 2
    x_train = t[:split, :-1]
    y_train = t[:split, -1]
    x_test = t[split:, :-1]
    y_test = t[split:, -1]
    cm = None
    if binary:
        y_train %= 2
        y_test %= 2
    if categorical:
        x_train = np.rint(x_train)
        x_test = np.rint(x_test)
        cm = 'all'
    return Dataset(x_train, y_train, x_test, y_test, cm)


def binary_slight_overlap(
        n_samples=160,  # 80 samples for each class and each of (train, test)
        n_features=8,
        random=None):
    """
    Generate two equally sized, normal distributed, slightly overlapping
    classes.
    """
    if random is None:
        random = check_random_state(42)
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    raw = np.block([
        [random.multivariate_normal(mean, cov, size=n_samples),  # features
         np.ones((n_samples, 1))],  # positive class label
        [random.multivariate_normal(mean + 3, cov, size=n_samples),  # features
         np.zeros((n_samples, 1))]  # negative class label
    ])
    random.shuffle(raw)
    train, test = np.array_split(raw, 2, axis=0)
    X = train[:, :-1]  # all but last column
    y = train[:, -1]  # last column
    X_test = test[:, :-1]
    y_test = test[:, -1]
    return Dataset(X, y, X_test, y_test)


def sklearn_make_classification(n_samples=1000, n_features=2, n_classes=2,
                                random=1,
                                *, categorize=False, test_ratio=.3,
                                **kwargs):
    n_informative = max(2, n_features // 3 * 2)
    x, y = make_classification(n_samples, n_features,
                               n_informative=n_informative,
                               n_redundant=0,
                               n_classes=n_classes,
                               class_sep=1.5,
                               random_state=random,
                               **kwargs)
    cf = None
    if categorize:
        x = np.rint(x)
        cf = 'all'
    test_start_idx = int((1 - test_ratio) * n_samples)
    return Dataset(x_train=x[:test_start_idx], y_train=y[:test_start_idx],
                   x_test=x[test_start_idx:], y_test=y[test_start_idx:],
                   categorical_features=cf)


def sklearn_make_moons(n_samples=400,
                       random=42,
                       test_ratio=0.3):
    test_start_idx = int((1 - test_ratio) * n_samples)
    x, y = make_moons(n_samples, random_state=random)
    return Dataset(x[:test_start_idx], y[:test_start_idx],
                   x[test_start_idx:], y[test_start_idx:],
                   categorical_features=None)


# TODO: SimpleSeCo,CN2 learn very bad rules
def artificial_disjunction(n_samples=2_000, n_random_features=10,
                           test_ratio=.3, noise_ratio=.05,
                           random=42):
    """
    Binary `ab | ac | ade` on binary features. Imbalanced class distribution.

    See also (Cohen 1995).

    :param noise_ratio: float between 0 and 1
        Add this percentage of class noise, i.e. inverted label. Stratified,
        i.e. positive and negative class both have this percentage of noise.
    """
    random = check_random_state(random)
    n_features = 5 + n_random_features
    x = random.randint(2, size=(n_samples, n_features), dtype=bool)
    informative = random.permutation(n_features)[:5]
    a, b, c, d, e = x.T[informative]
    y = a * b + a * c + a * d * e
    y_classes, y_indices, y_counts = np.unique(y, return_inverse=True,
                                               return_counts=True)
    for c, count in zip(y_classes, y_counts):
        n_noise = int(noise_ratio * count)
        noise_idx = random.permutation(np.nonzero(y_indices == c)[0])[:n_noise]
        y[noise_idx] = np.invert(y[noise_idx])
    test_start_idx = int((1 - test_ratio) * n_samples)
    feature_names = ['ft_%d' % i for i in range(n_features)]
    for ft, fi in zip('abcde', informative):
        feature_names[fi] = ft
    return Dataset(x[:test_start_idx], y[:test_start_idx],
                   x[test_start_idx:], y[test_start_idx:],
                   categorical_features='all', feature_names=feature_names)


def staged():
    """low-noise binary 2d testcase for stopping criteria

    Has a big (~200) negative cluster and two different (100 and 10)
    positive clusters, each containing a single negative sample.
    """
    filename = dirname(__file__) + '/staged.arff'
    dec = arff.load(open(filename))
    data = np.array(dec['data'], dtype=float)
    return Dataset(data[:, :-1], data[:, -1],
                   feature_names=[ft[0] for ft in dec['attributes'][:-1]])
