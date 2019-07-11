import functools
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from time import perf_counter_ns
from types import SimpleNamespace
from typing import Callable, List

import numpy as np
from sklearn import clone
from sklearn.compose import make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.datasets.openml import _DATA_FILE, _get_data_features, \
    _get_local_path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch

from sklearn_seco.concrete import RipperEstimator

CACHE_DIR = 'openml_cache/'
RESULT_DIR = 'evaluation/'
WEKA_CMD = ['java', '-cp', '/usr/share/java/weka/weka.jar', 'weka.Run']

_logfile_prefix = RESULT_DIR + datetime.now().isoformat()
logging.basicConfig(
    format='%(asctime)s:' + logging.BASIC_FORMAT,
    handlers=[logging.StreamHandler(),
              logging.FileHandler(_logfile_prefix + '_complete.log')])
logging.captureWarnings(True)  # warnings.*filter don't work with joblib.Parallel
logger = logging.getLogger('evaluation')
logger.setLevel(logging.DEBUG)
weka_stdout_logger = logger.getChild('weka_stdout')
weka_stderr_logger = logger.getChild('weka_stderr')
sksrip_theory_logger = logger.getChild('sklearn_seco_Ripper_theory')
result_logger: logging.Logger = logger.getChild('results')
result_logger.addHandler(logging.FileHandler(_logfile_prefix + '_results.log'))


class WEKA_REGEX(SimpleNamespace):
    """regular expressions to parse weka/JRip output

    defined in weka/src/main/java/weka/classifier/evaluation/Evaluation.java
    """
    FULL_MODEL = re.compile(
        r'^=== Classifier model \(full training set\) ===$', re.MULTILINE)
    JRIP_N_RULES = re.compile(r'^Number of Rules :\s*(?P<no>\d+)\s*$', re.MULTILINE)
    RUNTIME_TRAIN = re.compile(r'^Time taken to build model:\s*'
                               r'(?P<time>\d+(\.\d*)?)\s*seconds', re.MULTILINE)
    RUNTIME_CV = re.compile(r'^Time taken to perform cross-validation:\s*'
                            r'(?P<time>\d+(\.\d*)?)\s*seconds', re.MULTILINE)
    CV_AVG_METRICS = re.compile(
        r'^Weighted Avg.\s*(?P<values>[?\s,0-9]*[,0-9])\s*$', re.MULTILINE)


UCI_DATASETS = [  # tuples (name, id), ordered by n_features. # comment: n_samples,n_features
    ('haberman', 43),  # 306,4
    # ('hayes-roth', 329),  # 160,5
    ('balance-scale', 11),  # 625,5
    ('iris', 61),  # 150,5
    ('tae', 48),
    ('monks-problems-1', 333),
    ('monks-problems-2', 334),
    ('monks-problems-3', 335),
    ('liver-disorders', 8),  # 345,7
    # ('car', 40975),  # 1728,7
    # ('mfeat-morphological', 962),  # 2000,7
    # ('shuttle-landing-control', 172),  # 15,7
    ('ecoli', 39),
    ('diabetes', 37),
    ('nursery', 26),
    ('postoperative-patient-data', 34),
    ('breast-cancer', 13),
    ('breast-w', 15),
    ('cmc', 23),  # 1473,10
    ('glass', 41),  # 214,10
    ('tic-tac-toe', 50),  # 958,10
    ('page-blocks', 30),  # 5073,11
    # ('bridges', 327),  # 105,13
    # ('solar-flare_1', 40686),  # 315,13
    # ('solar-flare_2', 40687),  # 1066,13
    # ('vowel', 307),  # 990,13
    ('heart-c', 49),  # 303,14
    # ('wine', 187),  # 178,14
    ('heart-h', 51),
    ('heart-statlog', 53),
    ('credit-a', 29),
    ('labor', 4),  # 57,17
    ('letter', 6),
    ('pendigits', 32),  # 10992,17
    ('vote', 56),
    ('zoo', 62),  # 101,17
    # ('primary-tumor', 171),  # 339,18
    ('lymph', 10),  # 148,19
    ('vehicle', 54),
    ('hepatitis', 55),  # 155,20
    ('segment', 36),
    ('credit-g', 31),
    ('colic', 27),  # 368,23
    ('mushroom', 24),  # 8124,23
    # ('spect', 336),  # 267,23
    ('autos', 9),  # 205,26
    # ('flags', 285),  # 194,30
    ('hypothyroid', 57),  # 3772,30
    ('sick', 38),
    ('trains', 52),  # 10,33
    ('dermatology', 35),
    ('ionosphere', 59),
    ('soybean', 42),
    ('kr-vs-kp', 3),
    ('anneal', 2),  # 898,39
    # ('cylinder-bands', 6332),  # 540,40
    ('waveform-5000', 60),  # 5000,41
    # ('spectf', 1600),  # 267,45
    # ('sponge', 1001),  # 76,46
    ('mfeat-zernike', 22),  # 2000,48
    # ('lung-cancer', 163),  # 32,57
    ('spambase', 44),  # 4601,58
    ('molecular-biology_promoters', 164),  # 106,59
    ('sonar', 40),  # 208,61
    ('splice', 46),  # 3190,61
    ('mfeat-karhunen', 16),
    ('optdigits', 28),  # 5620,65
    ('audiology', 7),  # 226,70
    ('mfeat-fourier', 14),  # 2000,77
    # ('spectrometer', 313),  # 531,103
    ('mfeat-factors', 12),  # 2000,217
    ('mfeat-pixel', 20),  # 2000,241
    ('arrhythmia', 5),  # 452,280

    # ('kdd_JapaneseVowels', 375),
    # ('kdd_SyskillWebert-Bands', 380),
    # ('kdd_SyskillWebert-BioMedical', 374),
    # ('kdd_SyskillWebert-Goats', 379),
    # ('kdd_SyskillWebert-Sheep', 376),
    # ('kdd_UNIX_user_data', 373),
    # ('kdd_internet_usage', 372),
    # ('kdd_ipums_la_97-small', 382),  # 7019,61
    # ('kdd_ipums_la_98-small', 381),  # 7485,61
    # ('kdd_ipums_la_99-small', 378),  # 8844,61
    # ('kdd_synthetic_control', 377),
]


sklearn_cross_validate = functools.partial(
    cross_validate,
    return_train_score=True,
    n_jobs=-1,
    # todo: fix ↓
    # sklearn warns about undefined precision if classes are not
    # present in prediction. weka just ignores, so do we
    scoring=['f1_weighted', 'precision_weighted',
             'recall_weighted', 'accuracy',
             'balanced_accuracy'],
    error_score=np.nan,
    cv=10,
    return_estimator=True
)


def _categorical_mask(dataset: Bunch):
    return np.array([ft in dataset.categories
                     for ft in dataset.feature_names], dtype=bool)


def run_sklearn_cart(dataset: Bunch, log_results: Callable):
    """run the decision tree classifier from scikit-learn, which is an
    implementation of CART.

    NOTE this means as of sklearn 0.21 it doesn't know categorical attributes,
      and therefore we preprocess these OneHotEncoder.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = make_pipeline(
            # impute NaN values with mean
            SimpleImputer(strategy='constant'),
            # NOTE: other strategies drop columns, breaking the column mask below
            # OneHotEncode categorical features
            make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                     _categorical_mask(dataset)),
                                    remainder='passthrough'),
            DecisionTreeClassifier(random_state=0,
                                   max_depth=np.log(len(dataset.target))),
            memory=tmpdir)  # cache the preprocessing

        logger.info("sklearn_seco: cross-validate dtree")
        # start_time = perf_counter_ns()
        cv_result = sklearn_cross_validate(estimator=pipeline,
                                           X=dataset.data,
                                           y=dataset.target)
        # runtime_cv = perf_counter_ns() - start_time
        logger.debug(cv_result)

    log_results(algorithm='sklearn.dtree',
                runtime_single=None,
                runtime_cv=cv_result['fit_time'].sum(), # runtime_cv / 10 ** 9,
                n_rules=None,
                metrics=[
                    cv_result['train_precision_weighted'].mean(),
                    cv_result['train_recall_weighted'].mean(),
                    cv_result['train_f1_weighted'].mean(),
                ])


def run_sklearn_seco_ripper(dataset: Bunch, log_results: Callable):
    """prepare & run sklearn_seco"""
    categorical = _categorical_mask(dataset)
    estimator = RipperEstimator()
    logger.info("sklearn_seco: build single Ripper")
    try:
        simple_rip: RipperEstimator = clone(estimator)
        start_time = perf_counter_ns()
        simple_rip.fit(
            dataset.data, dataset.target,
            categorical_features=categorical)
        runtime_single = (perf_counter_ns() - start_time) / 10 ** 9
        sksrip_theory_logger.info(
            [base.export_text(dataset.feature_names)
             for base in simple_rip.get_seco_estimators()])
    except ValueError as e:
        logger.info("sklearn_seco: single Ripper failed with " + str(e))
        simple_rip = runtime_single = None

    logger.info("sklearn_seco: cross-validate Ripper")
    # start_time = perf_counter_ns()
    cv_result = sklearn_cross_validate(
        estimator=estimator, X=dataset.data, y=dataset.target,
        fit_params=dict(categorical_features=categorical))
    # runtime_cv = perf_counter_ns() - start_time
    logger.debug(cv_result)

    log_results(algorithm='sklearn_seco.Ripper',
                runtime_single=runtime_single,
                runtime_cv=cv_result['fit_time'].sum(),  # runtime_cv / 10 ** 9,
                n_rules='+'.join([str(len(e.theory_))
                                  for e in simple_rip.get_seco_estimators()])
                        if simple_rip else None,
                metrics=[
                    cv_result['train_precision_weighted'].mean(),
                    cv_result['train_recall_weighted'].mean(),
                    cv_result['train_f1_weighted'].mean(),
                ])


def run_weka_JRip(*args):
    return _run_weka('JRip',
                     ['-O', '0',  # no global optimization (sklearn_seco doesn't have it yet)
                      ],
                     *args)


def run_weka_J48(*args):
    return _run_weka('J48', [], *args)


def _run_weka(name: str, extra_args: List[str],
              dataset: Bunch, log_results: Callable, cache_dir: str):
    logger.info('weka: prepare dataset')
    arffgz_path = _get_local_path(
        _DATA_FILE.format(dataset.details['file_id']),  # maybe != ds_id
        data_home=cache_dir + '/openml/')
    with tempfile.TemporaryDirectory() as tmpdirname:
        # link to a file named '*.arff.gz', otherwise weka doesn't
        # recognize it's gzipped
        arffgz_linked = (tmpdirname + '/'
                         + os.path.basename(arffgz_path)
                         + '.arff.gz')
        os.symlink(arffgz_path, arffgz_linked)

        # get target column
        target_column_ids = [int(feature['index'])
                             for feature in _get_data_features(
                                dataset.details['id'], data_home=cache_dir)
                             if feature['is_target'] == 'true']
        if len(target_column_ids) != 1:
            logger.warning("identified {} target columns [{}], using only {}"
                           .format(len(target_column_ids),
                                   ','.join(map(str, target_column_ids)),
                                   target_column_ids[0]))

        logger.info('run weka.' + name)
        _cmd = WEKA_CMD + [name,
                           '-t', arffgz_linked,  # "train" ARFF file
                           '-c', str(target_column_ids[0] + 1),
                           # todo: hardcode metrics
                           ] + extra_args
        logger.debug(_cmd)
        weka_process = subprocess.run(_cmd, capture_output=True, text=True)
    logger.info('weka.{} returned with {}'
                .format(name, weka_process.returncode))
    if weka_process.stdout:
        weka_stdout_logger.debug(weka_process.stdout)
    if weka_process.stderr:
        weka_stderr_logger.debug(weka_process.stderr)

    weka_out: str = weka_process.stdout
    if WEKA_REGEX.FULL_MODEL.search(weka_out):  # if has successful run
        cv_start = weka_out.find('cross-validation')
        jrip_n_rules = WEKA_REGEX.JRIP_N_RULES.search(weka_out)
        if jrip_n_rules:
            jrip_n_rules = int(jrip_n_rules.group('no'))
        weka_metrics = [
            float('nan') if val == '?' else float(val.replace(',', '.'))
            for val in WEKA_REGEX.CV_AVG_METRICS
                                 .search(weka_out, cv_start)
                                 .group('values')
                                 .split()]
        tp_rate, fp_rate, precision, recall, f1, mcc, roc_a, prc_a = \
            weka_metrics
        log_results(
            algorithm='weka.' + name,
            runtime_single=float(WEKA_REGEX.RUNTIME_TRAIN.search(weka_out)
                                 .group('time')),
            runtime_cv=float(WEKA_REGEX.RUNTIME_CV.search(weka_out)
                             .group('time')),
            n_rules=jrip_n_rules,
            metrics=[precision, recall, f1])
    else:
        logger.warning('weka.{}: no successful output found')


def _log_results(algorithm: str,
                 dataset: Bunch,
                 runtime_single: float,
                 runtime_cv: float,
                 n_rules: int or List[int],  # list of theories barely comparable
                 metrics: List[float]):
    """Record the results of the specified algorithm on the current
    dataset.

    :param algorithm: algorithm name.
    :param runtime_single: seconds needed to fit one instance of the
        classifier on the whole training set
    :param runtime_cv: seconds needed for the whole ten-fold
        cross-validation, split, fit and predict.
    :param n_rules: number of rules learned on the whole training
        set
    :param metrics: todo split parameter
    """
    n_samples, n_features = dataset.data.shape
    def formatfloat(f):
        if isinstance(f, float):
            return  '{:.3f}'.format(f)
        return f

    result_logger.info(','.join(map(
        lambda v: '' if v is None else str(v),
        [str(dataset.details['id']) + '_' + dataset.details['name'],
         n_samples,
         n_features,
         len(dataset.categories),
         algorithm,
         n_rules,
         formatfloat(runtime_single),
         formatfloat(runtime_cv),
         ] + [formatfloat(m) for m in metrics])))


def main(args: List[str]):
    result_logger.info(','.join(["dataset",
                                 "n_samples",
                                 "n_features",
                                 "n_categorical_features",
                                 "algorithm",
                                 "n_rules",
                                 "runtime_single",
                                 "runtime_cv",
                                 "precision",
                                 "recall",
                                 "f1"]))
    skip_until = None
    if args[1:]:
        skip_until = int(args[1])
        logger.info('got command line args, skipping datasets up to #{}'
                    .format(skip_until))
    cache_dir = os.path.realpath(CACHE_DIR)
    for ds_name, ds_id in UCI_DATASETS:
        if skip_until is not None:
            if ds_id == skip_until:
                skip_until = None
            continue

        try:
            dataset = fetch_openml(data_id=ds_id, data_home=cache_dir)
            logger.info('dataset #{}: {}'.format(dataset.details['id'],
                                                 dataset.details['name']))
        except Exception as e:
            logger.error('dataset #{} {} not found: {!r}'
                         .format(ds_id, ds_name, e))
            continue

        log_results = functools.partial(_log_results, dataset=dataset)

        try:
            run_sklearn_cart(dataset, log_results)
        except Exception as e:
            logger.warning('sklearn.dtree failed on dataset #{} {} '
                           'with {!r}'.format(ds_id, ds_name, e))

        try:
            run_sklearn_seco_ripper(dataset, log_results)
        except Exception as e:
            logger.warning('sklearn_seco.Ripper failed on dataset #{} {} '
                           'with {!r}'.format(ds_id, ds_name, e))

        try:
            run_weka_JRip(dataset, log_results, cache_dir)
        except Exception as e:
            logger.warning('weka.JRip failed on dataset #{} {} '
                           'with {!r}'.format(ds_id, ds_name, e))

        try:
            run_weka_J48(dataset, log_results, cache_dir)
        except Exception as e:
            logger.warning('weka.J48 failed on dataset #{} {} '
                           'with {!r}'.format(ds_id, ds_name, e))


if __name__ == '__main__':
    main(sys.argv)
