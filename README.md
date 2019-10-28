# sklearn-SeCo

Implementation of the *Separate and Conquer* or *Covering*-Algorithm for [scikit-learn](http://scikit-learn.org).
This is a classifier learning a collection (*theory*) of human-comprehensible *rule*s.

sklearn_seco aims to be
- computationally fast,
- open to algorithm modification (think e.g. new heuristics),
- yet still comprising of understandable code.

It was developed as a masters thesis at the
[*Knowledge Engineering Group* at TU Darmstadt](https://www.ke.tu-darmstadt.de/),
under supervision of Prof. Johannes Fürnkranz. Check the thesis document (see the [releases page](https://github.com/azrdev/sklearn-seco/releases)) for
additional documentation, including evaluation plots.


## Testing / Evaluation

For current test suite results, check
[Continuous Integration](https://travis-ci.com/azrdev/sklearn-seco).

To run a comparison of `sklearn_seco.RipperEstimator` with
[weka.JRip](http://weka.sourceforge.net/doc.stable/weka/classifiers/rules/JRip.html),
[weka.J48](http://weka.sourceforge.net/doc.stable/weka/classifiers/trees/J48.html), and
[sklearn.dtree](https://scikit-learn.org/stable/modules/tree.html)
on a selected collection of UCI datasets, run `python3 evaluation.py`.


## Installation

Required are python >= 3.6 and the packages defined in `setup.py`.
If needed, a known-working list of versioned dependencies is pinned in `requirements.txt`.

1. If you want to work on the code, check out the repository:

    ~~~sh
    git clone https://github.com/azrdev/sklearn-seco
    ~~~

    and run directly from the working tree, or make an editable install:

    ~~~sh
    pip install --update -e "/path/to/sklearn-seco[numba]"
    ~~~

2. If you want to use the project without modifying it, install the latest `master` with

    ~~~sh
    pip install --update "git+https://github.com/azrdev/sklearn-seco#egg=sklearn_seco[numba]"
    ~~~

3. TODO: publish on pypi so getting a stable version with `pip install "sklearn-seco[numba]"` is possible.


Note that for the speed-optimized matching you need to install `numba`,
and for the coverage plots in `extra.py` and `tests/test_extra.py` you need `matplotlib`.
The former is installed above, through specified
["extra"](https://packaging.python.org/tutorials/installing-packages/#installing-setuptools-extras)
dependency sets "numba" and "tests".


## Development status

- abstract seco is implemented and usable, the compatibility test
  `sklearn.utils.estimator_checks.check_estimator()`
  succeeds completely for SimpleSeCo and CN2

- CN2 has not been thoroughly compared to
  [original code](https://www.cs.utexas.edu/users/pclark/software/) or
  [the Orange implementation](https://orange3.readthedocs.io/projects/orange-visual-programming/widgets/model/cn2ruleinduction.html),
  but should be complete

- Ripper lacks the original class binarization strategy and the global post-optimization,
  therefore results are not identical to JRip (the only other freely available implementation).

- testsuite has a few consistent failures (all in [test_concrete.py](sklearn_seco/tests/test_concrete.py)):

    - `test_perfectly_correlated_categories_multiclass[CN2Estimator]`
      fails because [CN2 learns only one rule when it gets to `x=[[… 2 0] [… 0 1]], y=[2 1]`
      ](https://github.com/azrdev/sklearn-seco/blob/master/sklearn_seco/tests/conftest.py#L87..L89)
      which is perfectly valid behavior.

    - `test_sklearn_check_estimator[IrepEstimator]`
      fails `sklearn.utils.estimator_checks.check_classifier_data_not_an_array`
      (NOTE: maybe later tests in check_estimator fail, too) because:
      if default rule is better than any refinement and survives
      rule_stopping_criterion, the learned theory is `[init_rule()]`.
      [We consider this an error and raise an exception
      ](https://github.com/azrdev/sklearn-seco/blob/master/sklearn_seco/abstract.py#L161..L168),
      maybe stopping criteria are buggy that they let the default rule through.

    - `test_sklearn_check_estimator[RipperEstimator]`
      fails `sklearn.utils.estimator_checks.check_classifiers_train` due to <https://github.com/scikit-learn/scikit-learn/issues/14124>
      (NOTE: maybe later tests in check_estimator fail, too)

    - pypy test runs timeout on Travis-CI after 10 minutes.

- various TODOs throughout the code mark missing details
  and/or ideas for improvement (of functionality or runtime performance)

