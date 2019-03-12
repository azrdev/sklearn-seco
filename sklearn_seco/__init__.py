"""Implementation of SeCo / Covering algorithm.

Limitations / Assumptions (partly TODO)
=====

- at most two tests per feature and rule
- no sparse input
- no missing values
- binary estimator, applies binarization to multi-class problems
  TODO: multi-class rule learning (learn any rule, find class with max covered examples)
- implicit default rule
- TODO: only ordered rule list (no unordered rule set / tree)
- limited operator set:
    - for categorical only ==
    - for numerical only <= and >=
- numerical features always assumed to be ordinal
- only float data supported (due to usage of np.inf in Rules)
- no NaN, inf, or -inf values in data
- no weighting
- classification only, no regression
"""

from sklearn_seco import abstract, common, concrete, extra, tests

__all__ = ['abstract', 'common', 'concrete', 'extra', 'tests']
