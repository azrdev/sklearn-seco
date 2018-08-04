"""Implementation of SeCo / Covering algorithm.

Limitations / Assumptions (TODO)
=====

- at most two tests per feature and rule
- no sparse input
- no missing values
- binary estimator, applies binarization to multi-class problems
- first class (from `sklearn.utils.unique_labels()`, i.e. the lowest class)
    always assumed to be positive (may be asymmetrical, because of default rule)
- implicit default rule
- only ordered rule list (no unordered rule set / tree)
- limited operator set:
    - for categorical only ==
    - for numerical only <= and >=
- numerical features always assumed to be ordinal
- no NaN, inf, or -inf values in data
- no weighting
"""
