"""Implementation of SeCo / Covering algorithm.

Limitations / Assumptions
=====

Could be considered as TODO.

- at most two tests (see "operator set" below) per feature and rule
- no sparse input
- no missing values
- binary problems are always solved as concept learning, i.e. all rules are
  learned to identify a "positive class" and only the default rule classifies
  as "negative class".
- implicit default rule, classifies no rule from the theory matched. I.e. the
  classifier cannot abstain from making a prediction.
- limited operator set:
    - for categorical only ==
    - for numerical only <= and >=
- numerical (i.e. non-categorical) features always assumed to be ordinal
- only float data supported (due to usage of np.inf in Rules)
- no NaN, inf, or -inf values in data
- no weighting
- classification only, no regression
"""

__all__ = ['abstract', 'common', 'concrete', 'extra', 'tests', 'util']
