"""
MDL (minimum description length) calculation for RIPPER.

Mostly adapted from JRip.java, see `sklearn_seco.concrete.RipperMdlStop`.
Most parts are described in
- (Quinlan 1995) MDL and Categorical Theories (Continued)
- (Quinlan 1994) The Minimum Description Length Principle and Categorical Theories
- (Cohen 1995) Fast Effective Rule Induction
- (Cohen 1998) the RIPPER patent.
"""

import numpy as np

from sklearn_seco.common import AugmentedRule, log2


def subset_description_length(n, k, p) -> float:
    """:return: DL of a subset of `k` elements out of a known set of `n`
    elements, where `p` is known.

    NOTE: Named `RuleStats.subsetDL` in weka.JRip.
    """
    return -k * log2(p) - (n - k) * log2(1 - p)


def data_description_length(expected_fp_over_err, covered, uncovered, fp, fn
                            ) -> float:
    """:return DL of the data, i.e. the errors of a rule (false positives &
    negatives).

    NOTE: Named `RuleStats.dataDL` in weka.JRip, and defined as equation (3) in
    (Quinlan 1995).
    """
    S = subset_description_length
    total_bits = log2(covered + uncovered + 1)
    if covered > uncovered:
        expected_error = expected_fp_over_err * (fp + fn)
        covered_bits = S(covered, fp, expected_error / covered) \
            if covered > 0 else 0  # TODO: seems JRip never has covered==0
        uncovered_bits = S(uncovered, fn, fn / uncovered) \
            if uncovered > 0 else 0
    else:
        expected_error = (1 - expected_fp_over_err) * (fp + fn)
        covered_bits = S(covered, fp, fp / covered) \
            if covered > 0 else 0
        uncovered_bits = S(uncovered, fn, expected_error / uncovered) \
            if uncovered > 0 else 0  # TODO: seems JRip never has uncovered==0
    return total_bits + covered_bits + uncovered_bits


def rule_description_length(rule: AugmentedRule) -> float:
    """:return: DL of `rule`.

    NOTE: Named `RuleStats.theoryDL` in weka.JRip.
    """
    n_conditions = np.count_nonzero(np.isfinite(rule.conditions))  # no. of conditions
    max_n_conditions = rule.conditions.size  # possible no. of conditions
    # TODO: JRip counts all thresholds (RuleStats.numAllConditions())

    kbits = log2(n_conditions)  # no. of bits to send `n_conditions`
    if n_conditions > 1:
        kbits += 2 * log2(kbits)
    rule_dl = kbits + subset_description_length(
        max_n_conditions, n_conditions, n_conditions / max_n_conditions)
    return rule_dl * 0.5  # redundancy factor


def minDataDLIfExists(expected_fp_over_err, p, n, P, N, theory_pn) -> float:
    return data_description_length(
        expected_fp_over_err=expected_fp_over_err,
        covered=sum(th_p + th_n for th_p, th_n in theory_pn[1:]),  # of theory
        uncovered=P + N - p - n,  # of rule
        fp=sum(th_n for th_p, th_n in theory_pn[1:]),  # of theory
        fn=P - p,  # of rule
    )


def minDataDLIfDeleted(expected_fp_over_err, p, n, P, N, theory_pn) -> float:
    # covered stats cumulate over theory
    coverage = sum(th_p + th_n for th_p, th_n in theory_pn[1:])
    fp = sum(th_n for th_p, th_n in theory_pn[1:])
    # uncovered stats are those of the last rule
    if len(theory_pn) > 1:
        # we're not at the first rule (`> 1` since theory_pn contains an entry
        # for the candidate, too)
        uncoverage = P + N - p - n
        fn = N - n
    else:
        # we're at the first rule
        uncoverage = P + N  # == coverage + uncoverage
        fn = p + N - n  # tp + fn
    return data_description_length(
        expected_fp_over_err=expected_fp_over_err,
        covered=coverage, uncovered=uncoverage, fp=fp, fn=fn)


def relative_description_length(rule: AugmentedRule,
                                expected_fp_over_err,
                                p, n, P, N, theory_pn) -> float:
    """:return: DL of the current theory relative to removing `rule`.

    NOTE: Named `RuleStats.relativeDL` in weka.JRip. JRip has the `index`
    parameter which is only `!= last_rule` in global optimization.
    """
    dDL_with = minDataDLIfExists(expected_fp_over_err, p, n, P, N, theory_pn)
    rule_DL = rule_description_length(rule)
    dDL_wo = minDataDLIfDeleted(expected_fp_over_err, p, n, P, N, theory_pn)
    return dDL_with + rule_DL - dDL_wo
