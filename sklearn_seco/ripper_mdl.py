"""
MDL (minimum description length) calculation for RIPPER.

Mostly adapted from JRip.java, see `sklearn_seco.concrete.RipperMdlStop`.
"""

import numpy as np

from sklearn_seco.common import AugmentedRule, log2


def subset_description_length(n, k, p):
    return -k * log2(p) - (n - k) * log2(1 - p)


def data_description_length(expected_fp_over_err, covered, uncovered, fp, fn):
    """XXX"""
    S = subset_description_length
    total_bits = log2(covered + uncovered + 1)
    if covered > uncovered:
        assert covered > 0
        expected_error = expected_fp_over_err * (fp + fn)
        covered_bits = S(covered, fp, expected_error / covered)
        uncovered_bits = S(uncovered, fn, fn / uncovered) \
            if uncovered > 0 else 0
    else:
        assert uncovered > 0
        expected_error = (1 - expected_fp_over_err) * (fp + fn)
        covered_bits = S(covered, fp, fp / covered) \
            if covered > 0 else 0
        uncovered_bits = S(uncovered, fn, expected_error / uncovered)
    return total_bits + covered_bits + uncovered_bits


def rule_description_length(rule: AugmentedRule):
    n_conditions = np.count_nonzero(rule.conditions)  # no. of conditions
    max_n_conditions = rule.conditions.size  # possible no. of conditions
    # TODO: JRip counts all thresholds (RuleStats.numAllConditions())

    kbits = log2(n_conditions)  # no. of bits to send `n_conditions`
    if n_conditions > 1:
        kbits += 2 * log2(kbits)
    rule_dl = kbits + subset_description_length(
        max_n_conditions, n_conditions, n_conditions / max_n_conditions)
    return rule_dl * 0.5  # redundancy factor


def minDataDLIfExists(expected_fp_over_err, p, n, P, N, theory_pn):
    return data_description_length(
        expected_fp_over_err=expected_fp_over_err,
        covered=sum(th_p + th_n for th_p, th_n in theory_pn),  # of theory
        uncovered=P + N - p - n,  # of rule
        fp=sum(th_n for th_p, th_n in theory_pn),  # of theory
        fn=N - n,  # of rule
    )


def minDataDLIfDeleted(expected_fp_over_err, p, n, P, N, theory_pn):
    # covered stats cumulate over theory
    coverage = sum(th_p + th_n for th_p, th_n in theory_pn)
    fp = sum(th_n for th_p, th_n in theory_pn)
    # uncovered stats are those of the last rule
    if len(theory_pn) > 1:
        # we're not at the first rule (theory_pn contains an entry for the
        # candidate, too)
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
                                expected_fp_over_err, p, n, P, N, theory_pn):
    """XXX: from JRip

    NOTE JRip has the index parameter which is only !=last_rule in global optimization
    """
    ddl_ex = minDataDLIfExists(expected_fp_over_err, p, n, P, N, theory_pn)
    rule_DL = rule_description_length(rule)
    ddl_del = minDataDLIfDeleted(expected_fp_over_err, p, n, P, N, theory_pn)
    return ddl_ex + rule_DL - ddl_del
