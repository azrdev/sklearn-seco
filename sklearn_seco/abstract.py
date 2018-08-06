"""
Implementation of SeCo / Covering algorithm: Abstract base algorithm.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.utils.validation import check_is_fitted


# noinspection PyAttributeOutsideInit
class _BinarySeCoEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 implementation: 'SeCoBaseImplementation',
                 trace_coverage='refinements'):
        super().__init__()
        self.implementation = implementation
        self.trace_coverage = trace_coverage  # TODO: document, get from SeCoEstimator

    def fit(self, X, y, categorical_features=None):
        """Build the decision rule list from training data `X` with labels `y`.

        :param categorical_features: None or “all” or array of indices or mask.

            Specify what features are treated as categorical, i.e. equality
            tests are used for these features, based on the set of values
            present in the training data.

            Note that numerical features may be tested in multiple (inequality)
            conditions of a rule, while multiple equality tests (for a
            categorical feature) would be useless.

            -   None (default): All features are treated as numerical & ordinal.
            -   'all': All features are treated as categorical.
            -   array of indices: Array of categorical feature indices.
            -   mask: Array of length n_features and with dtype=bool.

            You may instead transform your categorical features beforehand,
            using e.g. :class:`sklearn.preprocessing.OneHotEncoder` or
            :class:`sklearn.preprocessing.Binarizer`.
            TODO: compare performance
        """
        X, y = check_X_y(X, y)

        # prepare  target / labels / y

        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        self.target_class_ = self.classes_[0]

        # prepare  attributes / features / X
        self.n_features_ = X.shape[1]
        # categorical_features modeled after OneHotEncoder
        if (categorical_features is None) or not len(categorical_features):
            self.categorical_mask_ = np.zeros(self.n_features_, dtype=bool)
        elif isinstance(categorical_features, np.ndarray):
            self.categorical_mask_ = np.zeros(self.n_features_, dtype=bool)
            self.categorical_mask_[np.asarray(categorical_features)] = True
        elif categorical_features == 'all':
            self.categorical_mask_ = np.ones(self.n_features_, dtype=bool)
        else:
            raise ValueError("categorical_features must be one of: None,"
                             " 'all', np.ndarray of dtype bool or integer,"
                             " but got {}.".format(categorical_features))

        # run SeCo algorithm
        self.theory_ = self.abstract_seco(X, y)
        return self

    def find_best_rule(self) -> 'AugmentedRule':
        """Inner loop of abstract SeCo/Covering algorithm.

        :param X: Not yet covered examples.
        :param y: Classification for `X`.
        """

        # resolve methods once for performance
        init_rule = self.implementation.init_rule
        rate_rule = self.implementation.rate_rule
        select_candidate_rules = self.implementation.select_candidate_rules
        refine_rule = self.implementation.refine_rule
        inner_stopping_criterion = self.implementation.inner_stopping_criterion
        filter_rules = self.implementation.filter_rules

        # algorithm
        best_rule = init_rule()
        rate_rule(best_rule)
        rules: RuleQueue = [best_rule]
        while len(rules):
            for candidate in select_candidate_rules(rules):
                for refinement in refine_rule(candidate):  # TODO: parallelize here?
                    rate_rule(refinement)
                    if not inner_stopping_criterion(refinement):
                        rules.append(refinement)
                        if best_rule < refinement:
                            best_rule = refinement
            rules.sort()
            rules = filter_rules(rules)
        return best_rule

    def abstract_seco(self, X: np.ndarray, y: np.ndarray) -> 'Theory':
        """Main loop of abstract SeCo/Covering algorithm.

        :return: Theory
        """

        # resolve methods once for performance
        set_context = self.implementation.set_context
        rule_stopping_criterion = self.implementation.rule_stopping_criterion
        find_best_rule = self.find_best_rule
        simplify_rule = self.implementation.simplify_rule
        unset_context = self.implementation.unset_context
        post_process = self.implementation.post_process
        match_rule = self.implementation.match_rule

        target_class = self.target_class_
        if self.trace_coverage:
            self.coverage_log_ = []
            P0 = np.count_nonzero(y == target_class)
            self.NP0 = np.array((len(y) - P0, P0))
            previous_PN = np.array((0, 0))
        elif hasattr(self, 'coverage_log_'):  # cleanup previous run
            del self.coverage_log_
            del self.NP0
        # TODO: split growing/pruning set for ripper
        # main loop
        theory: Theory = list()
        while np.any(y == target_class):
            set_context(self, X, y)
            rule = find_best_rule()
            rule = simplify_rule(rule)
            if self.trace_coverage:  # TODO: optimize trace_coverage, maybe move elsewhere
                # note swapping from (p,n) to (x=n, y=p)
                from sklearn_seco.common import rule_ancestors
                if self.trace_coverage == 'best_rules':
                    coverages = np.array([(rule._n, rule._p)]) + previous_PN
                    previous_PN = coverages
                elif self.trace_coverage == 'refinements':
                    coverages = np.array([(r._n, r._p) + previous_PN
                                          for r in rule_ancestors(rule)])
                    previous_PN = coverages[0]
                elif self.trace_coverage == 'candidates':
                    coverages = np.empty((0,))  # XXX: different structure to distinguish refs from cands
                    previous_PN = coverages
                else:
                    raise ValueError("TODO")
                self.coverage_log_.append(coverages)
            if rule_stopping_criterion(theory, rule):
                break
            # ignore the rest of theory, because it already covered
            uncovered = ~ match_rule(rule)
            X = X[uncovered]  # TODO: use mask array instead of copy?
            y = y[uncovered]
            theory.append(rule.conditions)  # throw away augmentation
        unset_context()
        return post_process(theory)

    def plot_coverage_log(self, title=None):
        """TODO: doc"""
        check_is_fitted(self, ['coverage_log_'])
        import matplotlib.pyplot as plt
        theory_fig = plt.figure()
        theory_axis = theory_fig.gca(xlabel='n', ylabel='p',
                                     xlim=(0, self.NP0[0]),
                                     ylim=(0, self.NP0[1]))
        theory_axis.plot([0, self.NP0[0]], [0, self.NP0[1]], ':',
                         color='grey', alpha=0.5)  # "random theory" marker
        n_plot_sqrt = int(np.ceil(np.sqrt(len(self.coverage_log_))))
        rules_fig, axes = plt.subplots(n_plot_sqrt, n_plot_sqrt, squeeze=False,
                                       figsize=(10.24, 10.24))
        rule_axes = axes.flat
        for i in range(len(self.coverage_log_), len(rule_axes)):
            rules_fig.delaxes(rule_axes[i])

        previous_best_rule = np.array((0,0))  # equals (N, P) for some trace
        for rule_idx, rule_trace in enumerate(self.coverage_log_):
            best_rule = rule_trace[0]  # first=best_rule is in the theory
            theory_line = theory_axis.plot(
                best_rule[0], best_rule[1], '.',
                label="{2}: ({1}, {0})".format(*best_rule, rule_idx))
            # draw arrows between best_rules
            theory_axis.annotate("", xytext=previous_best_rule, xy=best_rule,
                                 arrowprops={'arrowstyle': "->"})

            # subplot with refinements of current rule (how it was found)
            rule_axis = rule_axes[rule_idx]
            # grey out area already covered by previous rules
            Ni, Pi = previous_best_rule
            rule_axis.fill_between([0, Ni, Ni, self.NP0[0]],
                                   [self.NP0[1], self.NP0[1], Pi, Pi],
                                   facecolor='lightgray', alpha=0.3)
            # draw "random theory" marker
            rule_axis.plot([0, self.NP0[0]], [0, self.NP0[1]], ':',
                           color='grey', alpha=0.5)
            # draw rule_trace
            rule_axis.plot(rule_trace[:, 0], rule_trace[:, 1], 'o-',
                           color=theory_line[0].get_color())
            rule_axis.set_title('Rule #%d' % rule_idx)
            rule_axis.set_xlabel('n')
            rule_axis.set_ylabel('p')
            rule_axis.set_xlim(0, self.NP0[0])
            rule_axis.set_ylim(0, self.NP0[1])
            rule_axis.label_outer()
            previous_best_rule = best_rule

        if title is not None:
            theory_axis.set_title("%s: Theory" % title)
            rules_fig.suptitle("%s: Rules" % title)  # TODO: intersects plots
        theory_fig.legend(title="rule: (p,n)")

        theory_fig.show()
        rules_fig.show()
        return theory_fig, rules_fig

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['theory_', 'categorical_mask_'])
        X: np.ndarray = check_array(X)
        target_class = self.target_class_
        match_rule = self.implementation.match_rule_raw
        result = np.repeat(self.classes_[1],  # negative class
                           X.shape[0])

        for rule in self.theory_:
            result = np.where(
                match_rule(rule, X),
                target_class,
                result)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prediction = self.predict(X) == self.target_class_
        return np.where(prediction[:, np.newaxis],
                        np.array([[1, 0]]),
                        np.array([[0, 1]]))


# noinspection PyAttributeOutsideInit
class SeCoEstimator(BaseEstimator, ClassifierMixin):
    """Wrap the base SeCo to provide class label binarization."""

    def __init__(self, implementation: 'SeCoBaseImplementation',
                 multi_class="one_vs_rest", n_jobs=1):
        self.implementation = implementation
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwargs):
        # TODO: document available kwargs or link `_BinarySeCoEstimator.fit`
        X, y = check_X_y(X, y, multi_output=False)

        self.base_estimator_ = _BinarySeCoEstimator(self.implementation)

        # copied from GaussianProcessClassifier
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError("SeCoEstimator requires 2 or more distinct "
                             "classes. Only class %s present."
                             % self.classes_[0])
        elif self.n_classes_ > 2:
            # TODO: multi_class strategy of ripper: OneVsRest, remove C_i after learning rules for it
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(self.base_estimator_,
                                                           n_jobs=self.n_jobs)
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = OneVsOneClassifier(self.base_estimator_,
                                                          n_jobs=self.n_jobs)
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class)

        self.base_estimator_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        check_is_fitted(self, ["classes_", "n_classes_"])
        X = check_array(X)
        return self.base_estimator_.predict(X)


# imports needed only for type checking, place here to break circularity
from sklearn_seco.common import \
    RuleQueue, AugmentedRule, SeCoBaseImplementation, Theory
