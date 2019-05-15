- INFO: other implementations
    - sklearn.dtree: arrays explained at <http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html>
    - R impl of C5.0 <https://github.com/topepo/C5.0/blob/master/src/trees.c>
    - weka source of C4.5 v8 <https://github.com/bnjmn/weka/blob/master/weka/src/main/java/weka/classifiers/trees/J48.java>

- performance comparison
    - with sklearn.RandomForest, CART
    - with weka.JRip J48 PRISM CN2
- optimization <http://www.scipy-lectures.org/advanced/optimizing/index.html>
    - categorical features as bitvectors
        - <https://stackoverflow.com/questions/5602155/numpy-boolean-array-with-1-bit-entries>
        - problem: NaN == no rule test, need separate mask array
    - (multiple?) rule matching as matrix operation/multiplication

- assert & document possibility to change rule format in SeCoImplementation
    without changing sklearn_seco, i.e. it could be extended to other rule
    language → language bias

- python3.5 compatibility, esp. type hints
- submission to upstream: sklearn-contrib? future maintainer needed?

