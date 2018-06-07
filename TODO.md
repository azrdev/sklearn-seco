- INFO: other implementations
    - sklearn.dtree: arrays explained at <http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html>
    - R impl of C5.0 <https://github.com/topepo/C5.0/blob/master/src/trees.c>
    - weka source of C4.5 v8 <https://github.com/bnjmn/weka/blob/master/weka/src/main/java/weka/classifiers/trees/J48.java>

- performance comparison
    - with sklearn.RandomForest, CART
    - with weka.JRip J48 PRISM CN2
    - test data sets: <http://scikit-learn.org/stable/datasets/index.html>

- CI
    - `setup.py`
    - autochecks flake8, `nosetest --with-coverage`
- optimization <http://www.scipy-lectures.org/advanced/optimizing/index.html>

- rule visualization
- regression?
- assert & document possibility to change rule format in SeCoImplementation
    without changing sklearn_seco, i.e. it could be extended to other rule
    language → language bias

- python2 compatibility, esp. type hints
- submission to upstream: sklearn-contrib? future maintainer needed?
