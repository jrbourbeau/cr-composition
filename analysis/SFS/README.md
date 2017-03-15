# Sequential feature selection (SFS)

Sequential feature selection algorithm implementation can be found in the [mlxtend documentation](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#sequential-feature-selector). According to the documentation,

<blockquote>
In a nutshell, SFAs remove or add one feature at the time based on the classifier performance until a feature subset of the desired size k is reached.
</blockquote>

### Layout:
---
* `sequential-feature-selection.py` &mdash; Runs SFS algorithm
* `submitter.py` &mdash; Submits `sequential-feature-selection.py` with various options to HTCondor
* `SFS-results/` &mdash; Directory for storing results of the SFS algorithms
* `plotting-SFS-results.ipynb` &mdash; Notebook for plotting results
