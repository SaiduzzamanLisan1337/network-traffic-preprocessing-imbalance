# Experimental Methodology

## Objective
Evaluate the effect of standardization, feature selection, class weighting, and SMOTE on minority-class intrusion detection.

## Leakage control
All preprocessing is fitted inside an imbalanced-learn pipeline separately within each cross-validation training fold. SMOTE is applied only to the training partition. The held-out fold remains untouched until evaluation.

## Primary metrics
- Minority-class recall
- Minority-class F1
- Macro-F1
- Balanced accuracy
- PR-AUC

Accuracy is reported for context but is not the primary decision metric because severe class imbalance can make it misleading.

## Statistical analysis
The experiment records fold-level metrics. Paired t-tests and Wilcoxon signed-rank tests compare each strategy with the baseline using matched CV folds. Results should be interpreted alongside effect size / mean difference, not p-values alone.

## Reproducibility
Random seeds are fixed in the configuration. Generated figures and tables are deterministic given the same software versions, dataset, and configuration.
