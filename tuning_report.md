# StreamLoop — Tuning the Churn Model

## Metric choice and business reason
Recall is the primary selection and comparison metric because failing to identify a customer who will churn (a false negative) is more costly than sending an unnecessary retention offer (a false positive). Precision and F1 are included to show the associated offer burden and balance.

## Data and leakage controls
The notebook downloads the IBM Telco Customer Churn CSV directly from the specified URL, drops `customerID`, converts blank `TotalCharges` entries to missing numeric values, and encodes `Churn` as Yes=1/No=0. It creates a stratified 80/20 split with `random_state=42` before fitting preprocessing. Median numeric imputation and most-frequent categorical imputation/one-hot encoding are contained in the pipeline.

The held-out test set is used only once for the default baseline and once for the final grid-search estimator. Randomized and grid searches, including preprocessing fits, receive only `X_train` and `y_train`.

## Baseline performance

The default `RandomForestClassifier(random_state=42)` was evaluated once on the held-out set:

| Model | Test recall | Test precision | Test F1 |
|---|---:|---:|---:|
| Default Random Forest | 0.4759 | 0.6159 | 0.5370 |

## RandomizedSearchCV approach
A broad Random Forest search evaluated 24 sampled candidates using five-fold cross-validation, recall scoring, `n_jobs=-1`, `random_state=42`, and `refit=True`. The best mean CV recall was **0.7906** (standard deviation **0.0217**). The randomized-search winner was `class_weight=balanced_subsample`, `max_depth=5`, `max_features=0.8`, `min_samples_leaf=3`, `min_samples_split=13`, and `n_estimators=154`. The test set did not inform the search.

## GridSearchCV approach
A smaller grid centered on the randomized winner's training-CV region evaluated 81 candidates using five-fold recall CV with `n_jobs=-1` and `refit=True`. Its best mean CV recall was **0.7926**. The notebook displays the five leading grid candidates with mean and standard deviation of CV recall.

## Final hyperparameters
The selected estimator is the grid-search refitted `best_estimator_` (not manually refit):

```text
classifier__class_weight: balanced_subsample
classifier__max_depth: 5
classifier__max_features: 0.8
classifier__min_samples_leaf: 2
classifier__min_samples_split: 11
classifier__n_estimators: 100
```

## Top CV candidates and stability
| Rank | Mean CV recall | Std. CV recall | Key settings (class weight / depth / leaf / split / trees) |
|---:|---:|---:|---|
| 1 (tie) | 0.792642 | 0.019502 | balanced_subsample / 5 / 2 / 11 / 100 |
| 1 (tie) | 0.792642 | 0.020399 | balanced_subsample / 5 / 2 / 11 / 154 |
| 1 (tie) | 0.792642 | 0.020940 | balanced_subsample / 5 / 2 / 11 / 254 |
| 4 (tie) | 0.791973 | 0.020442 | balanced_subsample / 5 / 2 / 13 / 100 |
| 4 (tie) | 0.791973 | 0.020442 | balanced_subsample / 5 / 2 / 13 / 154 |

All five candidates have similar mean recall and standard deviations around 0.02, indicating modest fold-to-fold variability rather than a large instability. The selected model is among the tied top mean-score candidates and has the lowest standard deviation of that tied group (0.019502); it also uses the smallest tree count, reducing compute while retaining the same measured mean recall. This is a small stability/efficiency preference, not evidence of a materially superior model.

Stability is assessed from the standard deviation of fold recall as well as the mean: a slightly lower mean paired with less fold-to-fold variation can be a more dependable choice than a narrow best-mean advantage.

## Baseline vs tuned test performance
The baseline and tuned model were each evaluated exactly once on the same untouched test set:

| Model | Test recall | Test precision | Test F1 | Recall change vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 0.4759 | 0.6159 | 0.5370 | 0.0000 |
| Tuned GridSearchCV model | 0.7754 | 0.5321 | 0.6311 | +0.2995 |

The tuned model recovers substantially more churners (about a 29.95 percentage-point recall improvement), at the cost of lower precision and therefore more false-positive retention offers. Given the stated higher cost of missed churners, this is the intended trade-off. Test results were not used to choose parameters.
