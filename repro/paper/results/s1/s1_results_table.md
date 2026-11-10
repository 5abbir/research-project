# Scenario 1 results: raw CSV (no filling, no dropping, no scaling, no resampling, no feature selection)
**Reproduction results, not validated performance.** Single stratified 80/20 split, seed 42, library defaults (AutoGluon: project settings carried over from v2). Every runnable model was scored once on the test rows; no model was selected, tuned or rejected on test performance. Do NOT compare these numbers with Scenario 2/3 (different missing-value treatment).

- Data: csv: ../raw_water.csv  sha256: 904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf
- Test rows: 656; eval (test) rows containing >=1 NaN: 252
- Majority-class baseline on test rows: **60.98%** (majority-class baseline on the test rows).
- Rough noise guide: standard error of one accuracy near the baseline = about 1.9 points (one test split; differences of 1-2 points are not meaningful).

## Consistency checks (computed from the saved files)
- PASS: same test rows in all three tools (656 / 656 / 656 rows)
- PASS: sklearn: accuracy equals accuracy recomputed from confusion counts (2 models)
- PASS: pycaret: accuracy equals accuracy recomputed from confusion counts (6 models)
- PASS: autogluon: accuracy equals accuracy recomputed from confusion counts (12 models)

## A. sklearn models (env_main)
Library versions: sklearn 1.9.1, xgboost 3.2.0. NaN handling: model's own NaN support (XGBoost, RandomForest native); AdaBoost rejects NaN.
AdaBoost is reported as: Not runnable on raw CSV with NaN values; missing-value handling unspecified in paper. Our status: NOT REPRODUCIBLE for AdaBoost under the raw-CSV protocol.

| model | role | result_label | accuracy_pct | above_baseline_pts | precision_pct | recall_pct | specificity_pct | f1_pct | roc_auc | balanced_acc | mcc | validation_auc | best_flags_validation_based |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | Primary | Raw CSV / NaN-native | 62.5 | 1.52 | 52.84 | 36.33 | 79.25 | 43.06 | 0.6378 | 0.5779 | 0.1715 |  |  |
| RandomForest | Primary | Raw CSV / NaN-native | 66.31 | 5.34 | 63.57 | 32.03 | 88.25 | 42.6 | 0.6513 | 0.6014 | 0.2489 |  |  |

**Not runnable:**

| model | role | result_label | cause_if_not_runnable |
|---|---|---|---|
| AdaBoost | Primary | Not runnable | ValueError: Input X contains NaN. | AdaBoostClassifier does not accept m... |

## B. PyCaret library models (env_pycaret)
Library versions: pycaret 3.3.2, sklearn 1.4.2. NaN handling: PyCaret imputation_type=None; each model's own NaN support.
Primary = xgboost, rf, ada (the paper's three). Exploratory = every other runnable model; they were NOT pre-specified. 'CV-best' flags use train-only 5-fold CV AUC, never test.

| model | role | result_label | accuracy_pct | above_baseline_pts | precision_pct | recall_pct | specificity_pct | f1_pct | roc_auc | balanced_acc | mcc | validation_auc | best_flags_validation_based |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | Exploratory | Raw CSV / NaN-native | 65.85 | 4.88 | 61.94 | 32.42 | 87.25 | 42.56 | 0.6662 | 0.5984 | 0.238 | 0.6204 |  |
| dt | Exploratory | Raw CSV / NaN-native | 57.62 | -3.35 | 45.77 | 46.48 | 64.75 | 46.12 | 0.5562 | 0.5562 | 0.112 | 0.5512 |  |
| dummy | Exploratory | Raw CSV / NaN-native | 60.98 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.5 | 0.5 | 0.0 | 0.5 |  |
| lightgbm | Exploratory | Raw CSV / NaN-native | 66.31 | 5.34 | 60.36 | 39.84 | 83.25 | 48.0 | 0.6627 | 0.6155 | 0.2576 | 0.618 |  |
| rf | Primary | Raw CSV / NaN-native | 65.4 | 4.42 | 60.74 | 32.03 | 86.75 | 41.94 | 0.6445 | 0.5939 | 0.2266 | 0.6297 | CV-best overall, CV-best primary |
| xgboost | Primary | Raw CSV / NaN-native | 62.5 | 1.52 | 52.84 | 36.33 | 79.25 | 43.06 | 0.6378 | 0.5779 | 0.1715 | 0.6115 |  |

**Not runnable:**

| model | role | result_label | cause_if_not_runnable |
|---|---|---|---|
| et | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| gbc | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| gpc | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| knn | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| lda | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| lr | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| mlp | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| nb | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| qda | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| rbfsvm | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| ridge | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| svm | Exploratory | Not runnable | Input X contains NaN (NaN not accepted) |
| ada | Primary | Not runnable | Input X contains NaN (NaN not accepted) |

## C. AutoGluon leaderboard (env_main)
Library versions: autogluon 1.6.3. NaN handling: AutoGluon internal, model-specific handling (default preprocessing untouched).
Label: Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 (medium_quality, 5 bag folds, 0 stack levels, 300 s, 8 CPUs; not AutoGluon defaults). Results reflect AutoGluon's internal handling, not a common imputation. Sorted by validation score, not test score.

| model | role | result_label | accuracy_pct | above_baseline_pts | precision_pct | recall_pct | specificity_pct | f1_pct | roc_auc | balanced_acc | mcc | validation_auc | best_flags_validation_based |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WeightedEnsemble_L2 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 67.23 | 6.25 | 66.94 | 31.64 | 90.0 | 42.97 | 0.6662 | 0.6082 | 0.2722 | 0.7285 | best overall by validation |
| NeuralNetFastAI_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 65.4 | 4.42 | 60.0 | 33.98 | 85.5 | 43.39 | 0.653 | 0.5974 | 0.2291 | 0.7103 | best non-ensemble by validation |
| NeuralNetTorch_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 65.4 | 4.42 | 58.58 | 38.67 | 82.5 | 46.59 | 0.6558 | 0.6059 | 0.2362 | 0.7064 |  |
| LightGBMXT_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 65.55 | 4.57 | 60.56 | 33.59 | 86.0 | 43.22 | 0.6614 | 0.598 | 0.2321 | 0.7031 |  |
| CatBoost_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 66.16 | 5.18 | 63.28 | 31.64 | 88.25 | 42.19 | 0.6655 | 0.5995 | 0.2448 | 0.6845 |  |
| LightGBMLarge_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 69.05 | 8.08 | 75.73 | 30.47 | 93.75 | 43.45 | 0.6752 | 0.6211 | 0.3247 | 0.6795 |  |
| LightGBM_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 68.75 | 7.77 | 72.17 | 32.42 | 92.0 | 44.74 | 0.6807 | 0.6221 | 0.3133 | 0.6789 |  |
| RandomForestEntr_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 66.01 | 5.03 | 63.64 | 30.08 | 89.0 | 40.85 | 0.6603 | 0.5954 | 0.2399 | 0.6777 |  |
| XGBoost_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 67.53 | 6.55 | 70.87 | 28.52 | 92.5 | 40.67 | 0.6875 | 0.6051 | 0.2818 | 0.6758 |  |
| ExtraTreesGini_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 64.63 | 3.66 | 66.22 | 19.14 | 93.75 | 29.7 | 0.6348 | 0.5645 | 0.1988 | 0.6755 |  |
| RandomForestGini_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 66.16 | 5.18 | 63.49 | 31.25 | 88.5 | 41.88 | 0.6481 | 0.5988 | 0.2446 | 0.6721 |  |
| ExtraTreesEntr_BAG_L1 | AutoGluon leaderboard | Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2 | 65.7 | 4.73 | 70.13 | 21.09 | 94.25 | 32.43 | 0.6218 | 0.5767 | 0.2325 | 0.6719 |  |

## D. Paper's Scenario 1 (Table 5) - REFERENCE ONLY, not our results, not merged
Source: Shehab et al. 2025, Table 5 (original data). The paper's split, seed, hyper-parameters and library versions are unknown, and its missing-value handling for this scenario is unspecified.

| model | precision_pct | sensitivity_pct | specificity_pct | f1_pct | accuracy_pct | note |
|---|---|---|---|---|---|---|
| XGBoost | 63.7 | 32.37 | 89.07 | 42.93 | 67.98 |  |
| RandomForest | 63.97 | 35.65 | 88.1 | 45.78 | 68.6 | text Sect. 5.4 says 68.50; Table 5 / conclusion say 68.60 |
| AdaBoost | 48.97 | 19.67 | 87.86 | 28.07 | 62.5 |  |

## E. Limits and honest reading
- One split of 656 test rows: accuracy standard error about 1.9 points, AUC about 0.02; ranking models on it would be over-reading.
- sklearn/xgboost versions differ between tools (see versions above); PyCaret's rf and the env_main RandomForest are different builds.
- AutoGluon's validation (out-of-fold) AUC is optimistic compared with test AUC; AutoGluon's bagged models are not the same configuration as the default sklearn models.
- The paper's Scenario 1 numbers are neither reproduced nor refuted by this single run; AdaBoost cannot be reproduced under the raw-CSV protocol.
- Scenario 1 uses a different missing-value treatment from Scenarios 2/3, so it must not be compared with them.
