# S1_PROTOCOL.md — Scenario 1 (original raw CSV), reproduction track
Save as: `~/research/repro/S1_PROTOCOL.md`. Status: PROPOSED, no code written, nothing run. Label: **reproduction results, not validated performance**.
Separate track: never compare S1 with S2/S3 (R3/R4) unless the missing-value treatment is identical.

## 1. Result labels (every row must carry exactly one)
`Raw CSV / NaN-native` | `Mean-imputed` | `Section 4.2 correlation-based imputation` | `Not runnable`
S1 allows only `Raw CSV / NaN-native` and `Not runnable`. Any tool-internal handling is written in a "NaN handling" column (not treated as equal to a common imputation).

## 2. Protocol table
| ID | Item | Setting | Status |
|---|---|---|---|
| P1 | Data | repro/paper/raw_water.csv, 3276x10, untouched | USER/[C] |
| P2 | Allowed preprocessing | none: no fill, drop, scale, resample, select | USER |
| P3 | Split | stratified 80/20, random_state=42 (train 2620 / test 656 expected [P]; script asserts counts and prints NaN-row counts for train and test) | USER |
| P4 | Hyper-parameters and determinism | library defaults; seed 42 and n_jobs set and logged. Use deterministic settings where available; save seeds, versions and parameters (s1_run_info.json). Do NOT claim bit-for-bit identical reruns: each rerun is compared and reported as "exact match" or "approximate match (max diff)" or "different". AutoGluon may have nondeterministic components | USER (corrected 2026-11-07) |
| P5 | XGBoost, RandomForest | run directly on NaN; label `Raw CSV / NaN-native` | A3a (confirm by smoke test) |
| P6 | AdaBoost | NOT run with any fill. Attempt fit once on raw X, catch exception, record. Report: "Not runnable on raw CSV with NaN values; missing-value handling unspecified in paper." Our status: NOT REPRODUCIBLE under raw-CSV protocol | USER (A3b = i) |
| P7 | AdaBoost record | exact error message, sklearn version, number of NaN-containing rows (total/train/test), whether XGB and RF ran, paper's AdaBoost S1 values (Table 5: precision 48.97, recall 19.67, specificity 87.86, F1 28.07, accuracy 62.50) shown as reference only | USER |
| P8 | PyCaret | `imputation_type=None`, `normalize=False`, no `fix_imbalance`, no feature selection; `data=train, test_data=test` (avoids 70%-rows bug [C]); session_id=42; fold=5 stratified. Each model in the library is tried ONE BY ONE in try/except (`create_model`), so exact errors are captured (compare_models may skip failures quietly [U]). If a model fails: save the FULL traceback to a file, record the exact error, continue with the next model; never hide or silently skip a failure. Runnable -> `Raw CSV / NaN-native`; failing -> `Not runnable` + exact error. Role column: Primary = xgboost, rf, ada (paper's three; PyCaret ids [U] to confirm); Exploratory = every other runnable model (NOT pre-specified; do not claim otherwise) | USER (A3c, Q1) |
| P9 | AutoGluon | run as-is, default preprocessing; same train/test; label `Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2`. Document: RF/XT fill NaN with 0, NeuralNet median, GBMs native [C from v2; confirm in smoke]. Settings (USER, Q2): presets="medium_quality", num_bag_folds=5, num_stack_levels=0, time_limit=300, num_cpus=8. These are a deliberate project setting for comparability with v2, NOT AutoGluon's defaults; not to be changed after seeing results. Default preprocessing untouched. Results reflect AutoGluon's internal handling, not a common imputation. Which AG model is the "best" label (v2 used best non-WeightedEnsemble by validation score) = OPEN, see Q3 | USER (A3c, Q2) |
| P10 | Metrics | accuracy (beside 61% majority baseline), precision, recall, specificity, F1, ROC-AUC, balanced acc, MCC, log loss, Brier; confusion matrix TN/FP/FN/TP saved. Predicted probabilities (class 1) saved per row where the model provides them (s1_predictions.csv), so ROC-AUC, log loss and Brier can be recomputed; hard labels saved too | USER (corrected 2026-11-07) |
| P11 | Test-set use | Every runnable model is scored once on the held-out test set. No model is selected, tuned, or rejected based on test performance. Any "best model" label uses cross-validation/validation scores only. Primary (XGBoost, RF, AdaBoost) and Exploratory (other runnable PyCaret models) reported separately | USER (Q1) |
| P12 | Leakage risks | none learned from data: no fill/scale/resample/selection. Remaining: PyCaret/AutoGluon internal CV only on train; smoke test must NOT use the real 656-row test rows | — |

## 3. NaN capability, installed versions
| Model | Can run on NaN? | Evidence |
|---|---|---|
| XGBoost (xgboost 3.2.0) | YES expected | sandbox synthetic, xgboost 3.4.1: accepts. Not yet on your machine |
| RandomForest (sklearn 1.9.1) | YES expected (native NaN) | sandbox synthetic, sklearn 1.9.1: accepts |
| AdaBoost (sklearn 1.9.1) | NO | sandbox synthetic: ValueError. Exact message to be captured on your machine |
| PyCaret library models (version in env_pycaret [?]) | UNKNOWN [?] | Not tested. I will NOT guess which pass; the probe records it. Likely (hypothesis [U]): LightGBM/XGBoost/CatBoost/HistGB-type pass; LR, KNN, SVM, LDA/QDA, NB, ET/AdaBoost-type may fail |
| AutoGluon 1.6.3 | runs, internal handling | [C] v2 (see P9) |

## 4. Smoke-test plan (no real test rows)
1. Take train rows only (2620 rows, seed 42), stratified subsample 500 rows, then internal stratified 80/20 split (seed 42). Test rows are never used, scored or printed. Wording note: the script reads the whole CSV once, computes the split from labels only, then removes test rows from memory; guards assert the smoke rows are disjoint from the test indices. "Not opened" is therefore stated as "not used" (a stricter option: split into two files first; ask if wanted).
2. sklearn script: XGB, RF run; AdaBoost expected error captured; check outputs have all fields (error text, sklearn version, NaN-row counts).
3. PyCaret probe on the subsample: 3 quick models first (rf, ada, lr) to confirm error capture, then full library loop with a per-model time cap.
4. AutoGluon: time_limit 60 s on the subsample; confirm leaderboard saved.
5. Run each script twice and REPORT whether outputs match exactly or only approximately (no bit-for-bit requirement; AutoGluon may differ). Smoke outputs go to results/s1/smoke/ and are never reported.
7. Before any full run: show the user the smoke outputs and the "test rows used: 0" line; wait for "go".
6. Pass criteria: no crash; errors recorded not hidden; no imputation call anywhere (grep check); counts printed.

## 5. Planned files and commands (scripts not yet written)
Code dir: `~/research/repro/paper/code/`. Outputs: `~/research/repro/paper/results/s1/`.
| Script | Env | Outputs |
|---|---|---|
| r2_s1_sklearn_models.py | env_main | s1_sklearn_metrics.csv, s1_adaboost_record.txt (error, sklearn version, NaN-row counts, XGB/RF success), s1_confusion.csv, s1_split_report.txt |
| r2_s1_pycaret.py | env_pycaret | s1_pycaret_model_status.csv (model, runnable/not runnable, exact error, CV metrics), s1_pycaret_test_metrics.csv, environment_pycaret.txt |
| r2_s1_autogluon.py | env_main | s1_autogluon_leaderboard.csv, s1_autogluon_metrics.csv, s1_autogluon_nan_notes.txt |
| r2_s1_report.py | env_main | s1_results_table.csv/.md (labelled rows; paper values beside ours; 61% baseline) |

Commands (prompt must show the env name):
```
cd ~/research/repro/paper/code
source ../../../env_main/bin/activate
python r2_s1_sklearn_models.py --csv ../raw_water.csv --target Potability --out ../results/s1 --smoke
python r2_s1_sklearn_models.py --csv ../raw_water.csv --target Potability --out ../results/s1
deactivate
source ../../../env_pycaret/bin/activate
python r2_s1_pycaret.py --csv ../raw_water.csv --target Potability --out ../results/s1 --smoke
deactivate
source ../../../env_main/bin/activate
python r2_s1_autogluon.py --csv ../raw_water.csv --target Potability --out ../results/s1 --smoke
python r2_s1_report.py --out ../results/s1
```
Full (non-smoke) runs only after smoke passes and you say go.

## 6. Questions needing your answer before coding
Q1 ANSWERED (USER): test-score every runnable PyCaret model once; Primary (xgboost, rf, ada) vs Exploratory (rest); "best" label by CV only.
Q2 ANSWERED (USER): carry over medium_quality, num_bag_folds=5, num_stack_levels=0, time_limit=300, num_cpus=8 (deliberate project setting, not changed after seeing results).
Q3 ANSWERED (USER, 2026-11-07: "do what's recommended"): for AutoGluon, test-score every model in its leaderboard once; label best non-ensemble and best overall by validation/OOF score only, both marked validation-based. No test-based selection.
