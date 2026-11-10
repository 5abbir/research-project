# S2_PROTOCOL.md — Scenario 2 (fill nulls per Sect. 4.2 + SMOTE), reproduction track
Save as: `~/research/repro/S2_PROTOCOL.md`. Status: DECISIONS RECORDED, plan PROPOSED, no S2 code run. Label: **reproduction results, not validated performance**.
Result label for S2 rows: "Section 4.2 correlation-based imputation" (+ track tag). Never compare with S1 (raw CSV) as if the treatment were identical.

## 1. Decisions (USER, 2026-11-07: "accept recommended")
| ID | Decision |
|---|---|
| B1 | Imputation model = Random Forest regressor |
| B2 | Tuning = grid search + 5-fold CV on the non-null rows of the column being filled. Grid (DELEGATED to Claude by the user 2026-11-08, [P]; user may overrule): n_estimators {100, 300} x max_depth {None, 10} x min_samples_leaf {1, 5, 25} = 12 combinations (section 4) |
| B3 | "Positive co-related" rule = DECIDED (DELEGATED to Claude 2026-11-08, [P]; user may overrule): predictor needs Pearson r > 0 with the column being filled, r from pairwise-complete TRAIN rows computed once before any filling (r3_corr_table.py, run on real data [C]); only columns that are fully filled at that step may be predictors (complete columns + earlier filled columns). Chosen because it is the paper's literal wording (sign, no threshold). If a rule leaves no candidate: stop and ask (exit 3) |
| B4 | Potability NOT used as a predictor (leak-free). Including it = labelled leaky diagnostic, only if you approve later |
| B5 | Fill order = fewest missing first: Trihalomethanes, then ph, then Sulfate (later columns may use earlier filled ones) |
| B6 | Two tracks: **Track L** (leak-free, PRIMARY): split first, fit fill models on train rows only, apply to test. **Track P** (paper-literal, leaky by design, DIAGNOSTIC ONLY): fill on all rows before the split |
| B7 | Fill once on the 80/20 split (not inside CV folds). Limitation: PyCaret/AutoGluon later cross-validate inside already-filled train data (mild fold-to-fold leakage) |
| B8 | Regression (columns are continuous) |
| B9 | N/A (no missing target) |
| B10 | Two tracks as B6: Track L = SMOTE after the split on train rows only; Track P = SMOTE before the split on all rows (diagnostic) |
| B11 | REVISED 2026-11-08 (DELEGATED to Claude, [P]; user may overrule) after re-reading the paper: the paper is inconsistent about whether its phase-2 results include SMOTE (section 6 below). So S2 is reported as TWO co-primary variants, both Track L: **S2a = fill only** (imbalanced train rows) and **S2b = fill + SMOTE** (train rows only). S3 = fill + SMOTE + fuzzy-rough (paper Sect. 5.3 and 5.4 agree on this). Earlier text ("fill-only as a side comparison") is superseded |
| B12 | SMOTE: imblearn defaults (k=5), random_state=42, balances to the majority count (about 1:1). The paper's 1508/1508 is NOT forced (cannot come from SMOTE alone); discrepancy logged (D4) |
| B13 | note only |

## 2. Track definitions (what makes them comparable or not)
- Track L test set = the SAME 656 original test rows as S1 (never filled with test statistics, never resampled). Train = 2620 rows filled with train-only fill models, then SMOTE.
- Track P: fill + SMOTE on all 3276 rows, then 80/20 split of the BALANCED data (test rows can be synthetic and can resemble train rows). Its test set differs from Track L, so P and L scores are NOT comparable. Purpose: check whether the paper's high accuracy can arise from order alone. Labelled "Paper-literal (leaky) diagnostic" everywhere.

## 3. Leakage risks to guard in code
1. Fill models and the correlation rule must see TRAIN rows only in Track L (the correlation table is already train-only).
2. Potability must not enter any fill model (B4); assert in code.
3. Test rows must be transformed with models fit on train, never refit.
4. SMOTE on train rows only in Track L; never on test rows.
5. Tuning CV for the fill model runs on non-null train rows of that column only.
6. Track P is leaky on purpose: its outputs carry the diagnostic label and are never merged into Track L tables.
7. Smoke runs use train rows only; no test rows opened.

## 4. Fill-model grid (decided by Claude on delegation, [P])
RandomForestRegressor(random_state=42, n_jobs=8). Grid: n_estimators {100, 300} x max_depth {None, 10} x min_samples_leaf {1, 5, 25} = 12 combinations; KFold(5, shuffle, seed 42) on non-null train rows; metric = negative mean squared error; best refit on all non-null train rows. Reported per column: best params, CV MSE, and CV R2 relative to the "predict the column mean" baseline (same folds). If a tuned model is not better than the mean, that is reported as a finding; NO fallback replaces it.
Why larger leaves are included (analysis, from the real train-only correlation table): all |r| are at most 0.20 and the positive ones at most 0.064; with that little signal a fully grown forest (leaf size 1) mostly fits noise, so the grid must be able to choose smooth models (leaf 25), otherwise the fill would be worse than the mean.
Real-data predictors under the B3 rule (from the table; fixed before filling): Trihalomethanes <- Chloramines, Organic_carbon, Conductivity (complete columns only at step 1); ph <- Hardness, Organic_carbon, Conductivity, Trihalomethanes (r 0.064, 0.057, 0.027, 0.0005); Sulfate <- Chloramines, ph, Organic_carbon (r 0.040, 0.034, 0.033). Every column keeps at least 3 predictors, so no stop-and-ask is expected.
Honest expectations: the only clear relation in the data is Solids vs Sulfate, r = -0.20 (95% CI -0.24 to -0.16), which the "positive" rule EXCLUDES. Fills will therefore be close to each column's mean. An optional later side comparison with an |r| rule could be added if you ask.

## 5. Step order (one step at a time)
S2-1 correlation table, train rows only: DONE [C].
S2-2 B3 and grid: DECIDED on delegation (above).
S2-3 fill script r3_fill_trackL.py (Track L): WRITTEN; smoke on train subsample, then full. Outputs: s2_filled_train_L.csv, s2_filled_test_L.csv, s2_fill_models_L.csv, s2_fill_report_L.txt, s2_fill_run_info_L.json.
S2-4a SMOTE data-prep script r3_make_smote_trackL.py (writes s2_smote_train_L.csv from the filled train rows only).
S2-4b XGBoost/RF/AdaBoost on S2a and S2b, then PyCaret and AutoGluon on S2a and S2b (E1 decided, section 7). One script at a time; smoke before full.
S2-5 fill-only ablation. S2-6 Track P diagnostic. S2-7 report table.

## 6. What the paper says about the scenarios (checked 2026-11-08 against q1_q2_23_98_.md) [C for the quotes, P for the reading]
- Sect. 5.4: "three categories, the first with original data, the second with phase 2 (fill missing data) and the third is phase 4 (fuzzy rough set + SMOTE + fill missing values)". Table 6 caption: "Phase 1: after filling null values". Table 7 caption: "Phase 4: After filling null values and apply fuzzy rough set". Fig. 3 caption: "after filling null values"; but the Fig. 3 discussion says "after preprocessing data and fill null values".
- Sect. 5.2 title: "Impact of filling missing data and balancing with SMOTE"; Sect. 5.3: "The balanced dataset assigned to fuzzy rough set"; Fig. 1 (user's image description): Phases 1 and 2 (fill nulls, balancing) form one preprocessing block before Phase 3 (fuzzy rough) and Phase 4 (classifiers).
- Reading: the user's reading (second experiment = fill, third = fuzzy-rough + SMOTE + fill) is supported by Sect. 5.4 wording and the Table 6 / Fig. 3 captions. The opposite reading (second experiment already includes SMOTE) is supported by Fig. 1 and the 5.2 title. Table 7's caption omits SMOTE although Sect. 5.3 says the FRS input was the balanced data, so captions are not reliable evidence of what was applied. The paper does not settle it, and numbers cannot settle it either (confusion-matrix digits are run together, [U]). Hence S2a and S2b.
- Phase numbering is inconsistent in the paper (Table 5 = phase 0, Table 6 = phase 1, text = phase 2, Table 7 = phase 4). We use our own names: S1, S2a, S2b, S3.

## 7. E1 decision: how PyCaret and AutoGluon get the SMOTE data (DELEGATED to Claude 2026-11-08, [P]; user may overrule)
Decision: every classifier and every tool receives the SAME already prepared train table, built once by r3_make_smote_trackL.py (SMOTE on train rows only, seed 42, saved to disk with hashes). Tool-internal resampling is OFF (PyCaret fix_imbalance=False; AutoGluon has none by default). The test set is always the same 656 original rows (filled by train-fitted models, never resampled).
Why (analysis):
1. Fair comparison: the project rule is same data, folds, seeds, metrics and budgets. AutoGluon has no SMOTE, so "tools do their own resampling" cannot be matched across tools; only a shared prepared table can.
2. The test set is clean either way, so test scores stay valid. For XGBoost/RF/AdaBoost there is no CV at all (one fit on train rows, test once), so there is no contamination.
3. Cost, stated honestly: PyCaret's CV and AutoGluon's out-of-fold validation run inside the already-resampled train table, so synthetic neighbours of a validation row can sit in the training folds. Their validation scores (PyCaret CV AUC, AutoGluon validation score, ensemble weights, and the validation-based best flags) will be optimistic. Rule: those flags are labelled "validation, contaminated by SMOTE-before-CV" and are never read as performance; only the test scores are results.
4. Optional later diagnostic (NOT planned now, ask first): PyCaret with fix_imbalance=True on the filled, un-resampled train table (SMOTE inside folds).
Known SMOTE limitation to check in the smoke run: imblearn SMOTE uses unscaled Euclidean distance, so large-scale features (Solids, about tens of thousands, [P] from general knowledge of this dataset) may dominate the choice of neighbours. The script prints each feature's standard deviation and its share of squared-distance variance so this is verified from the data. No scaling is added (B12 = imblearn defaults); a scaled-distance variant would be a protocol change and needs approval.

## 8. PLAN for the next script, r3_s2_sklearn_models.py (PROPOSED, no code yet; needs user approval)
Purpose: XGBoost, RandomForest, AdaBoost on S2a and S2b, Track L, same settings as S1 (library defaults, seed 42, n_jobs 8, single fit, no CV, no tuning).
Env: env_main. Inputs: S2a train = results/s2/s2_filled_train_L.csv (2620 rows, 1598/1022); S2b train = results/s2/s2_smote_train_L.csv (3196 rows, 1598/1598, features only: row_id and is_synthetic are dropped before fitting); test = results/s2/s2_filled_test_L.csv (656 original rows, filled by train-fitted models, never resampled).
Assumptions: no NaN anywhere (AdaBoost should now run); no scaling/selection; every model scored ONCE on the test rows, none selected, tuned or rejected on test; predicted probabilities saved.
Leakage risks and guards: (1) test row ids must equal the S1 test ids (checked against results/s1/s1_predictions.csv); (2) assert S2b train row count and class counts, and that is_synthetic and row_id are not in the feature list; (3) test file hash printed before and after; (4) the label never enters a feature list; (5) S2a vs S2b share the same fill and the same test rows, so their difference isolates SMOTE; S1 vs S2 must NOT be compared as if the missing-value treatment were identical.
Smoke (train rows only, test file not opened, logged by a file-access check): fit on results/s2/smote_smoke/s2_smote_train_L.csv (S2b) and on its original rows only (S2a); evaluate on 300 stratified ORIGINAL train rows that are outside the smoke SMOTE input (so no synthetic neighbour of an evaluation row is in the fit). Not reportable.
Commands (smoke, then full, then rerun with --compare-with) and outputs: results/s2/s2_sklearn_metrics.csv, s2_sklearn_confusion.csv, s2_sklearn_predictions.csv, s2_sklearn_report.txt, s2_sklearn_run_info.json (smoke -> results/s2/sklearn_smoke/).
Labels: S2a rows "Section 4.2 correlation-based imputation (Track L)"; S2b rows "Section 4.2 correlation-based imputation + SMOTE on train rows (Track L)".
Then (one script each, each after its own plan and smoke): PyCaret S2a/S2b (fix_imbalance=False, imputation not needed, 19 models expected runnable), AutoGluon S2a/S2b (project settings from v2), then the S2 report script.
Note (2026-11-08): scripts that compare S2b originals with S2a rows read CSVs with float_precision=round_trip and accept <= 1e-12 relative differences (text round-off from the default pandas parser, about 1e-16 seen in tests).

## 9. PLAN for r3_s2_pycaret.py (PROPOSED, no code yet; needs user approval)
Purpose: every PyCaret library model on S2a and S2b (Track L), same data files and test rows as r3_s2_sklearn_models.py.
Env: env_pycaret (pycaret 3.3.2, sklearn 1.4.2; different from env_main, so rf/ada builds differ; versions printed).
Data: train = s2_filled_train_L.csv (S2a) or s2_smote_train_L.csv (S2b), features + target only (row_id and is_synthetic dropped); test_data = s2_filled_test_L.csv (656 original rows). Read with float_precision=round_trip. Same pre-fit checks as the sklearn script (ids vs S1, originals = S2a rows, no NaN, hashes).
PyCaret setup: imputation_type=None (nothing left to impute), normalize=False, fix_imbalance=False (E1: SMOTE already applied once, outside), feature_selection=False, index=False, fold=5 stratified, session_id=42, n_jobs=8, data=train, test_data=test. Assert PyCaret used all supplied train rows (2620 / 3196) and that no NaN remain after its preprocessing.
Models: all 19 library models, one by one (create_model in try/except; full traceback saved; continue); per-model timeout 300 s (smoke 60 s); every runnable model scored ONCE on the 656 test rows; Primary = xgboost, rf, ada, Exploratory = the rest (not pre-specified). With no NaN all 19 are expected to run; any failure is reported with its exact error, not hidden.
"Best" flags: CV AUC on train rows only, labelled "validation; for S2b contaminated by SMOTE-before-CV" (E1); never used as performance, never test-based. Tables are sorted by role and id, not by test score.
Cross-check (diagnostic, not a requirement): PyCaret xgboost on S2a/S2b should equal the env_main XGBoost rows exactly (as in S1); any difference is reported.
Smoke (train rows only, test file not opened, file-access log in my test): S2b from results/s2/smote_smoke/s2_smote_train_L.csv, S2a = its original rows, eval = 300 original train rows outside the smoke fit; first --only rf,ada,lr, then the full library. Not reportable.
Outputs (results/s2/): s2_pycaret_test_metrics.csv, s2_pycaret_model_status.csv, s2_pycaret_confusion.csv, s2_pycaret_predictions.csv (probabilities), s2_pycaret_report.txt, s2_pycaret_run_info.json, tracebacks (smoke -> results/s2/pycaret_smoke/). Rerun with --compare-with reports exact or approximate match.
Labels: as in the sklearn script (S2a / S2b strings).
