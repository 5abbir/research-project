# S3_PROTOCOL.md — Scenario 3 (fill + SMOTE + fuzzy-rough feature selection), reproduction track
Save as: `~/research/repro/S3_PROTOCOL.md`. Status: PLAN APPROVED 2026-11-10 (all recommended decisions accepted). S3-1 script r4_frs_select.py written; real data not yet run. Label: **reproduction results, not validated performance**.
S3 = S2b (fill + SMOTE, Track L) + feature selection. S3 vs S2b uses the same train rows, the same fill, the same SMOTE and the same 656 test rows; only the feature set differs, so that comparison is clean. PyCaret and AutoGluon for S2 and S3 are deferred (user, 2026-11-09).

## 1. What the paper specifies about the fuzzy-rough step (checked against q1_q2_23_98_.md) [C quotes]
- Sect. 4.4: generic fuzzy-rough definitions: lower approximation min_y I(R(x,y),A(y)) and upper approximation max_y C(R(x,y),A(y)) (Eqs. 4-5, I = implicator, C = conjunctor); fuzzy-equivalence-class form: lower = inf_x max{1 - mu_Fi(x), mu_X(x)} (Eq. 7), upper = sup_x min{mu_Fi(x), mu_X(x)} (Eq. 8); positive region mu_POS(Q)(x) = sup over decision classes of the lower approximation (Eq. 9); dependency gamma_P(Q) = sum_x mu_POS(x) / |U| (Eq. 10). Closing sentence: "Like how the rough set approach is calculated, the reduction set is calculated in the same way." Cited sources: Jensen 2005, Tian et al. 2012, Jensen and Shen. This is the Jensen-Shen fuzzy-rough feature-selection framework (Jensen and Shen 2004, IEEE TKDE 16(12):1457-1471, doi 10.1109/tkde.2004.96, bibliographic details verified; I read only the abstract).
- Sect. 5.3: "The balanced dataset assigned to fuzzy rough set ... The selected features are [6, 7, 3, 2, 1, 5, 8] and ignore features [0 and 4]." Sect. 5 text: phases are fill, SMOTE, fuzzy-rough, classification.
- NOT specified (UNSPECIFIED): the fuzzy similarity relation per attribute, how attributes are combined (t-norm), the implicator/conjunctor, how fuzzy equivalence classes (Eq. 7-8) would be built, the search procedure and stopping rule (a "reduction" is mentioned but QuickReduct is never named), any threshold, whether FRS saw train rows only, normalisation, software/library, seed.

## 2. The feature-index ambiguity (new finding)
The paper's list [6,7,3,2,1,5,8] / dropped [0,4] needs an index-to-name map, and two maps are possible:
| Map | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Kept | Dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 = paper Table 2 order (extracted) | Hardness | Solids | Chloramines | Conductivity | Organic_carbon | Turbidity | Sulfate | Trihalomethanes | ph | Sulfate, THM, Conductivity, Chloramines, Solids, Turbidity, ph | Hardness, Organic_carbon |
| M2 = raw CSV column order | ph | Hardness | Solids | Chloramines | Sulfate | Conductivity | Organic_carbon | Trihalomethanes | Turbidity | Organic_carbon, THM, Chloramines, Solids, Hardness, Conductivity, Turbidity | ph, Sulfate |
Evidence [P]: Table 2 lists the six complete columns first (0-5) and the three filled columns last (6-8), which is what a pipeline that fills one column at a time and re-attaches it would produce, so M1 is the more likely map. M2 would drop ph and Sulfate, two of the three columns the paper spent Sect. 4.2 filling. Neither can be proven from the text. Observation: the kept list is not in ascending index order, so it may be a ranking or a selection order. Observation [P]: Table 2's Sulfate (314-338) and Trihalomethanes (64.7-67.6) ranges are as narrow as near-mean fills would give (our own fills have sd 6.5 and 1.9); the extracted ph range (-51 to 40) is physically impossible and may be garbled.

## 3. Decisions needed (recommended in bold; you can accept all with "accept recommended")
| ID | Decision | Options and effect | Recommended |
|---|---|---|---|
| C1 | Index map | M1, M2, or both | **Primary M1; M2 as a labelled sensitivity run** (cheap: classifier fits take seconds) |
| C2 | Route | (a) paper's list as given; (b) our own FRS implementation; both | **Both**: (a) first, (b) then compared with the paper's list; neither is tuned to reach the paper's accuracy |
| C3a | Definition used | relation-based Eqs. 4-5 (n x n similarity between objects) or equivalence-class form Eqs. 7-8 (needs fuzzy partitions per attribute that the paper never defines) | **Relation-based (Eqs. 4-5)**: scales, no undefined partitions |
| C3b | Similarity per attribute | linear R_a(x,y) = 1 - abs(a(x) - a(y)) / range_a (range from TRAIN rows); Gaussian (needs a width); other | **Linear, parameter-free** |
| C3c | Aggregation over a feature subset | min (Goedel t-norm, matches the paper's min/max operators); Lukasiewicz t-norm | **min** |
| C3d | Implicator | Kleene-Dienes max(1-a,b) (matches Eq. 7's form) or Lukasiewicz min(1,1-a+b) | **Kleene-Dienes; the choice does not matter here**: for crisp decision classes (b is 0 or 1) both reduce to 1-a or 1, so they give identical results |
| C3e | Search and stop | QuickReduct greedy forward (add the feature that raises gamma most; ties by column order); stop when gamma(R) = gamma(all 9); or stop at a tolerance; or force k = 7 (the paper's size) | **Strict stop gamma(R) = gamma(all)**, report the full selection order and the gamma path; ALSO report the 7-feature prefix of the order as a clearly labelled comparison (k taken from the paper, not discovered). Expect that strict stop may keep all or nearly all 9 features (with continuous data gamma rises with almost every feature); if so that is reported as a finding, no tolerance is invented |
| C4 | Data FRS is fit on | S2b train rows (balanced, as Sect. 5.3 says) vs S2a train | **S2b train rows (3196, train only, never test)**; plus a stability check on the 2620 original filled train rows |
Classifiers on the selected features: XGBoost, RF, AdaBoost, library defaults, seed 42, same test rows, same metrics (as before). PyCaret/AutoGluon deferred.

## 4. Closed form used (verified by me)
For crisp decision classes and the relation-based lower approximation, the positive-region membership of object x is mu_POS(x) = 1 - max over objects y of the OTHER class of R_P(x,y), so gamma_P = mean over x of mu_POS(x). I checked this against a brute-force evaluation of the paper's definitions (Eq. 4 with Kleene-Dienes, sup over decision classes, mean) on random data: identical to 10 decimals for 5 subsets; gamma is non-decreasing when features are added (min t-norm). Cost: only cross-class similarity matrices are needed (S2b train: 1598 x 1598 per attribute, about 10 MB each in float32, nine of them under 100 MB; about 45 subset evaluations in QuickReduct; seconds).

## 5. Leakage rules
FRS reads TRAIN rows only (the S2b train file); the test file is never opened by the selection script. Ranges for the similarity are train ranges. The selected column list is then applied to the test rows by name. Synthetic SMOTE rows are included in the FRS input in the primary run (paper Sect. 5.3) and excluded in the stability check; note that synthetic rows lie between real minority rows, so they can change similarities; this is a documented limitation, not a fix. The label is only the decision attribute, never a conditional attribute.

## 6. Step order (one script at a time; each after its own approval, smoke before full)
S3-1 r4_frs_select.py: QuickReduct on S2b train; outputs gamma path, order, reduct, comparison with the paper's list under M1 and M2, stability check; built-in self-test of the closed form against brute force on a small subsample. Smoke = small subsample of the smoke SMOTE file; test file not opened.
S3-2 r4_s3_sklearn_models.py: XGBoost/RF/AdaBoost on S2b train restricted to (i) paper list M1, (ii) paper list M2, (iii) own reduct, (iv) own 7-prefix if different; test = same 656 filled rows with the same columns; direct comparison with S2b (all 9 features).
S3-3 later: PyCaret, AutoGluon (S2 and S3), reports, optional Track P diagnostic.

## 7. S3-1 FULL RESULT (real data, user console 2026-11-10; Track L; reproduction results, not validated performance)
Input s2_smote_train_L.csv (sha256 b9eb679d...cc36), 3196 rows (2620 original + 576 synthetic), test rows used 0, all 6 checks PASS, rerun = EXACT match (same order/gamma file as the first run; reproducibility only).
| Item | Result |
|---|---|
| gamma(all 9) | 0.1112 (low: classes overlap heavily; single-feature gamma 0.0003-0.0005) |
| Our order | Sulfate, Solids, Turbidity, Conductivity, Trihalomethanes, Organic_carbon, Hardness, Chloramines, ph |
| Strict reduct | ALL 9 features (gamma reaches gamma(all) only at the last step) = same feature set as S2b; no separate classifier run is needed for it |
| Our first-7 (size from the paper) | Sulfate, Solids, Turbidity, Conductivity, Trihalomethanes, Organic_carbon, Hardness (drops Chloramines, ph); gamma 0.0888 |
| Paper list, M1 (Table 2 order [P]) | keeps Sulfate, THM, Conductivity, Chloramines, Solids, Turbidity, ph; gamma 0.0878 |
| Paper list, M2 (CSV order) | keeps Organic_carbon, THM, Chloramines, Solids, Hardness, Conductivity, Turbidity; gamma 0.0909 |
| Original-rows-only order (stability) | Hardness, Solids, Turbidity, Organic_carbon, Conductivity, THM, Chloramines, ph, Sulfate; first-7 drops ph, Sulfate |
Reading (mine, [P]):
1. Overlap with the paper's list is NOT evidence of agreement: any two 7-feature subsets of 9 share at least 5 features (expected about 5.4 by chance). Observed 5/7 (M1) and 6/7 (M2). I withdraw the word "encouraging" I used after the 600-row smoke.
2. gamma barely separates the candidate 7-sets (0.0878, 0.0888, 0.0909; gamma(all) 0.1112). With 9 features and heavily overlapping classes, gamma-based selection is nearly arbitrary here.
3. The selection is not stable: with synthetic rows the order starts Sulfate, Solids; on the original rows it starts Hardness, Solids and puts Sulfate last. The two filled-in columns most affected (Sulfate, ph) move from first/last in opposite directions. Hypothesis [P, untested]: our near-constant fills (sd 6.5 for Sulfate vs 41.5 real) and SMOTE interpolation both change similarities.
4. The paper's FRS selection is therefore neither reproduced nor refuted; our implementation differs from theirs in unspecified ways.

## 8. S3-2 plan: r4_s3_sklearn_models.py (PROPOSED, no code yet)
Purpose: effect of the feature set on XGBoost / RandomForest / AdaBoost, everything else identical to S2b (same S2b train rows, library defaults, seed 42, n_jobs 8, single fit, same 656 filled test rows, same metrics; test scored once per fit; nothing selected on test).
| Variant | Features | Role |
|---|---|---|
| S2b-ref | all 9 (re-fit; must reproduce the stored S2b numbers; check) | reference = also our strict FRS reduct |
| S3-paperM1 | paper list under M1 (7) | primary (C1) |
| S3-own7 | our first-7 | primary (our FRS; size from the paper) |
| S3-paperM2 | paper list under M2 (7) | sensitivity |
| S3-stab7 | first-7 of the original-rows-only order | sensitivity (selection stability) |
5 variants x 3 models = 15 fits (seconds). Feature lists are read BY NAME from s3_frs_selected_L.json, which was produced from train rows only; the script checks the json came from the same SMOTE file (sha256 in s3_frs_run_info_L.json). Pre-fit checks: feature list excludes Potability, row_id, is_synthetic; no NaN; test ids = S1 test ids; test hash unchanged; S2b-ref metrics equal the stored S2b metrics (tolerance printed).
Labels: "Section 4.2 fill + SMOTE (train rows only) + fuzzy-rough selection (our implementation / paper list, mapping [P])". Accuracy always beside the 60.98% baseline; add the paper's Table 7 values as reference only (XGBoost 98.83, RF 95.19, AdaBoost 92.21).
Leakage risks: (1) selection must never see test rows: ensured, FRS script has no test input; (2) 5 feature sets are each scored on the same test set: reported as a set of sensitivity results, none chosen by test score, so no test-based model selection; (3) SMOTE rows are in the training side only.
Environment: env_main. Outputs (results/s2/): s3_sklearn_metrics.csv, s3_sklearn_confusion.csv, s3_sklearn_predictions.csv (class-1 probabilities), s3_sklearn_status.csv, s3_sklearn_report.txt, s3_sklearn_run_info.json.
Smoke: same pattern as the S2 script: smoke SMOTE file, evaluate on 300 original train rows outside the smoke input, test file not opened, output to a smoke folder, not reportable.
Optional (needs your yes/no): a paired bootstrap on the saved test predictions (2000 resamples, seed 42, no refitting) giving the uncertainty of each S3-minus-S2b difference in accuracy and ROC-AUC. Recommended YES: it adds no tuning and shows whether differences are within noise (accuracy SE is about 1.9 points).

## 9. S3-2 FULL RESULT (real data, user console 2026-11-10; Track L; reproduction results, not validated performance)
656 test rows, majority baseline 60.98%, accuracy SE about 1.9 points. All 19 checks PASS (incl. test ids = S1 ids; test hash unchanged; S2b-ref reproduces stored S2b probabilities, max diff 1.1e-16). Rerun: EXACT match (metrics and predictions; reproducibility only). Label: "Section 4.2 fill + SMOTE (train rows only) + ..." as in section 8.
| Variant (features) | Model | Accuracy | Recall | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| S2b-ref (all 9) | XGBoost / RF / AdaBoost | 61.59 / 64.18 / 58.54 | .539 / .516 / .430 | .665 / .723 / .685 | .633 / .673 / .575 |
| S3-paperM1 (drops Hardness, Organic_carbon) | XGBoost / RF / AdaBoost | 58.99 / 60.98 / 49.85 | .523 / .531 / .606 | .633 / .660 / .430 | .622 / .638 / .566 |
| S3-own7 (drops Chloramines, ph) | XGBoost / RF / AdaBoost | 59.15 / 55.79 / 56.55 | .477 / .391 / .398 | .665 / .665 / .673 | .573 / .576 / .542 |
| S3-paperM2 = S3-stab7 (drops ph, Sulfate) | XGBoost / RF / AdaBoost | 52.29 / 55.49 / 46.19 | .445 / .387 / .688 | .573 / .663 / .318 | .517 / .540 / .518 |
S3-paperM2 and S3-stab7 have the IDENTICAL feature set (the 7 columns kept by the paper's list under M2 = the first 7 of our original-rows-only order), so their rows are identical; this is not a bug.
Paper Table 7 (reference only): XGBoost 98.83, RF 95.19, AdaBoost 92.21 accuracy. Our best S3 accuracy is 59.15 / 60.98 / 56.55: a gap of at least 35 points. NOT reproduced.
Paired bootstrap (2000 resamples, this test split only, 24 intervals, no multiplicity correction): relative to S2b-ref, M1 differences are mostly inside noise (exceptions: RF AUC -0.034, AdaBoost accuracy -0.087); own7 is lower in RF (accuracy -0.084, AUC -0.096) and XGBoost AUC (-0.061); the ph+Sulfate-dropping set is clearly lower for every model (AUC -0.057 to -0.133).
Reading (mine, [P] unless stated):
1. No fuzzy-rough variant beat the all-features reference on any model in accuracy or AUC [C: from the table]. Selection did not help in our implementation.
2. The loss tracks which features are dropped: sets that keep ph and Sulfate (M1) stay near S2b-ref; dropping ph alone (own7) costs about 0.06-0.10 AUC; dropping both gives AUC 0.52-0.54, close to chance. Hypothesis [P, untested]: in this data most of the weak signal is in ph and Sulfate. gamma did not notice this (it ranked Sulfate first on the SMOTE rows but last on original rows), i.e. gamma does not track predictive value here.
3. Coincidence to flag honestly: the first-7 set found on the original rows only equals the paper's list under M2 (drops ph, Sulfate). Chance of matching a given 2-feature drop is 1/36 (about 2.8%), but this was noticed after the fact among several comparisons and the algorithm is our own choice, so it is a curiosity, not evidence that M2 is the paper's mapping. Against M2 [P]: if the paper dropped ph and Sulfate it would have removed the most informative features in our data.
4. Possible reasons for the huge gap to the paper (all untested hypotheses): fill or SMOTE or selection done before the split (leakage) in the paper; different data handling; label involvement in the fill. Track P (paper-literal, leaky, diagnostic only) is the cheap way to test the first.
