# Water Potability Classification: Complete Project Report

> **Project**: AutoML Comparison, Hyperparameter Optimisation & Explainability (Water Potability Case Study) + Reproduction of Shehab et al. (2025)
> 
> **Researchers**: Nusrat Sharmin, Nafiz Al Anjim, Umme Hani Punam, M.Sabbir Hasnat Shaon
> 
> **Status Labels**: [C] = Confirmed (verified from outputs), [P] = Provisional, [U] = Unverified

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Environment & Infrastructure](#3-environment--infrastructure)
4. [Main Workstream: AutoML Comparison & Preprocessing](#4-main-workstream-automl-comparison--preprocessing)
   - [4.1 Phase 0: Data Preparation](#41-phase-0-data-preparation)
   - [4.2 Lesson 0: Missing Data Analysis](#42-lesson-0-missing-data-analysis)
   - [4.3 Lesson 1: Imputation Comparison](#43-lesson-1-imputation-comparison)
   - [4.4 Lesson 2: Scaling Experiment](#44-lesson-2-scaling-experiment)
   - [4.5 Lesson 3: Imbalance Handling](#45-lesson-3-imbalance-handling)
   - [4.6 Lesson 3b: GAN-Based Resampling](#46-lesson-3b-gan-based-resampling)
   - [4.7 Phase 1 v2: AutoGluon vs PyCaret (10-Split)](#47-phase-1-v2-autogluon-vs-pycaret-10-split)
   - [4.8 Phase 1 v3: Preprocessing Factorial (4 Combos)](#48-phase-1-v3-preprocessing-factorial-4-combos)
   - [4.9 Like-for-Like Tool Comparison](#49-like-for-like-tool-comparison)
   - [4.10 Preprocessing Decisions Summary](#410-preprocessing-decisions-summary)
5. [Reproduction Workstream: Shehab et al. (2025)](#5-reproduction-workstream-shehab-et-al-2025)
   - [5.1 Paper Overview & Suspected Problems](#51-paper-overview--suspected-problems)
   - [5.2 Data Verification](#52-data-verification)
   - [5.3 Scenario 1 — Raw CSV (Original Data)](#53-scenario-1--raw-csv-original-data)
   - [5.4 Scenario 2 — Correlation-Based Imputation ± SMOTE](#54-scenario-2--correlation-based-imputation--smote)
   - [5.5 Scenario 3 — Fuzzy-Rough Feature Selection](#55-scenario-3--fuzzy-rough-feature-selection)
   - [5.6 Reproduction Gap Summary](#56-reproduction-gap-summary)
6. [Key Findings](#6-key-findings)
7. [Limitations & Caveats](#7-limitations--caveats)
8. [Future Work (Not Started)](#8-future-work-not-started)
9. [Code & File Inventory](#9-code--file-inventory)
10. [References](#10-references)

---

## 1. Project Overview

This project comprises **two parallel workstreams** on the Kaggle Water Potability dataset:

### Workstream A: AutoML Comparison → HPO → XAI (Main Project)

**Goal**: Compare AutoGluon and PyCaret on a binary classification task, select the best model, tune it with multiple hyperparameter optimisation strategies (Grid, Random, Bayesian/Optuna, Genetic), and apply Explainable AI (SHAP) on the tuning results. The target is a Q1 journal publication.

**Research Framing** [P]: *"How does HPO optimiser choice change SHAP explanations/stability, across domains?"*

| Phase | Content | Status |
|-------|---------|--------|
| Phase 0 | Data preparation, stratified 80/20 split | **DONE** [C] |
| Phase 1 v1 | Single-seed AutoGluon/PyCaret tests | Superseded |
| Phase 1 v2 | 10-split baselines + AG + PC comparison | **DONE** [C] (PyCaret had bug) |
| Lessons 0–3b | One-factor-at-a-time preprocessing | **DONE** [C] |
| Phase 1 v3 | 4-combo factorial: PC and AG with chosen preprocessing | **DONE** [C] |
| Phase 2 | CatBoost HPO (Grid, Random, Bayesian, Genetic) | **NOT STARTED** |
| Phase 3 | XAI / SHAP | **NOT STARTED** |

### Workstream B: Reproduction of Shehab et al. (2025)

**Goal**: Reproduce the results of *"Shehab, Darwish, Hassanien (2025). Environment, Development and Sustainability, 27:1401–1419"* which reports XGBoost 98.83%, RF 95.19%, and AdaBoost 92.21% accuracy on the same dataset using a pipeline of correlation-based imputation → SMOTE → fuzzy-rough feature selection.

| Scenario | Content | Status |
|----------|---------|--------|
| S1 | Original raw CSV (no preprocessing) | **CLOSED** [C] |
| S2a | Fill missing values only (correlation-based, Track L) | **DONE** [C] |
| S2b | Fill + SMOTE (Track L, leak-free) | **DONE** [C] |
| S3 | Fill + SMOTE + Fuzzy-rough selection (Track L) | **DONE** [C] — sklearn only |
| PyCaret/AG S2+S3 | PyCaret and AutoGluon on S2/S3 | **DEFERRED** by user |

> [!IMPORTANT]
> **Core Finding**: The paper's reported accuracies (92–99%) were **NOT reproduced**. Our leak-free results stay at 58–69% accuracy across all scenarios and all models. The gap is ≥35 percentage points. The most likely explanation is data leakage in the paper's pipeline (preprocessing before the train/test split).

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| **Source** | Kaggle Water Potability dataset (Aditya Kadiwal, 2020) |
| **Citation** | `https://doi.org/10.34740/KAGGLE/DSV/1548804` |
| **File** | `data/raw.csv` (= `repro/paper/raw_water.csv`), 3276 rows × 10 columns |
| **Target** | `Potability` — binary (1 = potable, 0 = not potable) |
| **Class distribution** | 0: 1998 (61.0%), 1: 1278 (39.0%) — imbalance ratio 1.56:1 |
| **Duplicates** | 0 |

### Features (all continuous, float)

| Feature | Missing (%) | Range | Notes |
|---------|-------------|-------|-------|
| ph | 491 (14.99%) | 0.00 – 14.00 | Dataset text says 6.52–6.83; actual data spans 0–14 [U] |
| Hardness | 0 | — | — |
| Solids | 0 | ~ tens of thousands | Dominates SMOTE distance (99.99% of variance) |
| Chloramines | 0 | — | — |
| Sulfate | 781 (23.84%) | — | Strongest pairwise correlation with Solids (r = −0.20) |
| Conductivity | 0 | — | — |
| Organic_carbon | 0 | — | — |
| Trihalomethanes | 162 (4.95%) | — | — |
| Turbidity | 0 | — | — |

- **Rows with ≥1 missing value**: 1265 (38.6%)
- **Complete rows**: 2011 (61.4%)
- Missingness is MCAR-like: blank-flag correlations with Potability are near zero (chi² p > 0.15 for all three columns)

### Train/Test Split (Primary, used throughout)

| Set | N | Class 0 | Class 1 | Majority Baseline |
|-----|---|---------|---------|-------------------|
| **Train** | 2620 | 1598 (61.0%) | 1022 (39.0%) | — |
| **Test** | 656 | 400 (61.0%) | 256 (39.0%) | **60.98%** |

Split method: `stratified 80/20, random_state=42`.

> [!NOTE]
> All results use accuracy beside the **60.98% majority baseline**. Any model near this baseline is essentially guessing the majority class.

---

## 3. Environment & Infrastructure

### Hardware

| Component | Detail |
|-----------|--------|
| Machine | Lenovo IdeaPad Slim 3 15IRH8 |
| CPU | 13th-gen Intel i5-H, 8 cores / 12 threads |
| RAM | 16 GB |
| OS | Linux Mint |
| GPU | None (CPU only) |

### Software Environments

| Environment | Python | Key Libraries |
|-------------|--------|---------------|
| **env_main** | 3.11.17 | AutoGluon 1.6.3, scikit-learn 1.9.1, XGBoost 3.2.0, LightGBM 4.7.0, CatBoost 1.2.10, imbalanced-learn 0.14.2, PyTorch 2.13.0, pandas 2.3.3, numpy 2.4.6 |
| **env_pycaret** | 3.11.17 | PyCaret 3.3.2, scikit-learn 1.4.2, pandas 2.1.4, numpy 1.26.4 |

> [!WARNING]
> env_pycaret uses **scikit-learn 1.4.2** (predates NaN support in many models), while env_main uses **scikit-learn 1.9.1**. This affects which models can run on raw NaN data and may cause slight numerical differences in shared algorithms (e.g., RandomForest).

---

## 4. Main Workstream: AutoML Comparison & Preprocessing

### 4.1 Phase 0: Data Preparation

**Script**: [phase0_prepare.py](file:///home/sabbir/research/phase0_prepare.py)
**Output**: `data_cls/train.csv` (2620 rows), `data_cls/test.csv` (656 rows), `data_cls/phase0_report.txt`

- Stratified 80/20 split, seed 42
- No preprocessing applied at this stage
- Two features flagged as "ID-like" (Solids, Conductivity) — false alarms; all 9 features kept
- Train missing: ph 387 (14.8%), Sulfate 625 (23.9%), THM 134 (5.1%); 1013 rows (39%) with ≥1 blank

---

### 4.2 Lesson 0: Missing Data Analysis

**Script**: [lesson0_look_at_missing.py](file:///home/sabbir/research/lessons/lesson0_look_at_missing.py)

Confirmed that missingness is not strongly associated with the target. Chi-square p-values for blank vs present potability rates: ph p=0.286, Sulfate p=0.288, THM p=0.158. Blank-flag correlations near zero.

---

### 4.3 Lesson 1: Imputation Comparison

**Script**: [lesson1_imputation.py](file:///home/sabbir/research/lessons/lesson1_imputation.py) — `RepeatedStratifiedKFold(5 folds × 4 repeats = 20 evaluations)` on train.csv only; test.csv never opened.

#### Results (RF model, mean AUC over 20 folds)

| Imputer | RF AUC | HGB AUC | Notes |
|---------|--------|---------|-------|
| Iterative (BayesianRidge, 10 iter) | 0.6851 | 0.6637 | — |
| Native (leave NaN) | 0.6848 | 0.6623 | — |
| Mean | **0.6841** | **0.6685** | ← **Chosen** |
| Median | 0.6829 | 0.6677 | — |
| Median + indicator | 0.6795 | 0.6647 | — |
| KNN (k=5, min-max) | 0.6785 | 0.6600 | — |
| Zero fill | 0.6674 | 0.6458 | p=0.047 (RF) |
| Hot deck | 0.6651 | 0.6425 | p=0.037 (RF) |

> **Decision**: Mean imputation chosen — simplest option tied with the top six (difference < 0.005 AUC, p ≥ 0.48); works with SMOTE (no blanks); PyCaret's default. Hot deck and zero fill tested and rejected (slightly worse, not significant after Holm correction).

---

### 4.4 Lesson 2: Scaling Experiment

**Script**: [lesson2_scaling.py](file:///home/sabbir/research/lessons/lesson2_scaling.py)

| Model | No Scaling AUC | Min-Max AUC | Effect |
|-------|----------------|-------------|--------|
| RF | 0.6841 | 0.6841 | None (p=0.93) |
| KNN | 0.5095 | **0.6324** | **Huge** (p=0.000) |
| SVM-RBF | 0.5235 | **0.6903** | **Huge** (p=0.000) |
| MLP | 0.4913 | **0.6492** | **Huge** (p=0.003) |
| LogReg | 0.4802 | 0.4805 | None (at chance) |

> **Decision**: No scaling for tree-based models; min-max scaling only when using KNN/SVM/NN. LogReg is at chance regardless → no linear signal in these features.

---

### 4.5 Lesson 3: Imbalance Handling

**Script**: [lesson3_imbalance.py](file:///home/sabbir/research/lessons/lesson3_imbalance.py) — Pipeline: mean imputer → min-max → fix → model, inside CV folds.

#### Results (RF model, 20 CV folds)

| Fix | AUC | Accuracy | Bal. Acc | F1 |
|-----|-----|----------|---------|-----|
| None | 0.6841 | **0.6737** | 0.6139 | 0.4490 |
| Class weights | 0.6878 | 0.6635 | **0.6304** | 0.5257 |
| Threshold moving | 0.6841 | 0.6421 | 0.6252 | 0.5424 |
| SMOTE | 0.6835 | 0.6554 | 0.6271 | 0.5292 |
| SMOTE-Tomek | 0.6854 | 0.6517 | 0.6220 | 0.5207 |
| Borderline-SMOTE | 0.6810 | 0.6503 | 0.6236 | 0.5279 |
| ADASYN | 0.6806 | 0.6425 | 0.6218 | 0.5347 |
| Random oversample | 0.6824 | 0.6640 | 0.6192 | 0.4905 |
| Random undersample | 0.6735 | 0.6219 | 0.6190 | 0.5553 |
| SMOTE-ENN | 0.6363 | 0.5770 | 0.5978 | 0.5607 |

> **Findings**:
> - No fix changes AUC meaningfully (all p > 0.18) except SMOTE-ENN (−0.048, p=0.007) which **hurts**
> - SMOTE-ENN dropped
> - F1 gains from SMOTE/weights/threshold are mainly a decision-boundary shift, not improved discrimination
> - Accuracy is highest with NO fix
> - **Decision**: SMOTE carried forward for AutoML experiments (mild imbalance), but AUC is the primary metric and is unaffected

---

### 4.6 Lesson 3b: GAN-Based Resampling

**Script**: [lesson3b_gan.py](file:///home/sabbir/research/lessons/lesson3b_gan.py) — Tested per supervisor's request.

| Fix | RF AUC | SVM AUC | Notes |
|-----|--------|---------|-------|
| None | 0.6841 | 0.6903 | Baseline |
| Class weights | 0.6878 | 0.6845 | — |
| SMOTE | 0.6835 | 0.6778 | — |
| SMOTified-GAN | 0.6674 | 0.6676 | **Worse** (p=0.048 RF, p=0.010 SVM) |
| GANified-SMOTE | 0.6639 | 0.6629 | **Worse** |
| GAN only | 0.6639 | 0.6622 | **Worse** |

> **Finding**: All three GAN variants **hurt** AUC by 0.017–0.028 points. **Negative result** — consistent with Yang et al. 2025 (SMOTE outperforms GANs for tabular data). GAN hybrids are designed for extremely imbalanced data (IR > 5); our 61/39 split is mild.
>
> **Caveat**: GAN trained with only 200 epochs and lr=2e-4 (papers use ~1000–2000 epochs with lr=1e-5). Not carried forward.

---

### 4.7 Phase 1 v2: AutoGluon vs PyCaret (10-Split)

**Scripts**: [run_autogluon.py](file:///home/sabbir/research/v2/run_autogluon.py), [run_pycaret.py](file:///home/sabbir/research/v2/run_pycaret.py), [run_baselines.py](file:///home/sabbir/research/v2/run_baselines.py), [analyze.py](file:///home/sabbir/research/v2/analyze.py)

**Design**: 10 stratified 80/20 splits (seeds 0–9) of raw.csv. AutoGluon settings: `medium_quality`, `num_bag_folds=5`, `num_stack_levels=0`, `time_limit=300`, `num_cpus=8`. PyCaret: `compare_models(sort="AUC", budget_time=5)`.

#### Results (10 splits, mean ± std, test AUC)

| Method | Test AUC | CV AUC | Acc | F1 | Bal. Acc | MCC |
|--------|----------|--------|------|-----|---------|------|
| AutoGluon A_all | **0.6942 ± 0.0192** | 0.7004 (OOF) | 0.6639 | 0.5047 | 0.6308 | 0.2899 |
| Default RF (500 trees, median) | 0.6880 ± 0.0222 | 0.6718 | 0.6630 | 0.4519 | 0.6131 | 0.2709 |
| PyCaret A_all | 0.6751 ± 0.0249 | 0.6753 | 0.6572 | 0.4551 | 0.6138 | 0.2708 |
| Default HGB | 0.6612 ± 0.0232 | 0.6556 | 0.6548 | 0.4771 | 0.6079 | 0.2354 |
| LogReg (median + scale) | 0.5003 ± 0.0151 | 0.5033 | 0.5006 | 0.0054 | 0.5012 | 0.0176 |
| Dummy | 0.5000 ± 0.0000 | 0.5000 | 0.6097 | 0.0000 | 0.5000 | 0.0000 |

- **AG winners**: NeuralNetTorch 9/10 times, LightGBMXT 1/10
- **PC winners**: QDA 6/10, CatBoost 4/10

> [!CAUTION]
> **BUG FOUND**: PyCaret's `setup()` default `train_size=0.7` meant it only trained/CV'd on **70% of supplied training rows** (840 of 1200). The v2 PyCaret numbers (0.6751) are **invalid** and the paired test (PC−AG = −0.0191, p=0.053) is **RETRACTED**. Fixed in v3 with `test_data=test`.

---

### 4.8 Phase 1 v3: Preprocessing Factorial (4 Combos)

**Scripts**: [run_pycaret_preproc.py](file:///home/sabbir/research/v3/run_pycaret_preproc.py), [run_autogluon_preproc.py](file:///home/sabbir/research/v3/run_autogluon_preproc.py)

Combos tested: `imputer|scaler|resampler` = `mean|none|none`, `mean|minmax|none`, `mean|minmax|smote`, `mean|minmax|smotetomek`.

#### PyCaret v3 (bug-fixed, all train rows, 10 splits)

| Combo | CV AUC | Test AUC | Acc | Bal. Acc | F1 | MCC |
|-------|--------|----------|-----|---------|-----|------|
| mean\|none\|none | 0.6871 | **0.6959 ± 0.0188** | 0.6767 | 0.6208 | 0.4688 | 0.2853 |
| mean\|minmax\|none | 0.6865 | 0.6949 ± 0.0199 | 0.6764 | 0.6206 | 0.4688 | 0.2847 |
| mean\|minmax\|smotetomek | 0.6828 | 0.6892 ± 0.0235 | 0.6462 | 0.6326 | 0.5569 | 0.2635 |
| mean\|minmax\|smote | 0.6842 | 0.6836 ± 0.0212 | 0.6482 | 0.6330 | 0.5555 | 0.2651 |

- **PC v3 winners**: CatBoost 31/40 runs, QDA 9/40
- SMOTE variants: AUC unchanged, **accuracy drops ~3 points**, F1 gains ~0.09

#### AutoGluon v3 (10 splits)

| Combo | Test AUC | Acc | Bal. Acc | F1 | MCC |
|-------|----------|-----|---------|-----|------|
| mean\|minmax\|smote | 0.6816 ± 0.0223 | 0.6328 | 0.6265 | 0.5585 | 0.2494 |
| mean\|none\|none | 0.6814 ± 0.0215 | 0.6639 | 0.6223 | 0.4980 | 0.2687 |
| mean\|minmax\|none | 0.6770 ± 0.0226 | 0.6648 | 0.6209 | 0.4900 | 0.2690 |
| mean\|minmax\|smotetomek | 0.6714 ± 0.0180 | 0.6271 | 0.6166 | 0.5419 | 0.2313 |

- **AG v3 winners**: NeuralNetTorch 25/40, FastAI 5, LightGBMXT 4, CatBoost 4

> **Conclusions**: SMOTE/Tomek leave AUC unchanged, cost ~3–4 accuracy points, and boost F1 by ~0.06–0.09. Min-max has no effect for tree-based winners.

---

### 4.9 Like-for-Like Tool Comparison

**Script**: [compare_tools.py](file:///home/sabbir/research/v3/compare_tools.py) — Compares AG v2 (all train rows) with PC v3 `mean|none|none` (bug-fixed, all train rows) on the same 10 splits.

```
AutoGluon (v2) test AUC: 0.6942 ± 0.0192
PyCaret   (v3) test AUC: 0.6959 ± 0.0188
mean difference (PyCaret − AutoGluon): +0.0017
95% CI [-0.0040, +0.0069]
Nadeau–Bengio corrected resampled t-test p = 0.760
```

> **Conclusion**: AutoGluon and PyCaret are **statistically tied** (p = 0.76). The earlier claim of "AG > PC" was confounded by the PyCaret 70%-rows bug.

---

### 4.10 Preprocessing Decisions Summary

| Decision | Choice | Evidence |
|----------|--------|----------|
| Imputer | **Mean** | Six imputers tied (AUC diff < 0.005, p ≥ 0.48); simplest; SMOTE-compatible |
| Rejected imputers | Hot deck, zero-fill | −0.015 to −0.025 AUC; consistent across models |
| Scaling | **None for trees**; min-max for KNN/SVM/NN | Lesson 2: trees invariant; distance models need scaling |
| Imbalance fix | **None for AUC/accuracy**; class weights if bal_acc/F1 matter | Lessons 3, 3b, v3: no AUC change |
| SMOTE-ENN | **Dropped** | AUC −0.05 |
| GAN hybrids | **Not carried forward** | Negative result (AUC −0.017 to −0.028) |
| AG ≈ PC | **Tied** | p = 0.76 |
| Phase 2 model | **CatBoost** | Won 31/40 PC runs; standalone, ≥4 tunable HPs; post-hoc rule (disclose) |
| Honest ceiling | **AUC ≈ 0.69–0.70, Accuracy ≈ 67%** | Do NOT promise ≥80% |

---

## 5. Reproduction Workstream: Shehab et al. (2025)

### 5.1 Paper Overview & Suspected Problems

**Paper**: Shehab, Darwish, Hassanien (2025). *"Environment, Development and Sustainability"*, 27:1401–1419. DOI: 10.1007/s10668-023-03916-4.

**Claimed Pipeline**: Kaggle water data → fill nulls (Sect. 4.2, correlation-based ML + tuning) → SMOTE → fuzzy-rough feature selection → XGBoost / RF / AdaBoost. 80/20 split.

**Claimed Results** (Table 7, Scenario 3):
- XGBoost: **98.83%** accuracy
- RandomForest: **95.19%** accuracy
- AdaBoost: **92.21%** accuracy

#### Suspected Problems (Flagged, Not Fixed)

| Issue | Description |
|-------|-------------|
| **Leakage** | Split/SMOTE/selection order not stated; if done before the split → severe leakage/inflation |
| **Label in fill** | Imputation predictors unspecified; if target was used → leakage (cf. Kaggle notebook autopsy showing 80% accuracy from label-based fill alone) |
| **Class counts** | Balanced 1508/1508 doesn't match SMOTE of 1998/1278 → should be 1998/1998; class 1 count 1277 vs 1278 in data |
| **Accuracy gap** | 98.83% vs honest ceiling ~67% on the same data |
| **Garbled tables** | Table 2 ranges/order garbled; confusion-matrix digits run together |
| **Missing details** | No hyperparameters, seeds, CV details, library/software specified |
| **Inconsistencies** | FN/FP counts don't reconcile with reported accuracy; phase numbering inconsistent (0/1/2/4) |

---

### 5.2 Data Verification

**Script**: [r1_check_data.py](file:///home/sabbir/research/repro/paper/code/r1_check_data.py)
**Result**: [r1_data_check.txt](file:///home/sabbir/research/repro/paper/results/r1_data_check.txt)

- 3276 × 10, no duplicates
- Classes: 0:1998 / 1:1278 — matches the paper's Table 4
- Missing: ph 14.99%, Sulfate 23.84%, THM 4.95% — matches the paper
- ph spans 0–14 (provenance concern)

---

### 5.3 Scenario 1 — Raw CSV (Original Data)

**Protocol**: [S1_PROTOCOL.md](file:///home/sabbir/research/repro/paper/S1_PROTOCOL.md) — No fill, no drop, no scale, no resample, no feature selection. Library defaults, seed 42, single 80/20 split. Models scored once on the test set. Nothing selected on test.

**Label**: `Raw CSV / NaN-native`

#### S1 sklearn Results (env_main)

| Model | TN | FP | FN | TP | Acc % | Prec % | Recall % | Spec % | F1 % | ROC-AUC | MCC | Paper Acc % |
|-------|-----|-----|-----|-----|-------|--------|----------|--------|------|---------|------|------------|
| XGBoost | 317 | 83 | 163 | 93 | 62.50 | 52.84 | 36.33 | 79.25 | 43.06 | 0.6378 | 0.1715 | 67.98 |
| RandomForest | 353 | 47 | 174 | 82 | 66.31 | 63.57 | 32.03 | 88.25 | 42.60 | 0.6513 | 0.2489 | 68.60 |
| AdaBoost | — | — | — | — | NOT RUNNABLE | — | — | — | — | — | — | 62.50 |

- AdaBoost raised `ValueError: Input X contains NaN` (sklearn 1.9.1). Status: **NOT REPRODUCIBLE** under raw-CSV protocol.
- RF gap from paper: 2.3 points (within noise for SE ≈ 1.9 points)
- XGBoost gap: 5.5 points (borderline)

#### S1 PyCaret Results (env_pycaret)

| Model | Role | CV AUC | Test Acc % | Test ROC-AUC | Bal. Acc | F1 % | MCC |
|-------|------|--------|------------|--------------|---------|------|------|
| rf | Primary | 0.6297 | 65.40 | 0.6445 | 0.5939 | 41.94 | 0.2266 |
| xgboost | Primary | 0.6115 | 62.50 | 0.6378 | 0.5779 | 43.06 | 0.1715 |
| catboost | Exploratory | 0.6204 | 65.85 | 0.6662 | 0.5984 | 42.56 | 0.2380 |
| lightgbm | Exploratory | 0.6180 | 66.31 | 0.6627 | 0.6155 | 48.00 | 0.2576 |
| dt | Exploratory | 0.5512 | 57.62 | 0.5562 | 0.5562 | 46.12 | 0.1120 |
| dummy | Exploratory | 0.5000 | 60.98 | 0.5000 | 0.5000 | 0.00 | 0.0000 |
| ada + 12 others | Not runnable | — | — | — | — | — | — |

- 19 models tried: **6 runnable**, **13 not runnable** (all due to `Input X contains NaN`; sklearn 1.4.2 predates NaN support)
- PyCaret xgboost test numbers = IDENTICAL to env_main XGBoost run (same defaults, same data)

#### S1 AutoGluon Results (env_main, AutoGluon 1.6.3)

| Model | Val AUC (OOF) | Test Acc % | Test ROC-AUC | Bal. Acc | F1 % | MCC |
|-------|---------------|------------|--------------|---------|------|------|
| WeightedEnsemble_L2 ★ | 0.7285 | 67.23 | 0.6662 | 0.6082 | 42.97 | 0.2722 |
| NeuralNetFastAI ★★ | 0.7103 | 65.40 | 0.6530 | 0.5974 | 43.39 | 0.2291 |
| NeuralNetTorch | 0.7064 | 65.40 | 0.6558 | 0.6059 | 46.59 | 0.2362 |
| LightGBMXT | 0.7031 | 65.55 | 0.6614 | 0.5980 | 43.22 | 0.2321 |
| CatBoost | 0.6845 | 66.16 | 0.6655 | 0.5995 | 42.19 | 0.2448 |
| LightGBMLarge | 0.6795 | 69.05 | 0.6752 | 0.6211 | 43.45 | 0.3247 |
| LightGBM | 0.6789 | 68.75 | 0.6807 | 0.6221 | 44.74 | 0.3133 |
| RandomForestEntr | 0.6777 | 66.01 | 0.6603 | 0.5954 | 40.85 | 0.2399 |
| XGBoost | 0.6758 | 67.53 | **0.6875** | 0.6051 | 40.67 | 0.2818 |
| ExtraTreesGini | 0.6755 | 64.63 | 0.6348 | 0.5645 | 29.70 | 0.1988 |
| RandomForestGini | 0.6721 | 66.16 | 0.6481 | 0.5988 | 41.88 | 0.2446 |
| ExtraTreesEntr | 0.6719 | 65.70 | 0.6218 | 0.5767 | 32.43 | 0.2325 |

★ = best overall by validation. ★★ = best non-ensemble by validation. Settings: `medium_quality`, `num_bag_folds=5`, `num_stack_levels=0`, `time_limit=300`, `num_cpus=8`, `eval_metric=roc_auc`.

- Validation AUC (0.67–0.73) is higher than every test AUC (0.62–0.69) — OOF optimism
- All models land 3–8 points above the 60.98% baseline
- AutoGluon's NaN handling is internal (RF/XT fill 0, NN median, GBMs native)

> **S1 is CLOSED. All three tools' runs are reproducible (rerun = EXACT match).**

---

### 5.4 Scenario 2 — Correlation-Based Imputation ± SMOTE

**Protocol**: [S2_PROTOCOL.md](file:///home/sabbir/research/repro/paper/S2_PROTOCOL.md)

#### Design Decisions

| ID | Decision |
|----|----------|
| B1 | Imputation model: Random Forest regressor |
| B2 | Tuning: grid search (12 combos: n_estimators {100,300} × max_depth {None,10} × min_samples_leaf {1,5,25}) + 5-fold CV on non-null train rows |
| B3 | Predictor rule: Pearson r > 0 with the target column (paper's literal wording); correlations from pairwise-complete TRAIN rows |
| B4 | Potability NOT used as a predictor (leak-free) |
| B5 | Fill order: fewest missing first → Trihalomethanes → ph → Sulfate |
| B6 | Two tracks: **Track L** (leak-free, PRIMARY): split first, fit on train only. Track P (paper-literal, leaky, DIAGNOSTIC) |
| B7 | Fill once on the split, not inside CV folds |
| B10 | SMOTE after split on train rows only (Track L) |
| B11 | S2a = fill only; S2b = fill + SMOTE — both co-primary |
| B12 | SMOTE: imblearn defaults (k=5), seed 42, balance to majority count |

#### Step S2-1: Correlation Analysis (Train Rows Only)

**Script**: [r3_corr_table.py](file:///home/sabbir/research/repro/paper/code/r3_corr_table.py)

All pairwise |r| ≤ 0.20. Positive correlations used as predictors:

| Column to Fill | Predictors (r > 0) | r values |
|---------------|--------------------|---------| 
| Trihalomethanes | Chloramines, Organic_carbon, Conductivity | 0.009, 0.005, 0.0004 |
| ph | Hardness, Organic_carbon, Conductivity, THM | 0.064, 0.057, 0.027, 0.0005 |
| Sulfate | Chloramines, ph, Organic_carbon | 0.040, 0.034, 0.033 |

> **Critical finding**: The strongest relation in the data is Solids ↔ Sulfate (r = −0.20), which the "positive correlation" rule **excludes**. All positive correlations are ≤ 0.064.

#### Step S2-3: Track L Fill Results

**Script**: [r3_fill_trackL.py](file:///home/sabbir/research/repro/paper/code/r3_fill_trackL.py)

| Column | Best Params | CV R² vs Mean | Fills Nearly Constant? |
|--------|-------------|---------------|----------------------|
| Trihalomethanes | depth=10, leaf=25, trees=300 | **−0.0222** | Yes (sd 1.87 vs 16.22 observed) |
| ph | depth=None, leaf=25, trees=300 | **+0.0109** | Yes (sd 0.31 vs 1.59 observed) |
| Sulfate | depth=10, leaf=25, trees=300 | **−0.0159** | Yes (sd 6.47 vs 41.53 observed) |

> **Key finding**: The tuned RF fill model is **NOT better than predicting the column mean** for THM and Sulfate (negative CV R²). The fills are essentially near-constant values at the column mean with dramatically reduced variance. This is expected given r ≤ 0.064.
>
> **Grid edge limitation**: `min_samples_leaf=25` (largest in grid) was chosen for ALL three columns — an even smoother model might score marginally better, but CV R² is already ~0.

#### Step S2-4: SMOTE Data Preparation

**Script**: [r3_make_smote_trackL.py](file:///home/sabbir/research/repro/paper/code/r3_make_smote_trackL.py)

- 2620 train rows (1598/1022) → 3196 rows (1598/1598) after SMOTE
- 576 synthetic minority rows added
- All checks PASS; rerun EXACT

> [!WARNING]
> **SMOTE Distance Dominance**: Solids carries **99.99%** of the squared-distance variance (sd = 9129 vs next-highest Conductivity sd = 137). SMOTE neighbours are chosen almost entirely by Solids alone; the other 8 features are effectively random interpolations between Solids-neighbours. This is a documented limitation of unscaled SMOTE (imblearn defaults, as per B12).

#### S2 sklearn Results — Fill Only (S2a) vs Fill + SMOTE (S2b)

**Script**: [r3_s2_sklearn_models.py](file:///home/sabbir/research/repro/paper/code/r3_s2_sklearn_models.py) — Test = 656 filled original rows, never resampled.

| Variant | Model | Acc % | vs Baseline | Recall | Specificity | F1 | ROC-AUC | MCC |
|---------|-------|-------|------------|--------|-------------|-----|---------|------|
| **S2a** (fill only) | XGBoost | 64.02 | +3.04 | 0.375 | 0.810 | 0.449 | 0.622 | 0.205 |
| S2a | RandomForest | **68.45** | +7.47 | 0.340 | 0.905 | 0.457 | **0.657** | 0.304 |
| S2a | AdaBoost | 61.43 | +0.45 | 0.059 | 0.970 | 0.106 | 0.577 | 0.070 |
| **S2b** (fill+SMOTE) | XGBoost | 61.59 | +0.61 | 0.539 | 0.665 | 0.523 | 0.633 | 0.202 |
| S2b | RandomForest | 64.18 | +3.20 | 0.516 | 0.723 | 0.529 | 0.673 | 0.241 |
| S2b | AdaBoost | 58.54 | −2.44 | 0.430 | 0.685 | 0.447 | 0.575 | 0.117 |

**Paper Table 6** (reference only): XGBoost 95.88%, RF 90.24%, AdaBoost 91.31%.
**Paper Table 7** (reference only): XGBoost 98.83%, RF 95.19%, AdaBoost 92.21%.

> **Observations**:
> 1. SMOTE shifts the operating point (recall +16–37 pts, specificity −15–29 pts) but AUC barely changes (−0.002 to +0.016)
> 2. All accuracies: 58.5%–68.5% (−2.4 to +7.5 pts around baseline)
> 3. **Paper gap**: Our leak-free S2a/S2b are **27+ points lower** for every model. Far beyond the ~1.9-point SE.
> 4. AdaBoost is at or below baseline, almost never predicts class 1 in S2a (recall 0.059)

---

### 5.5 Scenario 3 — Fuzzy-Rough Feature Selection

**Protocol**: [S3_PROTOCOL.md](file:///home/sabbir/research/repro/paper/S3_PROTOCOL.md)

#### Paper's Specification (Incomplete)

The paper cites the Jensen-Shen fuzzy-rough feature selection framework but leaves **unspecified**: the fuzzy similarity relation, t-norm, implicator, search procedure, stopping rule, threshold, normalisation, software, and seed.

#### Feature-Index Ambiguity (New Finding)

The paper's selected features [6,7,3,2,1,5,8] / dropped [0,4] maps differently depending on the column order:

| Mapping | Dropped Features | Likelihood |
|---------|-----------------|------------|
| **M1** (Table 2 order) | Hardness, Organic_carbon | More likely [P] |
| **M2** (CSV order) | ph, Sulfate | Less likely (drops two filled features) |

#### Our FRS Implementation

**Script**: [r4_frs_select.py](file:///home/sabbir/research/repro/paper/code/r4_frs_select.py)

Method: Relation-based lower approximation (Eqs. 4–5), linear similarity (train ranges), min t-norm (Goedel), Kleene-Dienes implicator, QuickReduct greedy forward search, strict stopping at γ = γ(all 9).

**Verified**: Closed-form formula matches brute-force computation to 10 decimals (5 subsets), max diff 5.6e-17.

#### S3-1 FRS Selection Results

| Item | Result |
|------|--------|
| **γ(all 9 features)** | 0.1112 (low: classes overlap heavily) |
| **Single-feature γ** | 0.0003 – 0.0005 (all negligible) |
| **Strict reduct** | ALL 9 features (γ reaches γ(all) only at the last step) |
| **Our full selection order** | Sulfate → Solids → Turbidity → Conductivity → THM → Organic_carbon → Hardness → Chloramines → ph |
| **Our first-7 (paper's size)** | Sulfate, Solids, Turbidity, Conductivity, THM, Organic_carbon, Hardness (drops Chloramines, ph); γ = 0.0889 |
| **Paper list M1 γ** | 0.0878 (overlap 5/7 with our first-7) |
| **Paper list M2 γ** | 0.0909 (overlap 6/7 with our first-7) |
| **Original-rows-only order** | Hardness → Solids → Turbidity → Organic_carbon → Conductivity → THM → Chloramines → ph → Sulfate |

> **Critical observations**:
> 1. **Overlap is at chance**: Any two 7-feature subsets of 9 share at least 5 (expected ~5.4 by chance). Our 5/7 and 6/7 overlap is not evidence of agreement.
> 2. **γ barely separates** candidate 7-sets (0.0878, 0.0889, 0.0909 — all close)
> 3. **Selection is NOT stable**: With SMOTE rows, Sulfate is 1st; on original rows, Sulfate is 9th (last)
> 4. The paper's FRS selection is **neither reproduced nor refuted**

#### S3-2 Classifier Results (5 Feature-Set Variants × 3 Models)

**Script**: [r4_s3_sklearn_models.py](file:///home/sabbir/research/repro/paper/code/r4_s3_sklearn_models.py)

| Variant (Features) | XGBoost Acc / AUC | RF Acc / AUC | AdaBoost Acc / AUC |
|--------------------|-------------------|--------------|---------------------|
| **S2b-ref** (all 9) | 61.59 / 0.633 | 64.18 / 0.673 | 58.54 / 0.575 |
| **S3-paperM1** (drops Hardness, Organic_carbon) | 58.99 / 0.622 | 60.98 / 0.638 | 49.85 / 0.566 |
| **S3-own7** (drops Chloramines, ph) | 59.15 / 0.573 | 55.79 / 0.576 | 56.55 / 0.542 |
| **S3-paperM2** (drops ph, Sulfate) | 52.29 / 0.517 | 55.49 / 0.540 | 46.19 / 0.518 |
| **S3-stab7** (= identical to S3-paperM2) | — | — | — |

**Paper Table 7** (reference only): XGBoost **98.83**, RF **95.19**, AdaBoost **92.21**

> [!CAUTION]
> **Gap ≥ 35 percentage points**. Our best S3 accuracy (60.98%, RF paper-M1) vs paper's 95.19% RF. The paper's results are **NOT reproduced**.

> **Key observations from S3**:
> 1. **No variant beat all-9-features** in accuracy or AUC. Feature selection **did not help** in our implementation.
> 2. **Signal tracks specific features**: Sets keeping ph and Sulfate (M1) stay near S2b-ref; dropping both gives AUC 0.52–0.54 (near chance). Most of the weak predictive signal appears to be in ph and Sulfate.
> 3. γ-based feature importance **does not track** predictive value (Sulfate ranked 1st by γ on SMOTE rows but last on original rows).
> 4. Paired bootstrap (2000 resamples, this split only): differences between M1 and S2b-ref are mostly inside noise; dropping ph+Sulfate (M2) is clearly worse.

---

### 5.6 Reproduction Gap Summary

| Scenario | Our Best Acc % | Paper Acc % | Gap (pts) | Status |
|----------|---------------|-------------|-----------|--------|
| S1 (raw CSV) — RF | 66.31 | 68.60 | −2.3 | Within noise (SE ≈ 1.9) |
| S1 (raw CSV) — XGBoost | 62.50 | 67.98 | −5.5 | Borderline |
| S1 (raw CSV) — AdaBoost | NOT RUNNABLE | 62.50 | — | NOT REPRODUCIBLE |
| S2 (fill ± SMOTE) — RF | 68.45 (S2a) | 90.24–95.19 | ≥ 22 | **NOT REPRODUCED** |
| S2 (fill ± SMOTE) — XGBoost | 64.02 (S2a) | 95.88–98.83 | ≥ 32 | **NOT REPRODUCED** |
| S3 (fill+SMOTE+FRS) — Best | 60.98 (RF, M1) | 95.19–98.83 | ≥ 35 | **NOT REPRODUCED** |

#### Most Likely Explanation for the Gap

**Hypotheses** (all untested [P]):
1. **Leakage from preprocessing before the split**: The paper may have filled missing values, applied SMOTE, and/or performed feature selection on ALL rows before splitting into train/test. Our Kaggle notebook autopsy on synthetic data showed that class-mean filling alone raises accuracy from 64% to 80% (and from real AUC 0.674 to 0.884).
2. **Label used in imputation**: If Potability was used as a predictor in the fill model, blank values become near-perfect proxies for the target.
3. **Different data handling**: Undisclosed transformations.

**Track P** (paper-literal, leaky diagnostic) has been proposed to test hypothesis 1 but has **not been run** (awaiting user approval).

---

## 6. Key Findings

### Main Workstream

1. **Honest performance ceiling**: AUC ≈ 0.69–0.70, accuracy ≈ 67% (just 6 points above the 61% majority baseline). This dataset has very weak class separation.
2. **AutoGluon ≈ PyCaret**: Statistically tied (AUC diff = 0.0017, p = 0.76). The earlier claim of AG > PC was confounded by the PyCaret 70%-rows bug.
3. **Imputation barely matters**: Top 6 imputers are indistinguishable (AUC diff < 0.005). Mean imputation is the simplest and was chosen.
4. **Scaling**: Irrelevant for tree-based models; essential for distance-based ones (KNN/SVM). LogReg is at chance regardless (no linear signal).
5. **Imbalance handling**: No technique improves AUC. SMOTE shifts the precision-recall trade-off without improving discrimination. GAN-based approaches are actively harmful (−0.017 to −0.028 AUC).
6. **CatBoost**: Most frequently selected by PyCaret (31/40 runs); chosen as the Phase 2 tuning target. This is a **post-hoc** selection rule.

### Reproduction Workstream

7. **Paper's results NOT reproduced**: Under a leak-free protocol, all accuracies stay at 58–69%, a gap of ≥22–35 points from the paper's 90–99%.
8. **Imputation fills are near-constant**: With correlations ≤ 0.064, the RF fill models produce values clustered at the column mean (sd 7–20% of observed sd). The imputation adds no real information.
9. **SMOTE is dominated by Solids**: 99.99% of distance variance comes from one feature, so SMOTE neighbours are chosen by Solids alone.
10. **Fuzzy-rough selection is unstable**: The same algorithm produces different feature orders depending on whether SMOTE rows are included. γ does not track predictive value.
11. **Feature selection hurts**: No 7-feature subset outperforms all 9 features. Dropping ph and Sulfate pushes AUC to near chance (0.52–0.54), suggesting these carry most of the weak signal.

---

## 7. Limitations & Caveats

| Limitation | Impact |
|------------|--------|
| **Single dataset** | All findings are specific to Kaggle Water Potability; generalisation requires 8–15 datasets (reviewer expectation) |
| **Single split for S1–S3** | Reproduction scenarios use one 80/20 split (SE ≈ 1.9 pts accuracy). Main workstream uses 10 splits. |
| **Preprocessing chosen on overlapping data** | train.csv overlaps test parts of the 10 repeated splits (mild optimistic bias, equal across compared settings) |
| **Lesson p-values uncorrected** | Holm correction applied in some places but not consistently; Lesson 1 top-6 ties are robust |
| **AG v3 uses 64% of rows** | AutoGluon's holdout design reserves 20% for validation inside the train set, so only ~64% of provided rows are used for fitting |
| **Different sklearn versions** | env_pycaret (1.4.2) vs env_main (1.9.1) affects NaN support and may cause minor numerical differences |
| **GAN settings under-trained** | 200 epochs vs papers' 1000–2000; results are a lower bound on GAN potential |
| **SMOTE unscaled** | Solids dominates distance; unscaled SMOTE per imblearn defaults (protocol rule B12) |
| **FRS algorithm unspecified** | Our implementation may differ from the paper's in ways we cannot detect |
| **Fill grid edge** | min_samples_leaf=25 (largest offered) was chosen for all columns; a larger leaf might score marginally better |
| **Dataset provenance** | Collection and labelling process unclear; pH range 0–14 may be anomalous |
| **v2 PyCaret numbers invalid** | Trained on only 70% of rows; do NOT reuse. Fixed in v3. |

---

## 8. Future Work (Not Started)

| Item | Status | Priority |
|------|--------|----------|
| **Phase 2: CatBoost HPO** | NOT STARTED | Next (after import-check) |
| **Phase 3: XAI / SHAP** | NOT STARTED | After Phase 2 |
| **Track P (paper-literal leaky diagnostic)** | PROPOSED | Tests whether leakage alone explains the 92–99% gap |
| **PyCaret/AutoGluon for S2/S3** | DEFERRED by user | — |
| **Consolidated S2+S3 report table** | Not written | Low |
| **Per-feature signal check** | Not run | Cheap test of the ph/Sulfate hypothesis |
| **Multi-dataset extension** | NOT STARTED | Needed for Q1 journal |
| **Notebook autopsy on real data** | Not run (synthetic only) | Evidence for supervisor |
| **GAN 1000-epoch check** | Not run | Optional |
| **Literature spreadsheet** | NOT STARTED | Needed for paper |
| **GitHub repo setup** | [?] | Private repo planned; friend to add XAI |

### Phase 2 Design (Planned)

- Model: CatBoost (post-hoc selection rule; must disclose)
- Optimisers: Grid, Random (RandomizedSearchCV), Bayesian (Optuna TPE), Genetic (sklearn-genetic-opt)
- Same search space, 5-fold stratified CV, same #evaluations, 5 seeds (Grid deterministic)
- Convergence curves, trials.csv, best_params.json
- Test used once per final model
- Class weights as tuned hyperparameter
- Optuna hyperparameter importance
- Estimated time: ~30 min/seed (unmeasured)

### Awaiting User/Supervisor Input

- Import-check output (`catboost, optuna, shap, sklearn_genetic`)
- Supervisor answer on XAI ambiguity: (a) SHAP on HP importance vs (b) SHAP on feature importance
- Which GANified-SMOTE paper
- Additional datasets / regression task
- README and professor email updates (still cite outdated PC 0.675)

---

## 9. Code & File Inventory

### Main Workstream

| File | Location | Purpose |
|------|----------|---------|
| [phase0_prepare.py](file:///home/sabbir/research/phase0_prepare.py) | `~/research/` | Stratified 80/20 split |
| [common.py](file:///home/sabbir/research/v2/common.py) | `~/research/v2/` | Shared utilities (splits, scoring, resume) |
| [run_baselines.py](file:///home/sabbir/research/v2/run_baselines.py) | `~/research/v2/` | RF, HGB, LogReg, Dummy baselines |
| [run_autogluon.py](file:///home/sabbir/research/v2/run_autogluon.py) | `~/research/v2/` | AutoGluon 10-split runs |
| [run_pycaret.py](file:///home/sabbir/research/v2/run_pycaret.py) | `~/research/v2/` | PyCaret 10-split runs (**BUGGY** — 70% rows) |
| [analyze.py](file:///home/sabbir/research/v2/analyze.py) | `~/research/v2/` | Summary tables and paired tests |
| [common.py](file:///home/sabbir/research/v3/common.py) | `~/research/v3/` | v3 utilities with combo support |
| [preprocessors.py](file:///home/sabbir/research/v3/preprocessors.py) | `~/research/v3/` | Imputers, scalers, resamplers |
| [run_pycaret_preproc.py](file:///home/sabbir/research/v3/run_pycaret_preproc.py) | `~/research/v3/` | PyCaret v3 (bug-fixed) |
| [run_autogluon_preproc.py](file:///home/sabbir/research/v3/run_autogluon_preproc.py) | `~/research/v3/` | AutoGluon v3 |
| [compare_tools.py](file:///home/sabbir/research/v3/compare_tools.py) | `~/research/v3/` | AG v2 vs PC v3 fair comparison |
| [leakage_demo.py](file:///home/sabbir/research/v3/leakage_demo.py) | `~/research/v3/` | WRONG vs RIGHT preprocessing demo |
| [lesson0–3b](file:///home/sabbir/research/lessons/) | `~/research/lessons/` | Preprocessing lessons (7 scripts) |
| [gan_samplers.py](file:///home/sabbir/research/lessons/gan_samplers.py) | `~/research/lessons/` | GAN/SMOTified-GAN/GANified-SMOTE |
| [test_gan_samplers.py](file:///home/sabbir/research/lessons/test_gan_samplers.py) | `~/research/lessons/` | 21/21 tests PASS |

### Reproduction Workstream

| File | Location | Purpose |
|------|----------|---------|
| [r1_check_data.py](file:///home/sabbir/research/repro/paper/code/r1_check_data.py) | `repro/paper/code/` | Data verification |
| [r2_s1_sklearn_models.py](file:///home/sabbir/research/repro/paper/code/r2_s1_sklearn_models.py) | `repro/paper/code/` | S1 XGBoost/RF/AdaBoost |
| [r2_s1_pycaret.py](file:///home/sabbir/research/repro/paper/code/r2_s1_pycaret.py) | `repro/paper/code/` | S1 PyCaret library |
| [r2_s1_autogluon.py](file:///home/sabbir/research/repro/paper/code/r2_s1_autogluon.py) | `repro/paper/code/` | S1 AutoGluon |
| [r2_s1_report.py](file:///home/sabbir/research/repro/paper/code/r2_s1_report.py) | `repro/paper/code/` | S1 consolidated report |
| [r3_corr_table.py](file:///home/sabbir/research/repro/paper/code/r3_corr_table.py) | `repro/paper/code/` | S2 correlation analysis |
| [r3_fill_trackL.py](file:///home/sabbir/research/repro/paper/code/r3_fill_trackL.py) | `repro/paper/code/` | S2 Track L imputation |
| [r3_make_smote_trackL.py](file:///home/sabbir/research/repro/paper/code/r3_make_smote_trackL.py) | `repro/paper/code/` | S2 SMOTE data prep |
| [r3_s2_sklearn_models.py](file:///home/sabbir/research/repro/paper/code/r3_s2_sklearn_models.py) | `repro/paper/code/` | S2 classifiers (S2a + S2b) |
| [r4_frs_select.py](file:///home/sabbir/research/repro/paper/code/r4_frs_select.py) | `repro/paper/code/` | S3 fuzzy-rough selection |
| [r4_s3_sklearn_models.py](file:///home/sabbir/research/repro/paper/code/r4_s3_sklearn_models.py) | `repro/paper/code/` | S3 classifiers (5 variants) |

### Key Result Directories

| Directory | Contents |
|-----------|----------|
| `data_cls/` | train.csv, test.csv, phase0_report.txt |
| `results/` | v1 AG/PC leaderboards, v2 summary |
| `v2/results/water/` | summary_table.csv, paired_tests.csv, boxplots |
| `v3/results/water3/` | combo tables, paired tests, boxplots |
| `lessons/results/` | Lesson CSVs and logs |
| `repro/paper/results/s1/` | S1 metrics, predictions, confusion matrices, report |
| `repro/paper/results/s2/` | S2 fill reports, SMOTE reports, S2+S3 metrics, FRS outputs |

### Protocol Documents

| File | Location |
|------|----------|
| [S1_PROTOCOL.md](file:///home/sabbir/research/repro/paper/S1_PROTOCOL.md) | `repro/paper/` |
| [S2_PROTOCOL.md](file:///home/sabbir/research/repro/paper/S2_PROTOCOL.md) | `repro/paper/` |
| [S3_PROTOCOL.md](file:///home/sabbir/research/repro/paper/S3_PROTOCOL.md) | `repro/paper/` |
| [PROJECT_STATUS.md](file:///home/sabbir/research/repro/paper/PROJECT_STATUS.md) | `repro/paper/` |
| [Early work.md](file:///home/sabbir/research/repro/paper/Early%20work.md) | `repro/paper/` |

---

## 10. References

### Substantially Read

| Reference | Notes |
|-----------|-------|
| **Shehab, Darwish, Hassanien (2025)**. Env Dev Sustain 27:1401–1419. DOI: 10.1007/s10668-023-03916-4 | Paper being reproduced |
| **Sharma, Singh & Chandra (2022)**. "SMOTified-GAN…". IEEE Access 10:30655–30665. arXiv:2108.03235 | SMOTified-GAN method |
| **Cheah, Yang & Lee (2023)**. "Enhancing Financial Fraud Detection… Hybrid SMOTE-GAN". Int J Financial Studies 11(3):110 | GANified-SMOTE method |
| **Jensen & Shen (2004)**. IEEE TKDE 16(12):1457–1471. DOI: 10.1109/tkde.2004.96 | Fuzzy-rough feature selection framework (abstract only) |
| **Stekhoven & Bühlmann (2012)**. MissForest. Bioinformatics 28(1):112–118 | RF-based imputation (abstract only) |

### Abstract/Snippet Only

Bergstra & Bengio 2012, Probst et al. 2019, Lundberg & Lee 2017 (SHAP), Demšar 2006 (statistical tests), Nadeau & Bengio 2003 (corrected t-test), Yang et al. 2025 (SMOTE > GANs), van den Goorbergh et al. 2022 (imbalance corrections), arXiv:2601.12654 (SHAP multiplicity), and others — see [Early work.md](file:///home/sabbir/research/repro/paper/Early%20work.md) section I for the full list with read-level labels.

### Data

Aditya Kadiwal. (2020). *Water Quality* [Data set]. Kaggle. (https://www.kaggle.com/datasets/adityakadiwal/water-potability)

---

> **End of Report** — Last updated 2026-11-10. For questions about specific results or scripts, refer to the protocol documents and result files linked above.
