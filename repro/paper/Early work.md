PROJECT HANDOFF: AutoML vs PyCaret → HPO → XAI (Water Potability case study)

Status labels: [C]=CONFIRMED (seen in outputs/code run by user), [P]=PROVISIONAL, [U]=UNVERIFIED, [PL]=PLANNED, [?]=UNKNOWN.
Abbrevs: AG=AutoGluon, PC=PyCaret, NB=Nadeau–Bengio corrected resampled t-test, bal_acc=balanced accuracy, CV=cross-validation, HPO=hyperparameter optimisation, IR=imbalance ratio.

A. PROJECT IDENTITY
User (Sabbir): complete beginner, “slow learner”, Linux Mint, short on time (laptop can run 2–3 h unattended). Wants step-by-step, ONE lesson at a time, simple language, reason for each command, cross-checked/tested code, references.
Brief (user’s words, paraphrased): apply AutoML and PyCaret to one CSV → compare → pick best model → tune it with Bayesian, Grid, Random, Genetic optimisers → XAI/SHAP on the tuning results (“XAI upon hyperparameter tuning’s value; not on the features”). [?] Ambiguity unresolved: (a) SHAP/hyperparameter-importance on the tuning trials (surrogate SHAP, Optuna fANOVA importance), vs (b) SHAP on the model’s input features. User was told to ask the supervisor; no answer recorded.
Teacher: classification only [C, user-stated]. Supervisor also asked to test SMOTified-GAN and GANified-SMOTE (done, Lesson 3b).
Goal: Q1 journal. User is building a reusable workflow on the Kaggle Water Potability dataset, then will reuse it on other datasets (water/air/temperature/…). A friend will apply XAI later (GitHub repo planned; creation status [?]).
User expectation of ≥80% accuracy was shown to be unrealistic: honest ceiling here ≈67% accuracy / AUC≈0.69–0.70 (see D).
Research framing proposed by me [P, not confirmed with supervisor]: “How does HPO optimiser choice change SHAP explanations/stability, across domains?” Contributions: reproducible leak-free protocol; multi-dataset study (reviewer wants 8–15 datasets, else frame as case study); rank-based explanation-stability metrics (top-k overlap, Spearman/Kendall) with randomised null; random-configuration baseline (equal-accuracy models, how much SHAP varies); TreeSHAP exact for trees. Reviewer expectations: no leakage, equal budgets/folds/metrics, repeated splits, mean±std, NB/Holm or Friedman+Nemenyi across datasets, baselines (dummy, default RF, logreg), ablations, public code, limitations.
B. ENVIRONMENT AND DATA
Machine: Lenovo IdeaPad Slim 3 15IRH8, 16 GB RAM, 13th-gen i5-H, 8 cores/12 threads; system Python 3.12.3; uv installed.
Envs [C]: ~/research/env_pycaret (Py 3.11.17, pycaret[full]); ~/research/env_main (Py 3.11.17; autogluon.tabular[lightgbm,catboost,xgboost,fastai]==1.6.3, ray 2.58.0, torch 2.13.0 (CUDA wheels, runs on CPU), lightgbm 4.7.0, catboost 1.2.10, xgboost-cpu 3.2.0, imbalanced-learn 0.14.2, sklearn-compat). Optuna/shap/sklearn-genetic-opt/matplotlib/jupyterlab were in earlier install commands but [?] import check python -c "import catboost, optuna, shap, sklearn_genetic; print('all ok')" NOT yet reported. Exact PC/sklearn versions in user’s envs [?] (my sandbox: PC 3.3.2, sklearn 1.4.2 and 1.9.1, AG 1.6.3, imblearn 0.14.2, torch 2.14.1).
Gotcha: Lesson 2 once ran in system Python 3.12 (prompt lacked (env_main); warnings from ~/.local/lib/python3.12); results judged valid, but ALWAYS source ../env_main/bin/activate first.
Dirs [C unless noted]: ~/research/{data/raw.csv, data_cls/{train.csv,test.csv,phase0_report.txt}, v2/, v3/, lessons/, env_main/, env_pycaret/, logs.log (PyCaret's own log)}. archive_v1/ (old v1 scripts), autopsy/, NOTES.md, git repo: [?].
Data: data/raw.csv = Kaggle water_potability.csv, 3276×10 [C]. Features (all float): ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity. Target Potability (1=potable). Classes 0:1998 / 1:1278 (61/39, IR 1.56). No duplicates. Missing (full): Sulfate 23.8%, ph 15.0%, THM 4.9%. Phase-0 “ID-like column” warnings were false alarms (continuous values); all 9 features kept.
Provenance concern [U]: dataset text says pH 6.52–6.83 but data ph spans 0–14; collection/label process unclear.
Train file (data_cls/train.csv, n=2620) [C]: Sulfate 625 (23.9%), ph 387 (14.8%), THM 134 (5.1%) blank; rows with ≥1 blank 1013 (39%); complete rows 1607; P(Potable=1) blank vs present: ph .364/.395 (chi² p .286), Sulfate .371/.396 (p .288), THM .328/.393 (p .158); blank-flag correlations ≈0 (−0.016…0.012); skew ph 0.08, Sulfate −0.05, THM −0.07.
Splits [C]: (1) Primary split: stratified 80/20, seed 42 → data_cls train 2620 / test 656 (Phase 0). (2) Phase-1 v2/v3 outer evaluation: 10 stratified 80/20 splits of raw.csv, random_state=0..9. (3) Lessons: RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42) on data_cls/train.csv = 20 val scores; test.csv never opened in lessons. (4) notebook_autopsy: 75/25 stratified seeds.
Metrics policy: primary ROC-AUC; secondary bal_acc, F1, MCC, accuracy (vs 61% majority baseline), logloss, Brier.
C. WORKFLOW / PHASES
Phase	Content	Status
0	phase0_prepare.py → data_cls split, report	DONE [C]
1 v1	single-seed tests of AG/PC	superseded; AG first run invalid (missing libs)
1 v2	10 splits: baselines + AG + PC (+ intended A_all/B_trees; user set CONDITIONS=[“A_all”])	DONE; PC part has BUG (below)
Lessons 0–3b	one-factor-at-a-time preprocessing (missing, imputer, scaling, imbalance, GAN)	DONE [C]
v3	4-combo PC and AG runs on 10 splits; compare_tools	DONE [C]
v3 full 32-combo factorial (screen_preprocessing.py)	written + synthetically tested; NOT run on real data	optional
2	Tuning CatBoost: Grid, Random, Bayesian(Optuna TPE), Genetic	NOT STARTED [PL]
3	XAI/SHAP (+Optuna importance)	NOT STARTED [PL]
Pre-registered model rule (set AFTER Phase 1, must be disclosed as post-hoc) [P]: standalone sklearn-compatible estimator with ≥4 tunable hyperparameters, best mean CV AUC ⇒ CatBoost. QDA excluded (≈1 hyperparameter); AG’s NeuralNetTorch excluded (internal, not standalone). CatBoost also supports class weights (auto_class_weights) and TreeSHAP.
Phase-2 design discussed [PL]: same search space, same 5-fold stratified CV, same #evaluations for all four optimisers (Grid uses its grid size; state openly), 5 seeds (Grid deterministic), wall-clock logged, convergence curves, save trials.csv + best_params.json (friend’s XAI needs them; models rebuilt from params + deterministic splits), test used once per final model, class weights as tuned hyperparameter, Optuna hyperparameter importance. Libraries: Optuna TPE, GridSearchCV, RandomizedSearchCV, sklearn-genetic-opt GASearchCV or DEAP. Time guess ≈30 min/seed for all four (UNMEASURED).
D. EXPERIMENTS AND RESULTS
D1. v2 Phase 1 (10 splits, mean±std, test unless noted) [C]
Method	test AUC	CV AUC	F1	bal_acc	MCC
AG A_all (5-fold bagging, all train rows; best single non-ensemble model)	0.6942±0.0192	0.7004 (OOF)	0.5047	0.6308	0.2899
default RF (median imputer, 500 trees)	0.6880±0.0222	0.6718	0.4519	0.6131	0.2709
PC A_all (BUGGY, 70% of rows)	0.6751±0.0249	0.6753	0.4551	0.6138	0.2708
default HGB	0.6612±0.0232	0.6556	0.4771	0.6079	0.2354
logreg (median+scale)	0.5003±0.0151	0.5033	0.0054	0.5012	0.0176
dummy	0.500	0.5	0	0.5	0
AG winners: NeuralNetTorch 9, LightGBMXT 1. PC winners (buggy): QDA 6, CatBoost 4. Time caps (300 s) not binding (AG ≈39 s, PC ≈7 s).
Paired (NB), AUC: PC−AG −0.0191 (CI −0.0279…−0.0111, p .0526, Holm .158) RETRACTED (confounded by bug); AG vs RF +0.0061 (p .41); PC vs RF −0.0129 (p .18); both vs logreg ≈+0.17–0.19.
BUG [C]: PC setup() default train_size=0.7 ⇒ trained/CV’d on 70% of supplied training rows (verified: 840 of 1200). Fix: test_data=test (v3). v2 PC numbers must not be reused.
v1 single-seed (early): PC(seed0) QDA cv .6818 test .6711; AG rerun (full libs) NeuralNetTorch cv .6992 test .6579, 141 s.
D2. Lessons (data_cls/train.csv; 20 CV folds; mean AUC; NB p vs baseline)

Lesson 1 imputer (RF, then HGB) [C]:

Imputer	RF AUC	HGB AUC
iterative (IterativeImputer, BayesianRidge, 10 it)	.6851	.6637
native (leave NaN)	.6848	.6623
mean	.6841	.6685
median	.6829	.6677
median+indicator	.6795	.6647
KNN(k=5, min-max first)	.6785	.6600
zero fill	.6674 (−.0155 vs median, p .047)	.6458 (−.0219, p .055)
hot deck (random donor)	.6651 (−.0178, p .037)	.6425 (−.0252, p .048)
Top six tied (	diff	<.005, p≥.48). p-values NOT multiplicity-corrected (Holm over 7 ⇒ hot-deck/zero not significant). RF acc≈.673, bal_acc≈.614, F1≈.45. Decision: mean imputer (simplest tied option usable with SMOTE; PC default) [C, applied].
Lesson 2 min-max (mean imputer) [C]: RF .6841→.6841 (p .93); KNN .5095→.6324 (p .000); SVM-RBF .5235→.6903 (p .000); MLP .4913→.6492 (p .003); LogReg .4802→.4805 (p .95). Unscaled distance models ≈ chance; trees invariant; logreg at chance regardless (no linear signal).	
Lesson 3 imbalance (pipeline mean→min-max→fix→model) [C]:	
Fix	RF AUC/acc/bal/F1	SVM AUC/acc/bal/F1
---	---	---
none	.6841/.6737/.6139/.4490	.6903/.6690/.5927/.3659
class weights	.6878/.6635/.6304/.5257	.6845/.6511/.6308/.5459
threshold moving (inner CV, bal_acc)	.6841/.6421/.6252/.5424	.6903/.6393/.6302/.5569
SMOTE	.6835/.6554/.6271/.5292	.6778/.6308/.6253/.5587
SMOTE-Tomek	.6854/.6517/.6220/.5207	.6756/.6300/.6240/.5567
Borderline-SMOTE	.6810/.6503/.6236/.5279	.6735/.6168/.6227/.5693
ADASYN	.6806/.6425/.6218/.5347	.6730/.6025/.6221/.5826
rand. oversample	.6824/.6640/.6192/.4905	.6851/.6455/.6252/.5388
rand. undersample	.6735/.6219/.6190/.5553	.6725/.6386/.6202/.5360
SMOTE-ENN	.6363/.5770/.5978/.5607	.6334/.5411/.5815/.5650
No fix changes AUC meaningfully (all p>.18) except SMOTE-ENN (−.048 p .007 RF; −.057 p<.001 SVM) and SVM undersampling (−.018, p .042). RF bal_acc: class weights +.0166 (p .022). F1 gains (+.04…+.22) mostly = decision-threshold shift (threshold moving gives the same F1 gain as SMOTE, no data change). Accuracy is highest with NO fix. Ignore --quick (5-fold) tables. Decisions: SMOTE-ENN dropped; carried to AutoML check: none, SMOTE, SMOTE-Tomek (my selection was weakly justified—I wrongly claimed Tomek “most stable AUC”; SVM data don’t support it; reasons were popularity + PC/AG can’t do class weights/threshold moving).	
**Lesson 3b GAN (RF	SVM; 200 epochs, lr 2e-4)** [C]:	
Fix	RF AUC/bal/F1	SVM AUC/bal/F1
---	---	---
none	.6841/.6139/.4490	.6903/.5927/.3659
class weights	.6878/.6304/.5257	.6845/.6308/.5459
SMOTE	.6835/.6271/.5292	.6778/.6253/.5587
SMOTified-GAN	.6674/.6120/.4758	.6676/.6008/.4396
GANified-SMOTE	.6639/.6056/.4665	.6629/.5947/.4360
GAN only	.6639/.6036/.4695	.6622/.5956/.4401
AUC vs none: SMOTified −.0167 (p .048 RF), −.0227 (p .010 SVM); GANified −.0201 (p .096), −.0274 (p .013); GAN −.0201 (p .094), −.0281 (p .007). vs SMOTE F1: −.053…−.063 (RF, p≤.004); −.119…−.123 (SVM, p .000). Sanity std_ratio 0.66–0.75 (no collapse; doesn’t prove realism). Negative result; not carried forward. Caveat: epochs/lr far below papers (Sharma: lr 1e-5, ~2000 epochs, early stop on val F1); optional --epochs 1000 RF check suggested, [?] not run.	
D3. v3 AutoML with chosen preprocessing (10 splits; combo = imputer|scaler|resampler) [C]

PC (fixed: all train rows):

combo	cv AUC	test AUC	acc	bal_acc	F1	MCC
mean|none|none	.6871±.0073	.6959±.0188	.6767±.0126	.6208	.4688	.2853
mean|minmax|none	.6865	.6949±.0199	.6764	.6206	.4688	.2847
mean|minmax|smotetomek	.6828	.6892±.0235	.6462±.0178	.6326	.5569	.2635
mean|minmax|smote	.6842	.6836±.0212	.6482±.0204	.6330	.5555	.2651
Winners: CatBoost 31/40, QDA 9/40 (QDA 7/20 in SMOTE variants vs 2/20 without). Paired vs mean	minmax	none: AUC none	none +.0010 (Holm .84), Tomek −.0058 (.84), SMOTE −.0113 (CI −.0203…−.0025; raw p .239; Holm .72); acc SMOTE −.0282 (Holm .14), Tomek −.0302 (Holm .037).	
AG holdout design (AG sees 64% of rows; model chosen on one 20% validation set, valAUC .67–.75 optimistic vs test ≈.68):	
combo	test AUC	acc	bal_acc	F1	MCC	
---	---	---	---	---	---	
mean|minmax|smote	.6816±.0223	.6328±.0122	.6265	.5585	.2494	
mean|none|none	.6814±.0215	.6639±.0227	.6223	.498	.2687	
mean|minmax|none	.677±.0226	.6648	.6209	.490	.269	
mean|minmax|smotetomek	.6714±.018	.6271	.6166	.5419	.2313	
AG winners: NeuralNetTorch 25, FastAI 5, LightGBMXT 4, CatBoost 4, LightGBM 1, LightGBMLarge 1. Paired AUC all Holm 1.0; acc SMOTE −.0320 (Holm .121), Tomek −.0377 (Holm .076).	
Like-for-like tool comparison (compare_tools.py, same 10 splits) [C]: AG v2 0.6942±0.0192 vs PC v3 (mean	none	none) 0.6959±0.0188; diff +0.0017, 95% CI [−0.0040, +0.0069], NB p=0.760 ⇒ statistically tied. (AG v3 vs PC v3 NOT comparable: different training size/selection design.)	
Cross-tool conclusions [C]: SMOTE/Tomek leave AUC unchanged, cost ≈3–4 acc points, add ≈+.06–.09 F1; min-max no effect for winners (trees/QDA/AG’s own NN scaling). Best honest result: AUC≈.69–.70, accuracy≈67.7% (PC, no fix).	
D4. Leakage investigation of a Kaggle notebook (user-provided; claims GB 78%, Bagging 76%, H2O AutoML 80%, leaderboard AUC .864) [U on real data]

Notebook fills blanks with CLASS mean via groupby(target).transform('mean') on the WHOLE df before splitting (uses the label; each blank becomes one of two constants → trees read the label; also in test rows). SMOTE and scaler were correctly applied after split (SMOTE not the cause); CV/GridSearch were run on SMOTE-expanded data (inflated); H2O reused the leaky frame; single split; notebook’s own EDA shows near-zero class differences. Back-of-envelope: .39×1.0+.61×.66≈.79. notebook_autopsy.py on SYNTHETIC water-like data (3276 rows, 40% rows blank): honest acc .641/AUC .674; honest+SMOTE .640/.670; notebook fill only .800/.884 (acc on blank rows .996, complete rows .663); notebook full .794/.883. Run on the real file: [?] not yet reported. leakage_demo.py (WRONG vs RIGHT) tested on synthetic only (SMOTE-before-split: AUC .723→.792, acc .674→.710 on weak-signal synthetic).

E. DECISIONS AND REASONS
Decision	Status	Evidence/notes
Imputer = mean	[C]	six imputers tied; mean simplest, compatible with SMOTE (needs no NaN), PC default
Hot deck, zero-fill rejected	[P]	−.015…−.025 AUC, raw p .037–.055, not Holm-significant; consistent in 2 models
Scaling: none for trees; min-max only for KNN/SVM/NN	[C]	Lesson 2, PC/AG
Imbalance: none for AUC/accuracy; class weights/threshold moving if bal_acc/F1 matter	[C]	Lessons 3, 3b, v3
SMOTE-ENN dropped	[C]	AUC −.05
GAN hybrids not carried forward	[C]	Lesson 3b
v2 “AG > PC by .019” retracted	[C]	PC 70%-rows bug
AG≈PC	[C]	compare_tools p .76
Phase-2 model = CatBoost	[P]	post-hoc rule; disclose
Tree-only restriction (B_trees)	REMOVED	I added it without asking; user wants brief-faithful A_all (full libraries)
Staged (one-factor) preprocessing selection, optional 32-combo factorial as final check	[C]	beginner learning + fewer comparisons; interactions may be missed
Limitations to disclose: preprocessing chosen on data_cls/train.csv, which overlaps test parts of the 10 repeated splits (mild optimistic bias, equal across compared settings); Lesson p-values uncorrected; AG v3 uses 64% of rows; time caps non-binding; single weak-signal dataset; dataset provenance; GAN settings under-trained vs papers; PC version/sklearn versions [?]; v2 AG imputation was model-specific (RF/XT fill 0, NN median, GBMs native).	
F. CODE AND REPRODUCIBILITY

The new chat has NO access to these files; user must re-upload the ones needed (all on user’s disk). Verified-in-sandbox ≠ verified on real data unless stated.

~/research/ (v1, superseded): phase0_prepare.py [C used]: python phase0_prepare.py --csv data/raw.csv --target Potability --task classification --out data_cls (args --drop, --test-size 0.2, --seed 42). phase1_pycaret.py, phase1_autogluon.py, phase1_compare.py (v1, obsolete).

~/research/v2/ [C run]: common.py (N_CPUS=8, TIME_LIMIT=300, TEST_SIZE=.2, CONDITIONS=[“A_all”] edited by user, get_split, scores, resume via done_keys/save_row), run_baselines.py, run_autogluon.py (num_bag_folds=5, num_stack_levels=0, medium_quality, best non-WeightedEnsemble by score_val), run_pycaret.py (BUGGY), analyze.py. Run: python run_baselines.py --csv ../data/raw.csv --target Potability --name water (same for run_autogluon.py [env_main], run_pycaret.py [env_pycaret]), python analyze.py --name water. Outputs v2/results/water/{phase1_raw.csv,summary_table.csv,paired_tests.csv,auc_boxplot.png,*_leaderboards.csv,environment.txt}.

~/research/v3/ [C run unless noted]: common.py (adds combos “imputer|scaler|resampler”, BASE_COMBO “mean|none|none”, resolve_combos all|base|top:K|list), preprocessors.py (HotDeckImputer: random donor from training column values; IMPUTERS mean/median/knn/hotdeck; SCALERS none/minmax; RESAMPLERS none/smote/adasyn/smotetomek), screen_preprocessing.py [NOT run on real data], run_pycaret_preproc.py, run_autogluon_preproc.py, leakage_demo.py [synthetic only], analyze_preproc.py (--base option), compare_tools.py (reads ../v2/results/water/phase1_raw.csv and results/water3/pycaret_raw.csv). Commands (quote combos; | is a shell pipe):

cd ~/research/v3
python run_pycaret_preproc.py --csv ../data/raw.csv --target Potability --name water3 --combos "mean|none|none,mean|minmax|none,mean|minmax|smote,mean|minmax|smotetomek"   # env_pycaret, ~8–12 s/run
python run_autogluon_preproc.py  (same args)   # env_main
python analyze_preproc.py --name water3 --base "mean|minmax|none"
python compare_tools.py

Outputs v3/results/water3/{pycaret_raw.csv,autogluon_raw.csv,*_combo_table.csv,*_paired_vs_base.csv,*_boxplot.png,environment_v3.txt}. All runners resumable (skip finished (combo,repeat)).

~/research/lessons/ [C run]: lesson0_look_at_missing.py, lesson1_imputation.py --model rf|et|hgb, lesson2_scaling.py, lesson3_imbalance.py --model rf|svm [--quick], lesson3b_gan.py --model rf|svm [--epochs N] [--quick], gan_samplers.py, test_gan_samplers.py (user result: 21/21 PASS). Usage: python lessonX_*.py --train ../data_cls/train.csv --target Potability 2>&1 | tee results/<log>. Logs/CSVs in lessons/results/. notebook_autopsy.py (--csv ../data/raw.csv --target Potability --repeats 5) [real-data run ?]. Also drafted: README.md for GitHub (contains OUTDATED PC 0.675 and v2 table; must be updated), .gitignore advice (ignore env_*/, data/, data_cls/, catboost_info/, logs, results/smoke/), SSH-key GitHub steps (private repo; add friend as collaborator).

Critical code/logic (verbatim essentials):

python
# splits (v2/v3 common.py)
train_test_split(df, test_size=0.2, random_state=repeat, stratify=df[target])   # repeat 0..9
# NB corrected resampled t (ratio = n_test/n_train = 0.25 for 5-fold or 80/20)
t = d.mean()/np.sqrt((1/J + ratio)*d.var(ddof=1)); p = 2*stats.t.sf(abs(t), J-1)   # + bootstrap 95% CI, Holm
# PC v3 (leak-free; test_data fixes the 70% bug; scaler placed BEFORE SMOTE because PC applies balance before its own normalize)
setup(data=train.copy(), test_data=test.copy(), target=T, session_id=rep, fold=5, fold_strategy="stratifiedkfold",
      n_jobs=8, verbose=False, html=False, imputation_type=None, normalize=False,
      custom_pipeline=[("imp", imputer), ("scale", MinMaxScaler())], custom_pipeline_position=0,
      fix_imbalance=True, fix_imbalance_method=SMOTE(random_state=rep))   # only if resampler != none
compare_models(sort="AUC", n_select=1, budget_time=5); predict_model(best, data=test)  # proba = where(label==1, score, 1-score)

PC internals verified from source 3.3.2: whole preprocessing pipeline (imputer, scaler, SMOTE) is refit inside each CV fold; resampling only on training folds; canary (pure-noise labels + SMOTE inside folds) AUC .502. AG v3 holdout design: inner 80/20 split of train; imputer/scaler fit on inner-train; resampler on inner-train only; fit(train_df, tuning_data=valid_df, presets="medium_quality", num_cpus=8); best non-ensemble by score_val; test scored once.
Lesson pipeline pattern: imblearn.pipeline.Pipeline([imp, scaler, fix, model]) inside cross_validate (fit per fold, resampling only at fit).
GANSampler (fit_resample(X,y), imblearn-compatible, trained inside fold on received rows only; internal MinMax; modes gan | smotified_gan | ganified_smote; defaults epochs=200, batch 64, lr 2e-4 Adam(0.5,0.999), latent 16, gan_fraction .5, k_neighbors 5, smote_on="gan_only"; G = 3×[Linear128-BatchNorm-ReLU]+Linear+Sigmoid; D = Linear128-LeakyReLU(.2)-Linear1; BCEWithLogits; output = originals first + synthetic, classes balanced). smotified_gan: SMOTE rows are generator INPUT (Sharma 2022); ganified_smote: GAN makes gan_fraction of the gap, then SMOTE interpolates between GAN rows only (Cheah 2023); real_plus_gan is a non-published variant. Test checks: counts, originals untouched, synthetic in real range, deterministic, uses only passed rows.

G. CURRENT STATE
Completed & verified (user outputs): Phase 0; v2 baselines/AG; v3 PC & AG 4-combo runs; Lessons 0–3b; compare_tools; GAN self-test 21/21.
Completed, verified only in my sandbox/synthetic: notebook_autopsy, leakage_demo, screening script, 32-combo machinery, analyze scripts’ edge cases.
Awaiting from user: import-check output (catboost, optuna, shap, sklearn_genetic); notebook_autopsy real-data output; optional GAN 1000-epoch RF run; confirmation NOTES.md entries written; GitHub repo status; supervisor answers (XAI meaning; which GANified-SMOTE paper; extra datasets/regression).
Known issues: v2 PC numbers invalid (don’t reuse; README/professor email drafts still cite PC 0.675/”AG better”); see limitations in E.
Not started: Phase 2 tuning, Phase 3 XAI, multi-dataset extension, literature spreadsheet.
H. EXACT NEXT ACTIONS
(Immediate) Get import-check output; if OK → write Phase-2 scripts for CatBoost: fix tuning protocol first (data: data_cls train/test vs repeated splits; imputer = mean in pipeline vs CatBoost native NaN; search space incl. auto_class_weights; eval budget N; seeds 5; CV 5-fold stratified; metrics AUC primary + bal_acc/F1), four optimisers, trials.csv, best_params.json, convergence log, timing; run test-once evaluation. Do a 1-seed smoke test first and measure time to size windows (2–3 h).
Update professor message/README with corrected PC results (PC 0.6959 vs AG 0.6942, tied) and Lesson findings; write NOTES.md decisions (rule for CatBoost flagged post-hoc).
Run real-data notebook_autopsy.py (evidence for professor).
Optional: GAN 1000-epoch check; read Cheah Sect. 3.2.5/Table 1 (extraction garbled).
Later: Phase 3 SHAP design (resolve XAI ambiguity with supervisor), stability metrics, second dataset.
I. LITERATURE (read-level honesty)

Read substantially: Sharma, Singh & Chandra 2022, “SMOTified-GAN…”, IEEE Access 10:30655–30665, arXiv 2108.03235; Cheah, Yang & Lee 2023, “Enhancing Financial Fraud Detection…Hybrid SMOTE-GAN”, Int. J. Financial Studies 11(3):110 (GANified-SMOTE = GAN first, then SMOTE on GAN rows; Table 1 extraction garbled; they min-max scaled before splitting). User-attached (smote.md, CC BY-NC-ND; do not adapt): Lin et al. 2026, BSMOTE-DCGAN-EL, highly imbalanced (IR 5–20, ~100 train samples) biomedical data—not our regime. Skimmed: arXiv 2502.16535 (systematic map of GANs for imbalanced tabular data, ~100 studies). Abstract/snippet only: Yang et al. 2025 J. Oper. Res. Soc. 76(10):1981–2002 (SMOTE beat GAN-based methods; RF most robust); arXiv 2501.11293 (bluebottle stinger; side relevance; its CT-GAN scores may be on a balanced test with synthetic negatives—my inference); Ruslan, Arbaiy & Lin 2025 (GANified-SMOTE comparison; I earlier mis-attributed the method to it); van den Goorbergh et al. 2022 arXiv 2202.09101 / PMC9382395 (imbalance corrections: no AUC gain, worse calibration, threshold shift similar); medRxiv 2026 preprint on miscalibration; comment on Iacobescu et al., J. Cardiovasc. Dev. Dis. 2024, 11, 396 (PMC12842384; SMOTE-ENN before split leakage); sklearn docs (KNNImputer, IterativeImputer, add_indicator); water-potability papers via snippets only (RF ≈66%; a 99% XGBoost claim; leak status of any paper UNVERIFIED); imputation reviews (snippets: no universal best imputer; hot deck weaker at 30–40% missing); snippet-only: arXiv 2601.12654 (SHAP explanation multiplicity), 2311.15854 (Ray Tune HPO engines on tabular), 2206.11104 (OpenXAI), 2202.00858, sklearn-genetic-opt docs (vendor benchmark).
Recalled from memory in my first answer, NOT opened [U]: Hutter/Kotthoff/Vanschoren 2019 AutoML book; Erickson 2020 AutoGluon-Tabular; Feurer 2015; Gijsbers AMLB 2024; Bergstra & Bengio 2012; Bergstra 2011 TPE; Snoek 2012; Akiba 2019 Optuna; Olson & Moore 2016 TPOT; Bischl 2023; Probst 2019; Lundberg & Lee 2017; Lundberg 2020; Hutter 2014 (fANOVA); Ribeiro 2016 LIME; Demšar 2006; Kapoor & Narayanan 2023; Grinsztajn 2022; Nadeau & Bengio 2003. Outstanding: verify/read these, build the paper-analysis spreadsheet (problem, method, data, metrics, limits, gap), novelty search for “HPO + SHAP stability”.

J. CRITICAL RULES FOR THE NEXT CLAUDE
Split first; every learned step (impute, scale, resample, select) inside the pipeline/fold; resample only training folds; NEVER use the target to fill/scale/select; test set scored once; choose by CV, not test.
Same folds/metric/budget/CPUs for all compared methods; repeated splits; report mean±std; NB corrected t-test + bootstrap CI + Holm; don’t over-claim; report negative results; disclose post-hoc decisions.
Accuracy always vs 61% baseline; do not promise ≥80% (honest ≈67%).
Beginner mode: one step/lesson at a time, short simple explanations, commands with reasons; Linux Mint paths; tell which env to activate (prompt must show (env_main) or (env_pycaret)); quote "a|b|c" combos; refer to files by name.
Cross-check and test code before giving it (state what was/wasn’t tested and on what versions); cite sources and say if only abstracts were read; avoid long answers; user is frustrated by delays.
Don’t combine scripts the user asked to learn separately; don’t silently convert suggestions into decisions; admit own earlier errors (PC 70% bug, tree-only restriction, Tomek claim, GANified-SMOTE attribution/implementation).
PC/AG facts: PC train_size default .7; PC applies balance before normalize; AG RF/XT fill NaN with 0, NN median, GBMs native; AG v3 holdout sees 64% of rows.
