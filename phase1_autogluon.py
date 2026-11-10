"""
Phase 1b: AutoGluon. Run inside the main environment:
  source env_main/bin/activate
  python phase1_autogluon.py
We pick the best SINGLE model (not the WeightedEnsemble) by validation AUC, so it can be tuned in Phase 2.
With bagging, score_val is the out-of-fold AUC (comparable to PyCaret's 5-fold CV).
"""
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

TARGET = "Potability"
#SEEDS = [0, 1, 2, 3, 4]
SEEDS = [0]
#TIME_LIMIT = 600  # seconds per run; keep identical across runs
TIME_LIMIT = 300
TRAIN, TEST = "data_cls/train.csv", "data_cls/test.csv"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

rows, boards = [], []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    path = f"{OUT}/ag_seed{seed}"
    shutil.rmtree(path, ignore_errors=True)

    t0 = time.time()
    predictor = TabularPredictor(
        label=TARGET, problem_type="binary", eval_metric="roc_auc", path=path, verbosity=1
    ).fit(
        train, time_limit=TIME_LIMIT, presets="medium_quality",
        num_bag_folds=5, num_bag_sets=1, num_stack_levels=0,
    )
    elapsed = time.time() - t0

    lb = predictor.leaderboard(test, silent=True)
    single = lb[~lb["model"].str.startswith("WeightedEnsemble")]
    best = single.sort_values("score_val", ascending=False).iloc[0]
    name = best["model"]

    proba = predictor.predict_proba(test, model=name)[1]
    label = predictor.predict(test, model=name).astype(int)
    y = test[TARGET]

    rows.append({
        "tool": "autogluon", "seed": seed,
        "best_model": name,
        "cv_auc": best["score_val"],
        "test_auc": roc_auc_score(y, proba),
        "test_f1": f1_score(y, label),
        "test_bal_acc": balanced_accuracy_score(y, label),
        "test_acc": accuracy_score(y, label),
        "seconds": elapsed,
    })
    lb["seed"] = seed
    boards.append(lb)
    print(rows[-1])

pd.DataFrame(rows).to_csv(f"{OUT}/autogluon_results.csv", index=False)
pd.concat(boards).to_csv(f"{OUT}/autogluon_leaderboards.csv", index=False)
print("Saved to", OUT)
