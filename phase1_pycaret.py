"""
Phase 1a: PyCaret. Run inside the PyCaret environment:
  source env_pycaret/bin/activate
  python phase1_pycaret.py
Selection uses 5-fold CV AUC on TRAIN only. The test set is only scored, never used to choose.
"""
import os
import time

import numpy as np
import pandas as pd
from pycaret.classification import compare_models, predict_model, pull, setup
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

TARGET = "Potability"
#SEEDS = [0, 1, 2, 3, 4]
SEEDS = [0]
TRAIN, TEST = "data_cls/train.csv", "data_cls/test.csv"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

rows, boards = [], []
for seed in SEEDS:
    t0 = time.time()
    # PyCaret imputes missing values inside CV (default: mean/mode), fit on training folds only
    setup(
        data=train, target=TARGET, session_id=seed,
        fold=5, fold_strategy="stratifiedkfold",
        n_jobs=-1, verbose=False, html=False,
    )
    best = compare_models(sort="AUC", n_select=1, verbose=False)
    board = pull().copy()
    elapsed = time.time() - t0

    # score once on the held-out test set (reporting only)
    pred = predict_model(best, data=test.copy(), verbose=False)
    label = pred["prediction_label"].astype(int)
    score = pred["prediction_score"]
    proba = np.where(label == 1, score, 1 - score)  # P(class 1)
    y = test[TARGET]

    rows.append({
        "tool": "pycaret", "seed": seed,
        "best_model": board.iloc[0]["Model"],
        "cv_auc": board.iloc[0]["AUC"],
        "test_auc": roc_auc_score(y, proba),
        "test_f1": f1_score(y, label),
        "test_bal_acc": balanced_accuracy_score(y, label),
        "test_acc": accuracy_score(y, label),
        "seconds": elapsed,
    })
    board["seed"] = seed
    boards.append(board.reset_index())
    print(rows[-1])

pd.DataFrame(rows).to_csv(f"{OUT}/pycaret_results.csv", index=False)
pd.concat(boards).to_csv(f"{OUT}/pycaret_leaderboards.csv", index=False)
print("Saved to", OUT)
