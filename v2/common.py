"""Shared helpers for the Phase 1 v2 protocol (any binary-classification CSV)."""
import argparse
import os
import platform
from importlib import metadata

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split

# ---- protocol constants: keep IDENTICAL for every tool and every dataset ----
N_CPUS = 8          # physical cores of the laptop
TIME_LIMIT = 300    # seconds per run per tool/condition
TEST_SIZE = 0.2
#CONDITIONS = ["A_all", "B_trees"]
CONDITIONS = ["A_all"]
TREE_PYCARET = ["rf", "et", "lightgbm", "xgboost", "catboost"]
TREE_AG = {"RF": {}, "XT": {}, "GBM": {}, "XGB": {}, "CAT": {}}
RAW = "phase1_raw.csv"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--name", required=True, help="short dataset name, e.g. water")
    p.add_argument("--repeats", type=int, default=10)
    return p.parse_args()


def load(csv, target):
    df = pd.read_csv(csv).drop_duplicates()
    df = df.dropna(subset=[target]).reset_index(drop=True)
    classes = sorted(df[target].unique())
    if len(classes) != 2:
        raise SystemExit(f"Binary targets only; found classes {classes}")
    df[target] = (df[target] == classes[1]).astype(int)  # positive class = 1
    bad = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise SystemExit(f"Non-numeric feature columns {bad}: encode them first.")
    return df


def get_split(df, target, repeat):
    """Deterministic stratified split for a repeat number."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=repeat, stratify=df[target])


def scores(y, proba):
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= 0.5).astype(int)
    return {
        "test_auc": roc_auc_score(y, proba),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_bal_acc": balanced_accuracy_score(y, pred),
        "test_mcc": matthews_corrcoef(y, pred),
        "test_logloss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "test_brier": brier_score_loss(y, proba),
        "test_acc": accuracy_score(y, pred),
    }


def out_dir(name):
    d = os.path.join("results", name)
    os.makedirs(d, exist_ok=True)
    return d


def done_keys(name):
    path = os.path.join(out_dir(name), RAW)
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d["tool"], d["condition"], d["repeat"]))


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_row(name, row):
    _append(os.path.join(out_dir(name), RAW), pd.DataFrame([row]))


def save_board(name, filename, board):
    _append(os.path.join(out_dir(name), filename), board)


def log_env(name):
    """Record hardware/software once, for the paper's reproducibility section."""
    path = os.path.join(out_dir(name), "environment.txt")
    lines = [f"platform: {platform.platform()}", f"python: {platform.python_version()}",
             f"cpus_used: {N_CPUS}", f"time_limit_s: {TIME_LIMIT}"]
    for pkg in ["scikit-learn", "pycaret", "autogluon.tabular", "lightgbm", "xgboost", "catboost"]:
        try:
            lines.append(f"{pkg}: {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n---\n")
