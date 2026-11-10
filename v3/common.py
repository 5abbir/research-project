"""Shared helpers for the v3 preprocessing study (binary classification CSV)."""
import argparse
import os
import platform
from importlib import metadata

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             log_loss, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split

from preprocessors import IMPUTERS, RESAMPLERS, SCALERS

N_CPUS = 8            # physical cores of the laptop; same for every run
TIME_LIMIT = 300      # seconds cap per AutoML run
TEST_SIZE = 0.2
BASE_COMBO = "mean|none|none"   # reference: mean imputation only


def all_combos():
    return [f"{i}|{s}|{r}" for i in IMPUTERS for s in SCALERS for r in RESAMPLERS]


def out_dir(name):
    d = os.path.join("results", name)
    os.makedirs(d, exist_ok=True)
    return d


def resolve_combos(spec, name, rank_by="cv_auc"):
    """spec: 'all' | 'base' | 'top:K' (best K from the screening, plus the base) | 'a|b|c,d|e|f'."""
    if spec == "all":
        return all_combos()
    if spec == "base":
        return [BASE_COMBO]
    if spec.startswith("top:"):
        k = int(spec.split(":")[1])
        path = os.path.join(out_dir(name), "screen_raw.csv")
        if not os.path.exists(path):
            raise SystemExit("Run screen_preprocessing.py first (results/%s/screen_raw.csv missing)." % name)
        s = pd.read_csv(path)
        ranked = s.groupby("combo")[rank_by].mean().sort_values(ascending=False).index.tolist()
        picked = ranked[:k]
        return picked if BASE_COMBO in picked else picked + [BASE_COMBO]
    combos = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [x for x in combos if x not in all_combos()]
    if bad:
        raise SystemExit(f"Unknown combos {bad}. Valid example: {BASE_COMBO}")
    return combos


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--name", required=True, help="short dataset name, e.g. water")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--combos", default="all", help="all | base | top:K | comma list")
    p.add_argument("--rank-by", default="cv_auc", help="for top:K (cv_auc, cv_bal_acc, cv_f1, cv_mcc)")
    p.add_argument("--quick", action="store_true", help="tiny smoke test (few models, short budget)")
    return p.parse_args()


def load(csv, target):
    df = pd.read_csv(csv).drop_duplicates()
    df = df.dropna(subset=[target]).reset_index(drop=True)
    classes = sorted(df[target].unique())
    if len(classes) != 2:
        raise SystemExit(f"Binary targets only; found classes {classes}")
    df[target] = (df[target] == classes[1]).astype(int)
    bad = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise SystemExit(f"Non-numeric feature columns {bad}: encode them first.")
    return df


def get_split(df, target, repeat):
    """Same seeds (0..9) as v2, so splits are identical to the earlier Phase 1 run."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=repeat, stratify=df[target])


def scores(y, proba):
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= 0.5).astype(int)
    return {
        "test_auc": roc_auc_score(y, proba),
        "test_acc": accuracy_score(y, pred),
        "test_bal_acc": balanced_accuracy_score(y, pred),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_mcc": matthews_corrcoef(y, pred),
        "test_logloss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "test_brier": brier_score_loss(y, proba),
    }


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def done_keys(name, filename, cols):
    path = os.path.join(out_dir(name), filename)
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(*[d[c] for c in cols]))


def save_row(name, filename, row):
    _append(os.path.join(out_dir(name), filename), pd.DataFrame([row]))


def log_env(name):
    lines = [f"platform: {platform.platform()}", f"python: {platform.python_version()}",
             f"cpus_used: {N_CPUS}", f"time_limit_s: {TIME_LIMIT}"]
    for pkg in ["scikit-learn", "imbalanced-learn", "pycaret", "autogluon.tabular", "lightgbm"]:
        try:
            lines.append(f"{pkg}: {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass
    with open(os.path.join(out_dir(name), "environment_v3.txt"), "a") as f:
        f.write("\n".join(lines) + "\n---\n")
