"""LESSON 1 - Compare ways of filling blanks, ONE change at a time (same model, same folds).
Only the TRAINING file is used. The imputer sits INSIDE a pipeline, so in every fold it learns only from
that fold's training rows (no leakage). No scaling, no resampling: we change ONLY the imputation.
Run:  python lesson1_imputation.py --train ../data_cls/train.csv --target Potability --model rf
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (needed to unlock IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------- settings
p = argparse.ArgumentParser()
p.add_argument("--train", default="../data_cls/train.csv")
p.add_argument("--target", default="Potability")
p.add_argument("--model", default="rf", choices=["rf", "et", "hgb"])
p.add_argument("--cpus", type=int, default=8)
a = p.parse_args()
df = pd.read_csv(a.train)
X, y = df.drop(columns=[a.target]), df[a.target]


# ---------------------------------------------------------------- the one model used for every candidate
def make_model():
    if a.model == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=a.cpus, random_state=0)
    if a.model == "et":
        return ExtraTreesClassifier(n_estimators=300, n_jobs=a.cpus, random_state=0)
    return HistGradientBoostingClassifier(random_state=0)


# ---------------------------------------------------------------- hot deck (written out so you can read it)
class HotDeck(BaseEstimator, TransformerMixin):
    """fit: remember real observed values of each column. transform: fill a blank with a randomly chosen one."""

    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.donors_ = [col[~np.isnan(col)] for col in X.T]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        rng = np.random.RandomState(self.random_state)
        for j in range(X.shape[1]):
            blank = np.isnan(X[:, j])
            if blank.any():
                X[blank, j] = rng.choice(self.donors_[j], size=int(blank.sum()), replace=True)
        return X


# ---------------------------------------------------------------- candidates, simplest first
candidates = {
    "native (leave blanks)": lambda: make_pipeline(make_model()),
    "zero (AutoGluon-RF habit)": lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=0), make_model()),
    "mean": lambda: make_pipeline(SimpleImputer(strategy="mean"), make_model()),
    "median": lambda: make_pipeline(SimpleImputer(strategy="median"), make_model()),
    "median + indicator": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), make_model()),
    "hot deck": lambda: make_pipeline(HotDeck(random_state=0), make_model()),
    "KNN (k=5, scaled)": lambda: make_pipeline(MinMaxScaler(), KNNImputer(n_neighbors=5), make_model()),
    "iterative (MICE-like)": lambda: make_pipeline(IterativeImputer(max_iter=10, random_state=0), make_model()),
}
BASE = "median"

# ---------------------------------------------------------------- evaluation: 5-fold x 4 repeats = 20 scores each
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42)   # identical folds for every candidate
scoring = {"auc": "roc_auc", "acc": "accuracy", "bal_acc": "balanced_accuracy", "f1": "f1"}
results = {}
for name, build in candidates.items():
    try:
        r = cross_validate(build(), X, y, cv=cv, scoring=scoring, error_score="raise")
    except ValueError as e:
        print(f"-- skipped '{name}': this model cannot accept blanks ({str(e)[:60]}...)")
        continue
    results[name] = {k: r[f"test_{k}"] for k in scoring}
    print(f"done: {name}")

# ---------------------------------------------------------------- table
rows = []
for name, r in results.items():
    rows.append({"imputer": name, **{k: f"{v.mean():.4f} +/- {v.std():.4f}" for k, v in r.items()},
                 "_auc": r["auc"].mean()})
table = pd.DataFrame(rows).sort_values("_auc", ascending=False).drop(columns="_auc")
print(f"\n=== Lesson 1 ({a.model}): mean +/- std over 20 validation folds (training data only) ===")
print(table.to_string(index=False))

# ---------------------------------------------------------------- paired comparison against the median baseline
J, ratio = 20, 0.25          # 20 scores; validation/training size ratio = 1/4 for 5 folds
print(f"\n=== Paired AUC difference vs '{BASE}' (positive = better). Corrected t-test (Nadeau & Bengio) ===")
out = []
for name, r in results.items():
    if name == BASE or BASE not in results:
        continue
    d = r["auc"] - results[BASE]["auc"]
    t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1)) if d.std() > 0 else 0.0
    out.append({"imputer": name, "mean_diff": round(d.mean(), 4), "p_value": round(2 * stats.t.sf(abs(t), J - 1), 3)})
print(pd.DataFrame(out).sort_values("mean_diff", ascending=False).to_string(index=False))

# ---------------------------------------------------------------- simple, honest decision rule
best = max(results, key=lambda n: results[n]["auc"].mean())
se = results[best]["auc"].std() / np.sqrt(J)
for name in candidates:                                   # simplest first
    if name in results and results[name]["auc"].mean() >= results[best]["auc"].mean() - se:
        print(f"\nBest mean AUC: '{best}'. One-standard-error rule picks the SIMPLEST imputer within "
              f"{se:.4f} of it: '{name}'.")
        break

os.makedirs("results", exist_ok=True)
table.to_csv(f"results/lesson1_{a.model}.csv", index=False)
print(f"Saved results/lesson1_{a.model}.csv")
