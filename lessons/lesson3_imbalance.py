"""LESSON 3 - Which imbalance fix helps? ONE change at a time. Imputer = mean, scaler = min-max (Lessons 1-2).
Order inside every pipeline:  mean-fill -> min-max -> [imbalance fix] -> model
The fix (SMOTE etc.) is applied ONLY to the training rows of each fold, never to the validation rows.
Training data only; the real test set is never opened.
Run:  python lesson3_imbalance.py --train ../data_cls/train.csv --target Potability --model rf
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--train", default="../data_cls/train.csv")
p.add_argument("--target", default="Potability")
p.add_argument("--model", default="rf", choices=["rf", "svm"])
p.add_argument("--cpus", type=int, default=8)
p.add_argument("--quick", action="store_true", help="tiny smoke test: fewer trees, 5 folds x 1")
a = p.parse_args()
df = pd.read_csv(a.train)
X, y = df.drop(columns=[a.target]), df[a.target]


def model(class_weight=None):
    if a.model == "rf":
        return RandomForestClassifier(n_estimators=40 if a.quick else 300, n_jobs=a.cpus, random_state=0, class_weight=class_weight)
    return SVC(random_state=0, class_weight=class_weight)       # AUC uses decision_function, so no probability needed


def build(sampler=None, class_weight=None):
    steps = [("imp", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]
    if sampler is not None:
        steps.append(("fix", sampler))
    steps.append(("model", model(class_weight)))
    return Pipeline(steps)


candidates = {   # simplest first
    "none": lambda: build(),
    "class weights (balanced)": lambda: build(class_weight="balanced"),
    "random oversampling": lambda: build(RandomOverSampler(random_state=0)),
    "random undersampling": lambda: build(RandomUnderSampler(random_state=0)),
    "SMOTE": lambda: build(SMOTE(random_state=0)),
    "Borderline-SMOTE": lambda: build(BorderlineSMOTE(random_state=0)),
    "ADASYN": lambda: build(ADASYN(random_state=0)),
    "SMOTE-Tomek": lambda: build(SMOTETomek(random_state=0)),
    "SMOTE-ENN": lambda: build(SMOTEENN(random_state=0)),
}
try:   # threshold moving: the cut-off is tuned by an INNER cross-validation on training rows only
    from sklearn.model_selection import TunedThresholdClassifierCV
    candidates["threshold moving"] = lambda: TunedThresholdClassifierCV(
        build(), scoring="balanced_accuracy", cv=5, refit=True, random_state=0)
except ImportError:
    print("-- 'threshold moving' skipped: needs scikit-learn >= 1.5")

REPEATS = 1 if a.quick else 4
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=42)
scoring = {"auc": "roc_auc", "acc": "accuracy", "bal_acc": "balanced_accuracy", "f1": "f1",
           "mcc": make_scorer(matthews_corrcoef)}
results = {}
for name, make in candidates.items():
    try:
        r = cross_validate(make(), X, y, cv=cv, scoring=scoring, error_score="raise")
    except Exception as e:
        print(f"-- skipped '{name}': {type(e).__name__}: {str(e)[:70]}")
        continue
    results[name] = {k: r[f"test_{k}"] for k in scoring}
    print("done:", name)

rows = [{"fix": n, **{k: f"{v.mean():.4f} +/- {v.std():.4f}" for k, v in r.items()}, "_s": r["bal_acc"].mean()}
        for n, r in results.items()]
print(f"\n=== Lesson 3 ({a.model}): mean +/- std over {5 * REPEATS} validation folds (training data only) ===")
print(pd.DataFrame(rows).sort_values("_s", ascending=False).drop(columns="_s").to_string(index=False))

J, ratio = 5 * REPEATS, 0.25
for metric in ["auc", "bal_acc", "f1"]:
    out = []
    for n, r in results.items():
        if n == "none":
            continue
        d = r[metric] - results["none"][metric]
        t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1)) if d.std() > 0 else 0.0
        out.append({"fix": n, "mean_diff": round(d.mean(), 4), "p_value": round(2 * stats.t.sf(abs(t), J - 1), 3)})
    print(f"\n--- {metric.upper()} difference vs 'none' (positive = better; corrected t-test) ---")
    print(pd.DataFrame(out).sort_values("mean_diff", ascending=False).to_string(index=False))

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).drop(columns="_s").to_csv(f"results/lesson3_{a.model}.csv", index=False)
print(f"\nSaved results/lesson3_{a.model}.csv")
