"""LESSON 3b - GAN-based oversampling vs SMOTE vs nothing (same pipeline and same folds as Lesson 3).
  mean-fill -> min-max -> [sampler] -> model        (the sampler touches TRAINING rows of each fold only)
Candidates: none | class weights | SMOTE | GAN | SMOTified-GAN | GANified-SMOTE
Run:  python lesson3b_gan.py --train ../data_cls/train.csv --target Potability --model rf --quick     (smoke test)
      python lesson3b_gan.py --train ../data_cls/train.csv --target Potability --model rf
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from gan_samplers import GANSampler

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--train", default="../data_cls/train.csv")
p.add_argument("--target", default="Potability")
p.add_argument("--model", default="rf", choices=["rf", "svm"])
p.add_argument("--cpus", type=int, default=8)
p.add_argument("--epochs", type=int, default=200, help="GAN training epochs per fit")
p.add_argument("--quick", action="store_true", help="smoke test: 15 epochs, 5 folds x 1")
a = p.parse_args()
EPOCHS = 15 if a.quick else a.epochs
df = pd.read_csv(a.train)
X, y = df.drop(columns=[a.target]), df[a.target]


def model(class_weight=None):
    if a.model == "rf":
        return RandomForestClassifier(n_estimators=40 if a.quick else 300, n_jobs=a.cpus, random_state=0,
                                      class_weight=class_weight)
    return SVC(random_state=0, class_weight=class_weight)


def build(sampler=None, class_weight=None):
    steps = [("imp", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]
    if sampler is not None:
        steps.append(("fix", sampler))
    steps.append(("model", model(class_weight)))
    return Pipeline(steps)


candidates = {
    "none": lambda: build(),
    "class weights (balanced)": lambda: build(class_weight="balanced"),
    "SMOTE": lambda: build(SMOTE(random_state=0)),
    "GAN only": lambda: build(GANSampler("gan", epochs=EPOCHS, random_state=0)),
    "SMOTified-GAN": lambda: build(GANSampler("smotified_gan", epochs=EPOCHS, random_state=0)),
    "GANified-SMOTE": lambda: build(GANSampler("ganified_smote", epochs=EPOCHS, random_state=0)),
}
REPEATS = 1 if a.quick else 4
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=42)     # same folds as Lessons 1-3
scoring = {"auc": "roc_auc", "acc": "accuracy", "bal_acc": "balanced_accuracy", "f1": "f1",
           "mcc": make_scorer(matthews_corrcoef)}
results = {}
for name, make in candidates.items():
    try:
        r = cross_validate(make(), X, y, cv=cv, scoring=scoring, error_score="raise")
    except Exception as e:
        print(f"-- skipped '{name}': {type(e).__name__}: {str(e)[:80]}")
        continue
    results[name] = {k: r[f"test_{k}"] for k in scoring}
    print("done:", name)

rows = [{"fix": n, **{k: f"{v.mean():.4f} +/- {v.std():.4f}" for k, v in r.items()}, "_s": r["bal_acc"].mean()}
        for n, r in results.items()]
print(f"\n=== Lesson 3b ({a.model}): mean +/- std over {5 * REPEATS} validation folds (training data only) ===")
print(pd.DataFrame(rows).sort_values("_s", ascending=False).drop(columns="_s").to_string(index=False))

J, ratio = 5 * REPEATS, 0.25
for base in ["none", "SMOTE"]:
    for metric in ["auc", "bal_acc", "f1"]:
        out = []
        for n, r in results.items():
            if n == base or base not in results:
                continue
            d = r[metric] - results[base][metric]
            t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1)) if d.std() > 0 else 0.0
            out.append({"fix": n, "mean_diff": round(d.mean(), 4), "p_value": round(2 * stats.t.sf(abs(t), J - 1), 3)})
        print(f"\n--- {metric.upper()} difference vs '{base}' (positive = better; corrected t-test) ---")
        print(pd.DataFrame(out).sort_values("mean_diff", ascending=False).to_string(index=False))

print("\n=== Sanity check of the synthetic rows (fit once on all training rows) ===")
print("std_ratio = spread of synthetic rows / spread of real minority rows. Below ~0.3 means mode collapse.\n(It only catches collapse; it does NOT prove the rows are realistic.)")
for name in ["GAN only", "SMOTified-GAN", "GANified-SMOTE"]:
    if name in results:
        pipe = candidates[name]().fit(X, y)
        rep = pipe.named_steps["fix"].report_
        print(f"{name:16s} synthetic rows = {rep['n_synthetic']:5d}   std_ratio = {rep['std_ratio_vs_real']:.2f}")

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).drop(columns="_s").to_csv(f"results/lesson3b_{a.model}.csv", index=False)
print(f"\nSaved results/lesson3b_{a.model}.csv")
