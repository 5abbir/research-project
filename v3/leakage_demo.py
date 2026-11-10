"""WRONG vs RIGHT: why some papers report >80% accuracy on weak-signal data.
WRONG-1: SMOTE (and imputation) applied to the WHOLE dataset, then split  -> synthetic copies of test rows leak into training.
WRONG-2: imputation fitted on the whole dataset, then split                -> small leak.
RIGHT  : split first; imputation + SMOTE fitted on the training part only; test set untouched.
Run in env_main:  python leakage_demo.py --csv ../data/raw.csv --target Potability --name water
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import common as c

args = c.get_args()
df = c.load(args.csv, args.target)
X_all, y_all = df.drop(columns=[args.target]), df[args.target]
rows = []


def rf(seed):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=c.N_CPUS)


def score(name, model, Xte, yte, rep):
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({"method": name, "repeat": rep, "acc": accuracy_score(yte, pred),
                 "bal_acc": balanced_accuracy_score(yte, pred), "f1": f1_score(yte, pred),
                 "auc": roc_auc_score(yte, proba)})


for rep in range(args.repeats):
    # WRONG-1: impute + SMOTE on everything, then split
    Xi = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X_all), columns=X_all.columns)
    Xs, ys = SMOTE(random_state=rep).fit_resample(Xi, y_all)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=rep, stratify=ys)
    score("WRONG-1 SMOTE before split", rf(rep).fit(Xtr, ytr), Xte, yte, rep)

    # WRONG-2: impute on everything, then split (no SMOTE)
    Xtr, Xte, ytr, yte = train_test_split(Xi, y_all, test_size=0.2, random_state=rep, stratify=y_all)
    score("WRONG-2 impute before split", rf(rep).fit(Xtr, ytr), Xte, yte, rep)

    # RIGHT: split first, then everything inside a pipeline fitted on train only
    train, test = c.get_split(df, args.target, rep)
    Xtr, ytr = train.drop(columns=[args.target]), train[args.target]
    Xte, yte = test.drop(columns=[args.target]), test[args.target]
    pipe = Pipeline([("imp", SimpleImputer(strategy="mean")), ("smote", SMOTE(random_state=rep)), ("rf", rf(rep))])
    score("RIGHT  split first + SMOTE in pipeline", pipe.fit(Xtr, ytr), Xte, yte, rep)
    print("repeat", rep, "done")

res = pd.DataFrame(rows)
g = res.groupby("method")[["acc", "bal_acc", "f1", "auc"]]
print("\n=== mean +/- std over repeats ===")
print((g.mean().round(3).astype(str) + " +/- " + g.std().round(3).astype(str)).to_string())
res.to_csv(f"{c.out_dir(args.name)}/leakage_demo.csv", index=False)
