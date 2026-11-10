"""LESSON 2 - Does min-max scaling help? (imputer is now FIXED to the Lesson 1 winner: mean)
Same idea as Lesson 1: only ONE thing changes (scaling), same folds, training data only, all inside a pipeline.
We test 1 tree model (should not care) and 4 models that DO care about the size of numbers.
Run:  python lesson2_scaling.py --train ../data_cls/train.csv --target Potability
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

p = argparse.ArgumentParser()
p.add_argument("--train", default="../data_cls/train.csv")
p.add_argument("--target", default="Potability")
p.add_argument("--cpus", type=int, default=8)
a = p.parse_args()
df = pd.read_csv(a.train)
X, y = df.drop(columns=[a.target]), df[a.target]

models = {   # name -> function that builds a fresh model
    "Random Forest (trees)": lambda: RandomForestClassifier(n_estimators=300, n_jobs=a.cpus, random_state=0),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=15),
    "SVM (RBF)": lambda: SVC(probability=True, random_state=0),
    "MLP (small neural net)": lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0),
    "Logistic regression": lambda: LogisticRegression(max_iter=2000),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42)   # same folds as Lesson 1
J, ratio = 20, 0.25
rows = []
for mname, build in models.items():
    res = {}
    for sname in ["none", "minmax"]:
        steps = [SimpleImputer(strategy="mean")]            # the fixed imputer
        if sname == "minmax":
            steps.append(MinMaxScaler())                    # learned from training rows of each fold only
        steps.append(build())
        r = cross_validate(make_pipeline(*steps), X, y, cv=cv,
                           scoring={"auc": "roc_auc", "acc": "accuracy", "bal_acc": "balanced_accuracy"})
        res[sname] = r
    d = res["minmax"]["test_auc"] - res["none"]["test_auc"]
    t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1)) if d.std() > 0 else 0.0
    rows.append({"model": mname,
                 "AUC none": f"{res['none']['test_auc'].mean():.4f}",
                 "AUC minmax": f"{res['minmax']['test_auc'].mean():.4f}",
                 "diff (minmax-none)": round(d.mean(), 4),
                 "p_value": round(2 * stats.t.sf(abs(t), J - 1), 3),
                 "bal_acc none": f"{res['none']['test_bal_acc'].mean():.4f}",
                 "bal_acc minmax": f"{res['minmax']['test_bal_acc'].mean():.4f}"})
    print("done:", mname)

table = pd.DataFrame(rows)
print("\n=== Lesson 2: effect of min-max scaling (20 validation folds, training data only) ===")
print(table.to_string(index=False))
os.makedirs("results", exist_ok=True)
table.to_csv("results/lesson2_scaling.csv", index=False)
print("Saved results/lesson2_scaling.csv")
