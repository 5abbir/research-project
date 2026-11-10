"""Baselines on every repeated split. Run in env_main:
  python run_baselines.py --csv data/raw.csv --target Potability --name water
"""
import time

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as c

args = c.get_args()
df = c.load(args.csv, args.target)
c.log_env(args.name)
done = c.done_keys(args.name)


def make_models(seed):
    return {
        "baseline_dummy": DummyClassifier(strategy="prior"),
        "baseline_logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                         LogisticRegression(max_iter=2000)),
        "baseline_rf_default": make_pipeline(SimpleImputer(strategy="median"),
                                             RandomForestClassifier(n_estimators=500, random_state=seed,
                                                                    n_jobs=c.N_CPUS)),
        "baseline_hgb_default": HistGradientBoostingClassifier(random_state=seed),
    }


for rep in range(args.repeats):
    train, test = c.get_split(df, args.target, rep)
    Xtr, ytr = train.drop(columns=[args.target]), train[args.target]
    Xte, yte = test.drop(columns=[args.target]), test[args.target]
    cv = StratifiedKFold(5, shuffle=True, random_state=rep)
    for name, model in make_models(rep).items():
        if (name, "-", rep) in done:
            continue
        t0 = time.time()
        cv_auc = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        row = {"dataset": args.name, "tool": name, "condition": "-", "repeat": rep,
               "best_model": name, "cv_auc": cv_auc, "seconds": time.time() - t0,
               **c.scores(yte, proba)}
        c.save_row(args.name, row)
        print(rep, name, round(row["cv_auc"], 4), round(row["test_auc"], 4))
