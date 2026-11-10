"""Leak-free screening of every preprocessing combination with a fast model (Random Forest).
Pipeline order inside each fold: impute -> scale -> resample -> model.
Resampling happens ONLY when fitting (imblearn Pipeline), never on validation folds or the test set.
Run in env_main:  python screen_preprocessing.py --csv ../data/raw.csv --target Potability --name water
"""
import time

import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate

import common as c
from preprocessors import make_imputer, make_resampler, make_scaler

N_TREES = 200
args = c.get_args()
df = c.load(args.csv, args.target)
c.log_env(args.name)
combos = c.resolve_combos(args.combos, args.name, args.rank_by)
done = c.done_keys(args.name, "screen_raw.csv", ["combo", "repeat"])
SCORING = {"auc": "roc_auc", "acc": "accuracy", "bal_acc": "balanced_accuracy", "f1": "f1",
           "mcc": make_scorer(matthews_corrcoef)}


def build(combo, seed):
    imp, sc, res = combo.split("|")
    steps = [("imp", make_imputer(imp, seed))]
    if make_scaler(sc) is not None:
        steps.append(("scale", make_scaler(sc)))
    if make_resampler(res, seed) is not None:
        steps.append(("res", make_resampler(res, seed)))
    steps.append(("rf", RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=c.N_CPUS)))
    return Pipeline(steps)


for rep in range(args.repeats):
    train, test = c.get_split(df, args.target, rep)
    Xtr, ytr = train.drop(columns=[args.target]), train[args.target]
    Xte, yte = test.drop(columns=[args.target]), test[args.target]
    cv = StratifiedKFold(5, shuffle=True, random_state=rep)
    for combo in combos:
        if (combo, rep) in done:
            continue
        t0 = time.time()
        try:
            pipe = build(combo, rep)
            cvres = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=SCORING, error_score="raise")
            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xte)[:, 1]
        except Exception as e:  # e.g. ADASYN can fail on odd data; keep going and report
            print(f"!! {rep} {combo} failed: {type(e).__name__}: {e}")
            continue
        imp, sc, res = combo.split("|")
        row = {"dataset": args.name, "combo": combo, "imputer": imp, "scaler": sc, "resampler": res,
               "repeat": rep, **{f"cv_{k}": float(np.mean(cvres[f"test_{k}"])) for k in SCORING},
               "seconds": time.time() - t0, **c.scores(yte, proba)}
        c.save_row(args.name, "screen_raw.csv", row)
        print(f"{rep} {combo:28s} cvAUC={row['cv_auc']:.4f} testAUC={row['test_auc']:.4f} "
              f"testAcc={row['test_acc']:.4f} balAcc={row['test_bal_acc']:.4f} ({row['seconds']:.0f}s)")
