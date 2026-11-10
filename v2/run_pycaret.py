"""PyCaret on every repeated split, conditions A_all and B_trees. Run in env_pycaret:
  python run_pycaret.py --csv data/raw.csv --target Potability --name water
Safe to stop (Ctrl+C) and rerun: finished runs are skipped.
"""
import time

import numpy as np
from pycaret.classification import compare_models, predict_model, pull, setup

import common as c

args = c.get_args()
df = c.load(args.csv, args.target)
c.log_env(args.name)
done = c.done_keys(args.name)

for rep in range(args.repeats):
    train, test = c.get_split(df, args.target, rep)
    for cond in c.CONDITIONS:
        if ("pycaret", cond, rep) in done:
            continue
        t0 = time.time()
        setup(data=train.copy(), target=args.target, session_id=rep,
              fold=5, fold_strategy="stratifiedkfold",
              n_jobs=c.N_CPUS, verbose=False, html=False)
        kw = dict(sort="AUC", n_select=1, verbose=False, budget_time=c.TIME_LIMIT / 60)
        if cond == "B_trees":
            kw["include"] = c.TREE_PYCARET
        best = compare_models(**kw)
        board = pull().copy()
        seconds = time.time() - t0

        pred = predict_model(best, data=test.copy(), verbose=False)
        label = pred["prediction_label"].astype(int)
        proba = np.where(label == 1, pred["prediction_score"], 1 - pred["prediction_score"])

        row = {"dataset": args.name, "tool": "pycaret", "condition": cond, "repeat": rep,
               "best_model": board.iloc[0]["Model"], "cv_auc": board.iloc[0]["AUC"],
               "seconds": seconds, **c.scores(test[args.target], proba)}
        c.save_row(args.name, row)
        board["repeat"], board["condition"] = rep, cond
        c.save_board(args.name, "pycaret_leaderboards.csv", board.reset_index())
        print(rep, cond, row["best_model"], round(row["cv_auc"], 4), round(row["test_auc"], 4),
              f"{seconds:.0f}s")
