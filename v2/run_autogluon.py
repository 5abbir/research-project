"""AutoGluon on every repeated split, conditions A_all and B_trees. Run in env_main:
  python run_autogluon.py --csv data/raw.csv --target Potability --name water
Safe to stop (Ctrl+C) and rerun: finished runs are skipped.
"""
import random
import shutil
import tempfile
import time

import numpy as np
from autogluon.tabular import TabularPredictor

import common as c

args = c.get_args()
df = c.load(args.csv, args.target)
c.log_env(args.name)
done = c.done_keys(args.name)

for rep in range(args.repeats):
    train, test = c.get_split(df, args.target, rep)
    for cond in c.CONDITIONS:
        if ("autogluon", cond, rep) in done:
            continue
        random.seed(rep)
        np.random.seed(rep)
        path = tempfile.mkdtemp(prefix="ag_")
        t0 = time.time()
        fit_kw = dict(time_limit=c.TIME_LIMIT, presets="medium_quality",
                      num_bag_folds=5, num_bag_sets=1, num_stack_levels=0,
                      num_cpus=c.N_CPUS, num_gpus=0)
        if cond == "B_trees":
            fit_kw["hyperparameters"] = c.TREE_AG
        predictor = TabularPredictor(label=args.target, problem_type="binary",
                                     eval_metric="roc_auc", path=path, verbosity=0).fit(train, **fit_kw)
        seconds = time.time() - t0

        lb = predictor.leaderboard(test, silent=True)
        single = lb[~lb["model"].str.startswith("WeightedEnsemble")]
        best = single.sort_values("score_val", ascending=False).iloc[0]
        proba = predictor.predict_proba(test, model=best["model"])[1]

        row = {"dataset": args.name, "tool": "autogluon", "condition": cond, "repeat": rep,
               "best_model": best["model"], "cv_auc": float(best["score_val"]),
               "seconds": seconds, **c.scores(test[args.target], proba)}
        c.save_row(args.name, row)
        lb["repeat"], lb["condition"] = rep, cond
        c.save_board(args.name, "autogluon_leaderboards.csv", lb)
        shutil.rmtree(path, ignore_errors=True)
        print(rep, cond, row["best_model"], round(row["cv_auc"], 4), round(row["test_auc"], 4),
              f"{seconds:.0f}s")
