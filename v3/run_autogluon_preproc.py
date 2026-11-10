"""AutoGluon on the chosen preprocessing combinations. Run in env_main:
  python run_autogluon_preproc.py --csv ../data/raw.csv --target Potability --name water --combos top:4
AutoGluon cannot resample inside its own CV, so we use a leak-free HOLDOUT design instead:
  train (80%) -> inner-train (80% of train) + validation (20% of train, original class balance)
  imputer/scaler are fitted on inner-train ONLY, applied to validation and test;
  resampling (SMOTE...) is applied to inner-train ONLY; the validation and test sets are never resampled;
  AutoGluon trains on the resampled inner-train and selects models on the untouched validation set.
Note: AutoGluon therefore sees 64% of all rows (PyCaret sees 80%). State this in the paper.
"""
import random
import shutil
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

import common as c
from preprocessors import make_imputer, make_resampler, make_scaler

warnings.filterwarnings("ignore")
args = c.get_args()
df = c.load(args.csv, args.target)
c.log_env(args.name)
combos = c.resolve_combos(args.combos, args.name, args.rank_by)
done = c.done_keys(args.name, "autogluon_raw.csv", ["combo", "repeat"])
feats = [col for col in df.columns if col != args.target]
print("combos to run:", len(combos), "| repeats:", args.repeats)

for rep in range(args.repeats):
    train, test = c.get_split(df, args.target, rep)
    inner, valid = train_test_split(train, test_size=0.2, random_state=rep, stratify=train[args.target])
    for combo in combos:
        if (combo, rep) in done:
            continue
        imp, sc, res = combo.split("|")
        t0 = time.time()
        random.seed(rep)
        np.random.seed(rep)
        # ---- fit preprocessing on inner-train only
        imputer = make_imputer(imp, rep).fit(inner[feats])
        scaler = make_scaler(sc)
        if scaler is not None:
            scaler.fit(imputer.transform(inner[feats]))

        def tf(d):
            X = imputer.transform(d[feats])
            if scaler is not None:
                X = scaler.transform(X)
            return pd.DataFrame(X, columns=feats)

        Xtr, Xva, Xte = tf(inner), tf(valid), tf(test)
        ytr = inner[args.target].values
        sampler = make_resampler(res, rep)
        try:
            if sampler is not None:
                Xtr, ytr = sampler.fit_resample(Xtr, ytr)      # training rows only
        except Exception as e:
            print(f"!! {rep} {combo} resampling failed: {type(e).__name__}: {e}")
            continue
        train_df = pd.concat([pd.DataFrame(Xtr).reset_index(drop=True),
                              pd.Series(np.asarray(ytr), name=args.target)], axis=1)
        valid_df = Xva.assign(**{args.target: valid[args.target].values})

        path = tempfile.mkdtemp(prefix="ag_")
        fit_kw = dict(time_limit=60 if args.quick else c.TIME_LIMIT, presets="medium_quality",
                      num_cpus=c.N_CPUS, num_gpus=0)
        if args.quick:
            fit_kw["hyperparameters"] = {"RF": {}, "GBM": {}}
        try:
            predictor = TabularPredictor(label=args.target, problem_type="binary", eval_metric="roc_auc",
                                         path=path, verbosity=0).fit(train_df, tuning_data=valid_df, **fit_kw)
            test_df = Xte.assign(**{args.target: test[args.target].values})
            lb = predictor.leaderboard(test_df, silent=True)
            single = lb[~lb["model"].str.startswith("WeightedEnsemble")]
            best = single.sort_values("score_val", ascending=False).iloc[0]
            proba = predictor.predict_proba(Xte, model=best["model"])[1]
        except Exception as e:
            print(f"!! {rep} {combo} failed: {type(e).__name__}: {e}")
            shutil.rmtree(path, ignore_errors=True)
            continue
        shutil.rmtree(path, ignore_errors=True)
        row = {"dataset": args.name, "tool": "autogluon", "combo": combo, "imputer": imp, "scaler": sc,
               "resampler": res, "repeat": rep, "best_model": best["model"],
               "val_auc": float(best["score_val"]), "seconds": time.time() - t0,
               **c.scores(test[args.target], proba)}
        c.save_row(args.name, "autogluon_raw.csv", row)
        print(f"{rep} {combo:26s} {row['best_model'][:22]:22s} valAUC={row['val_auc']:.4f} "
              f"testAUC={row['test_auc']:.4f} testAcc={row['test_acc']:.4f} ({row['seconds']:.0f}s)")
