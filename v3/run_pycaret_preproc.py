"""PyCaret on the chosen preprocessing combinations. Run in env_pycaret:
  python run_pycaret_preproc.py --csv ../data/raw.csv --target Potability --name water --combos all
Leak-free: imputer -> scaler -> (PyCaret's own) resampler are refit inside every CV fold, resampling
only touches training folds. Resumable: finished (combo, repeat) pairs are skipped.
FIX vs v2: test_data=... makes PyCaret train/CV on ALL given training rows (v2 silently used only 70%).
"""
import time
import warnings

import numpy as np
from pycaret.classification import compare_models, predict_model, pull, setup

import common as c
from preprocessors import make_imputer, make_resampler, make_scaler

warnings.filterwarnings("ignore")
args = c.get_args()
df = c.load(args.csv, args.target)
c.log_env(args.name)
combos = c.resolve_combos(args.combos, args.name, args.rank_by)
done = c.done_keys(args.name, "pycaret_raw.csv", ["combo", "repeat"])
print("combos to run:", len(combos), "| repeats:", args.repeats)

for rep in range(args.repeats):
    train, test = c.get_split(df, args.target, rep)
    for combo in combos:
        if (combo, rep) in done:
            continue
        imp, sc, res = combo.split("|")
        t0 = time.time()
        custom = [("imp", make_imputer(imp, rep))]
        if make_scaler(sc) is not None:
            custom.append(("scale", make_scaler(sc)))   # scaling BEFORE resampling (distance-based SMOTE)
        kw = dict(data=train.copy(), test_data=test.copy(), target=args.target, session_id=rep,
                  fold=5, fold_strategy="stratifiedkfold", n_jobs=c.N_CPUS, verbose=False, html=False,
                  imputation_type=None, normalize=False,
                  custom_pipeline=custom, custom_pipeline_position=0)
        if res != "none":
            kw.update(fix_imbalance=True, fix_imbalance_method=make_resampler(res, rep))
        try:
            setup(**kw)
            ck = dict(sort="AUC", n_select=1, verbose=False, budget_time=c.TIME_LIMIT / 60)
            if args.quick:
                ck.update(include=["rf", "lr", "dt"], budget_time=1)
            best = compare_models(**ck)
            board = pull().copy()
            pred = predict_model(best, data=test.copy(), verbose=False)
        except Exception as e:
            print(f"!! {rep} {combo} failed: {type(e).__name__}: {e}")
            continue
        label = pred["prediction_label"].astype(int)
        proba = np.where(label == 1, pred["prediction_score"], 1 - pred["prediction_score"])
        row = {"dataset": args.name, "tool": "pycaret", "combo": combo, "imputer": imp, "scaler": sc,
               "resampler": res, "repeat": rep, "best_model": board.iloc[0]["Model"],
               "cv_auc": board.iloc[0]["AUC"], "cv_acc": board.iloc[0]["Accuracy"],
               "cv_f1": board.iloc[0]["F1"], "seconds": time.time() - t0,
               **c.scores(test[args.target], proba)}
        c.save_row(args.name, "pycaret_raw.csv", row)
        print(f"{rep} {combo:26s} {row['best_model'][:22]:22s} cvAUC={row['cv_auc']:.4f} "
              f"testAUC={row['test_auc']:.4f} testAcc={row['test_acc']:.4f} ({row['seconds']:.0f}s)")
