#!/usr/bin/env python
"""
r3_fill_trackL.py  --  Scenario 2, Track L (LEAK-FREE, PRIMARY): Section 4.2 filling of null values.

LABEL: "Section 4.2 correlation-based imputation", Track L. Reproduction results, not validated performance.

Section 4.2 steps as implemented (decisions in S2_PROTOCOL.md):
  1 identify columns with nulls (counted on TRAIN rows)
  2 correlation chart = Pearson r on pairwise-complete TRAIN rows, computed ONCE before any filling (r3_corr_table.py)
  3 for each incomplete column, in order fewest-missing-first (B5): predictors = columns that are fully filled at that step
    (complete columns + earlier filled columns) AND have Pearson r > --rule-min-r (B3, default 0 = "positive") with that column.
    If NO predictor qualifies: STOP AND ASK (exit code 3), never a silent fallback.
  4 drop rows where the column is null, tune a RandomForestRegressor by grid search (5-fold CV, neg MSE) on those TRAIN rows,
    refit the best model on all of them, predict the null cells of the TRAIN rows and of the TEST rows
  5 repeat until all columns are filled

Leakage rules: fill models and the correlation rule see TRAIN rows only; test rows are only transformed with train-fitted models;
the target (Potability) is carried through but is NEVER given to any fill model (asserted).

Modes: --smoke : train rows only (subsample), inner split; test rows not used; output -> <out>/fill_smoke/. Not reportable.
       default : official split (stratified 80/20, seed 42).
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_s1_sklearn_models import sha256_file, split_indices  # noqa: E402

GRID = {"n_estimators": [100, 300], "max_depth": [None, 10], "min_samples_leaf": [1, 5, 25]}


def pairwise_r(a, b):
    m = a.notna() & b.notna()
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--rule-min-r", type=float, default=0.0, help="B3: predictor needs Pearson r > this value (0 = 'positive')")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=800)
    ap.add_argument("--check-split-with", default=None, help="results/s1/s1_predictions.csv (test row ids must match)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        sys.exit(f"STOP: target '{args.target}' not found. Columns: {list(df.columns)}")
    y_all = df[args.target].to_numpy()
    if set(np.unique(y_all)) != {0, 1}:
        sys.exit("STOP: target must contain exactly the labels 0 and 1")
    train_idx, test_idx = split_indices(y_all, args.seed)
    test_set = set(test_idx.tolist())
    mode = "smoke" if args.smoke else "full"
    out_dir = os.path.join(args.out, "fill_smoke") if args.smoke else args.out
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke:
        df = df.iloc[train_idx]  # test rows leave memory here
        sub, _ = train_test_split(train_idx, train_size=args.smoke_n, random_state=args.seed, stratify=y_all[train_idx])
        fit_idx, eval_idx = train_test_split(sub, test_size=0.2, random_state=args.seed, stratify=y_all[sub])
        fit_idx, eval_idx = np.sort(fit_idx), np.sort(eval_idx)
        assert not ((set(fit_idx.tolist()) | set(eval_idx.tolist())) & test_set), "SMOKE GUARD FAILED"
        assert set(df.index.tolist()).isdisjoint(test_set), "SMOKE GUARD FAILED (test rows in memory)"
        test_rows_used = 0
    else:
        fit_idx, eval_idx = train_idx, test_idx
        test_rows_used = len(test_idx)

    feats = [c for c in df.columns if c != args.target]
    Xtr = df.loc[fit_idx, feats].copy()
    Xte = df.loc[eval_idx, feats].copy()
    ytr = df.loc[fit_idx, args.target].to_numpy()
    yte = df.loc[eval_idx, args.target].to_numpy()
    n_miss_tr = Xtr.isna().sum()
    incomplete = [c for c in feats if n_miss_tr[c] > 0]
    order = sorted(incomplete, key=lambda c: (int(n_miss_tr[c]), feats.index(c)))  # B5 fewest missing first
    complete = [c for c in feats if n_miss_tr[c] == 0]
    assert order, "no incomplete column in the training rows"

    rep = [f"mode: {mode}" + ("  (NOT reportable)" if args.smoke else ""),
           f"csv sha256: {sha256_file(args.csv)}",
           f"seed {args.seed}; official train {len(train_idx)}, official test {len(test_idx)}",
           f"rows given to the fill step: train-side {len(Xtr)}, test-side {len(Xte)}; test rows used for fitting: 0",
           ("TEST ROWS USED: 0 (smoke)" if args.smoke else "test rows are only TRANSFORMED with train-fitted models"),
           f"missing cells in train-side rows: { {c: int(n_miss_tr[c]) for c in order} }; fill order (B5): {order}",
           f"complete columns at start: {complete}",
           f"B3 rule: Pearson r > {args.rule_min_r} (r from pairwise-complete train-side rows, computed once before filling)",
           f"grid: {GRID}; 5-fold KFold(shuffle, seed {args.seed}); scoring neg_mean_squared_error"]

    if (not args.smoke) and args.check_split_with and os.path.exists(args.check_split_with):
        same = set(pd.read_csv(args.check_split_with)["row_id"].tolist()) == set(eval_idx.tolist())
        rep.append(f"split check vs S1 sklearn run: {'IDENTICAL test rows' if same else 'MISMATCH'}")
        if not same:
            print("\n".join(rep))
            sys.exit("STOP: test rows differ from the S1 run. Ask the user.")

    # step 2: correlations, once, on train-side rows, before any filling
    r_tab = {(a, b): pairwise_r(Xtr[a], Xtr[b]) for a in incomplete for b in feats if a != b}

    filled = list(complete)
    models_rows, detail = [], []
    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for col in order:
        cands = [c for c in filled if r_tab[(col, c)] > args.rule_min_r]
        detail.append(f"column {col}: fully-filled columns at this step {filled}; r with {col}: "
                      f"{ {c: round(r_tab[(col, c)], 4) for c in filled} }; selected predictors {cands}")
        if not cands:
            print("\n".join(rep + detail))
            print(f"STOP AND ASK: no predictor satisfies r > {args.rule_min_r} for column '{col}' at this step. "
                  "No fallback was applied. Do not change the rule without the user.", file=sys.stderr)
            sys.exit(3)
        assert args.target not in cands, "target must never be a predictor (B4)"
        obs = Xtr[col].notna()
        Xf, yf = Xtr.loc[obs, cands], Xtr.loc[obs, col]
        assert Xf.notna().all().all(), "predictors must be complete at this step"
        t0 = time.time()
        gs = GridSearchCV(RandomForestRegressor(random_state=args.seed, n_jobs=args.n_jobs), GRID, cv=cv,
                          scoring="neg_mean_squared_error", refit=True, n_jobs=1)
        gs.fit(Xf, yf)
        mse_best = -float(gs.best_score_)
        mse_mean = -float(cross_val_score(DummyRegressor(strategy="mean"), Xf, yf, cv=cv, scoring="neg_mean_squared_error").mean())
        r2_rel = 1 - mse_best / mse_mean
        miss_tr, miss_te = ~obs, Xte[col].isna()
        pred_tr = gs.predict(Xtr.loc[miss_tr, cands]) if miss_tr.any() else np.array([])
        if miss_te.any():
            assert Xte.loc[miss_te, cands].notna().all().all(), "test predictors incomplete: earlier fill failed"
            pred_te = gs.predict(Xte.loc[miss_te, cands])
        else:
            pred_te = np.array([])
        Xtr.loc[miss_tr, col] = pred_tr
        Xte.loc[miss_te, col] = pred_te
        filled.append(col)
        models_rows.append({
            "column": col, "predictors": ";".join(cands), "n_fit_rows": int(obs.sum()),
            "best_params": json.dumps(gs.best_params_), "cv_mse_best": round(mse_best, 6),
            "cv_mse_mean_predictor": round(mse_mean, 6), "cv_r2_vs_mean_predictor": round(r2_rel, 5),
            "better_than_mean_predictor": bool(r2_rel > 0), "n_filled_train": int(miss_tr.sum()),
            "n_filled_test": int(miss_te.sum()), "observed_mean": round(float(yf.mean()), 4),
            "observed_sd": round(float(yf.std()), 4),
            "filled_train_mean": round(float(pred_tr.mean()), 4) if len(pred_tr) else np.nan,
            "filled_train_sd": round(float(pred_tr.std()), 4) if len(pred_tr) else np.nan,
            "filled_test_mean": round(float(pred_te.mean()), 4) if len(pred_te) else np.nan,
            "seconds": round(time.time() - t0, 1)})
        print(f"[{col}] fit rows {int(obs.sum())}, predictors {cands}, best {gs.best_params_}, "
              f"CV R2 vs mean-predictor {r2_rel:+.4f}, filled train {int(miss_tr.sum())} / test {int(miss_te.sum())}", flush=True)

    assert not Xtr.isna().any().any() and not Xte.isna().any().any(), "NaN remain after filling"
    tr_out = Xtr.assign(row_id=fit_idx, **{args.target: ytr})
    te_out = Xte.assign(row_id=eval_idx, **{args.target: yte})
    tr_out.to_csv(os.path.join(out_dir, "s2_filled_train_L.csv"), index=False)
    te_out.to_csv(os.path.join(out_dir, "s2_filled_test_L.csv"), index=False)
    mdf = pd.DataFrame(models_rows)
    mdf.to_csv(os.path.join(out_dir, "s2_fill_models_L.csv"), index=False)
    with open(os.path.join(out_dir, "s2_fill_report_L.txt"), "w") as f:
        f.write("\n".join(rep + [""] + detail) + "\n\n" + mdf.to_string(index=False) + "\n")
    info = {"mode": mode, "track": "L (leak-free, primary)", "timestamp": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv), "seed": args.seed, "rule": f"r > {args.rule_min_r}", "grid": GRID,
            "order": order, "test_rows_used_for_fitting": 0, "test_rows_used": test_rows_used,
            "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__, "platform": platform.platform()}}
    with open(os.path.join(out_dir, "s2_fill_run_info_L.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n=== S2 Track L fill | mode={mode} | out={out_dir} ===")
    print("\n".join(rep + [""] + detail))
    print("\n" + mdf[["column", "predictors", "n_fit_rows", "best_params", "cv_r2_vs_mean_predictor",
                      "better_than_mean_predictor", "n_filled_train", "n_filled_test"]].to_string(index=False))
    print("\nobserved vs filled values (mean/sd):")
    print(mdf[["column", "observed_mean", "observed_sd", "filled_train_mean", "filled_train_sd", "filled_test_mean"]].to_string(index=False))
    if not mdf["better_than_mean_predictor"].all():
        print("\nNOTE: for at least one column the tuned model is NOT better than predicting the column mean (CV). "
              "Nothing was changed (no fallback); this is reported as a finding.")
    print("\nNaN remaining after filling: train-side 0, test-side 0. Target was never given to a fill model.")
    if args.smoke:
        print("SMOKE: not reportable. Test rows used: 0.")


if __name__ == "__main__":
    main()
