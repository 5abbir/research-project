#!/usr/bin/env python
"""
r2_s1_autogluon.py  --  REPRO Scenario 1 (S1): AutoGluon on the RAW CSV.

LABEL for every result: "reproduction results, not validated performance".
Result label: "Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2".

Protocol (see S1_PROTOCOL.md, P9/P10/P11):
  * Data = CSV exactly as given; this script does NO filling/scaling/resampling/selection.
    AutoGluon's own default preprocessing is untouched; results reflect AutoGluon's internal NaN handling,
    not a common imputation method.
  * Same official split as the other S1 scripts: stratified 80/20, random_state=42.
  * Deliberate PROJECT settings carried over from v2 (NOT AutoGluon defaults; do not change after seeing results):
      presets="medium_quality", num_bag_folds=5, num_stack_levels=0, time_limit=300, num_cpus=8.
  * AutoGluon trains on ALL train rows (no tuning_data); validation score of a bagged model = out-of-fold score.
  * Every model in the final leaderboard is scored ONCE on the held-out test rows. No model is selected, tuned or
    rejected based on test performance. "Best non-ensemble" and "best overall" flags use the validation
    (out-of-fold) score only.

Modes:
  --smoke : TRAIN rows only (test rows removed right after the split, never used); time_limit 60 s.
            Output -> <out>/autogluon_smoke/. Not reportable.
  default : full run on the official split.
"""
import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_s1_sklearn_models import score_all, sha256_file, split_indices  # noqa: E402

RESULT_LABEL = "Raw CSV / NaN-native (tool-internal handling), project settings carried over from v2"
PROTOCOL_SETTINGS = dict(presets="medium_quality", num_bag_folds=5, num_stack_levels=0, time_limit=300, num_cpus=8)


def one_line(s, n=None):
    s = str(s).replace("\n", " | ")
    return s if n is None else s[:n]


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001
            pass


def compare_dirs(new_dir, old_dir, tol=1e-9):
    out = []
    for fname, keys in (("s1_autogluon_metrics.csv", ["model"]), ("s1_autogluon_predictions.csv", ["model", "row_id"])):
        a_p, b_p = os.path.join(new_dir, fname), os.path.join(old_dir, fname)
        if not (os.path.exists(a_p) and os.path.exists(b_p)):
            out.append(f"{fname}: cannot compare (file missing)")
            continue
        a, b = pd.read_csv(a_p), pd.read_csv(b_p)
        m = a.merge(b, on=keys, suffixes=("_new", "_old"), how="outer", indicator=True)
        if (m["_merge"] != "both").any():
            out.append(f"{fname}: DIFFERENT model/row sets between runs (AutoGluon may train a different model set)")
            continue
        exact, worst = True, 0.0
        for col in a.columns:
            if col in keys or col in ("fit_time", "pred_time_val"):  # timing columns are expected to differ
                continue
            x, y = m[f"{col}_new"], m[f"{col}_old"]
            if pd.api.types.is_numeric_dtype(x):
                if not ((x == y) | (x.isna() & y.isna())).all():
                    exact = False
                    d = (x - y).abs().max()
                    worst = max(worst, float(d) if pd.notna(d) else float("inf"))
            elif not (x.fillna("") == y.fillna("")).all():
                exact, worst = False, float("inf")
        out.append(f"{fname}: EXACT match (timing columns ignored)" if exact else
                   (f"{fname}: APPROXIMATE match (max abs diff {worst:.3g})" if worst <= tol
                    else f"{fname}: DIFFERENT (max abs diff {worst:.3g}); AutoGluon may have nondeterministic parts"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=500)
    ap.add_argument("--time-limit", type=int, default=None, help="default 300 (full), 60 (smoke)")
    ap.add_argument("--num-cpus", type=int, default=PROTOCOL_SETTINGS["num_cpus"])
    ap.add_argument("--eval-metric", default="roc_auc",
                    help="AutoGluon eval_metric [P: v2 value not recorded in the handoff; roc_auc = project primary metric]")
    ap.add_argument("--keep-models", action="store_true", help="keep AutoGluon's model folder (large)")
    ap.add_argument("--compare-with", default=None, help="folder of a previous run of this script")
    ap.add_argument("--check-split-with", default=None, help="results/s1/s1_predictions.csv from the sklearn run")
    args = ap.parse_args()
    time_limit = args.time_limit or (60 if args.smoke else PROTOCOL_SETTINGS["time_limit"])

    try:
        import autogluon.tabular as agt
        from autogluon.tabular import TabularPredictor
    except ImportError as e:
        sys.exit(f"STOP: autogluon.tabular cannot be imported ({e}). Activate env_main (prompt must show (env_main)). "
                 "Nothing was run.")

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        sys.exit(f"STOP: target '{args.target}' not found. Columns: {list(df.columns)}")
    y_all = df[args.target].to_numpy()
    if set(np.unique(y_all)) != {0, 1}:
        sys.exit("STOP: target must contain exactly the labels 0 and 1")
    train_idx, test_idx = split_indices(y_all, args.seed)
    test_set = set(test_idx.tolist())

    mode = "smoke" if args.smoke else "full"
    out_dir = os.path.join(args.out, "autogluon_smoke") if args.smoke else args.out
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke:
        df = df.iloc[train_idx]  # test rows leave memory here and are never used
        sub_idx, _ = train_test_split(train_idx, train_size=args.smoke_n, random_state=args.seed,
                                      stratify=y_all[train_idx])
        fit_idx, eval_idx = train_test_split(sub_idx, test_size=0.2, random_state=args.seed, stratify=y_all[sub_idx])
        fit_idx, eval_idx = np.sort(fit_idx), np.sort(eval_idx)
        assert not ((set(fit_idx.tolist()) | set(eval_idx.tolist())) & test_set), "SMOKE GUARD FAILED"
        assert set(df.index.tolist()).isdisjoint(test_set), "SMOKE GUARD FAILED (test rows in memory)"
        test_rows_used = 0
    else:
        fit_idx, eval_idx = train_idx, test_idx
        test_rows_used = len(test_idx)

    feats = [c for c in df.columns if c != args.target]
    train_df = df.loc[fit_idx].reset_index(drop=True)
    X_eval = df.loc[eval_idx, feats].reset_index(drop=True)
    y_eval = df.loc[eval_idx, args.target].to_numpy()
    nan_rows_fit = int(train_df[feats].isna().any(axis=1).sum())
    assert nan_rows_fit > 0, "No NaN in the training data: the data was altered or has no blanks"

    fit_kw = dict(presets=PROTOCOL_SETTINGS["presets"], num_bag_folds=PROTOCOL_SETTINGS["num_bag_folds"],
                  num_stack_levels=PROTOCOL_SETTINGS["num_stack_levels"], time_limit=time_limit,
                  num_cpus=args.num_cpus)
    deviations = []
    if not args.smoke:
        if time_limit != PROTOCOL_SETTINGS["time_limit"]:
            deviations.append(f"time_limit {time_limit} != protocol {PROTOCOL_SETTINGS['time_limit']}")
        if args.num_cpus != PROTOCOL_SETTINGS["num_cpus"]:
            deviations.append(f"num_cpus {args.num_cpus} != protocol {PROTOCOL_SETTINGS['num_cpus']}")

    rep = [f"mode: {mode}" + ("  (smoke results are NOT reportable)" if args.smoke else ""),
           f"csv: {args.csv}  sha256: {sha256_file(args.csv)}",
           f"seed {args.seed}; stratified 80/20; official train {len(train_idx)}, official test {len(test_idx)}",
           f"fit rows {len(train_df)}; eval rows {len(X_eval)}; fit rows with >=1 NaN: {nan_rows_fit}",
           ("TEST ROWS USED: 0 (removed from memory after the split; guards asserted)" if args.smoke else
            f"eval (test) rows with >=1 NaN: {int(X_eval.isna().any(axis=1).sum())}; "
            "TEST ROWS USED: scored once per leaderboard model"),
           f"AutoGluon fit settings: {fit_kw}; eval_metric={args.eval_metric} [P, see --eval-metric]",
           "preprocessing by this script: NONE (AutoGluon default internal preprocessing untouched)"]
    if deviations:
        rep.append("DEVIATION FROM PROTOCOL SETTINGS: " + "; ".join(deviations))
    if (not args.smoke) and args.check_split_with:
        if os.path.exists(args.check_split_with):
            same = set(pd.read_csv(args.check_split_with)["row_id"].tolist()) == set(eval_idx.tolist())
            rep.append(f"split check vs sklearn run: {'IDENTICAL test rows' if same else 'MISMATCH'}")
            if not same:
                print("\n".join(rep))
                sys.exit("STOP: test rows differ from the sklearn run. Ask the user.")
        else:
            rep.append(f"split check skipped: {args.check_split_with} not found")
    with open(os.path.join(out_dir, "s1_autogluon_split_report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")

    # ---------------- fit (log lines are captured so model failures inside AutoGluon are not hidden)
    handler = ListHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    for name in ("autogluon", ""):
        logging.getLogger(name).addHandler(handler)
    model_dir = os.path.join(out_dir, "ag_models")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    t0 = time.time()
    try:
        predictor = TabularPredictor(label=args.target, eval_metric=args.eval_metric, path=model_dir,
                                     problem_type="binary", verbosity=2)
        predictor.fit(train_data=train_df, **fit_kw)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        with open(os.path.join(out_dir, "s1_autogluon_fit_traceback.txt"), "w") as f:
            f.write(tb)
        with open(os.path.join(out_dir, "s1_autogluon_fit_log.txt"), "w") as f:
            f.write("\n".join(handler.lines))
        print("\n".join(rep))
        sys.exit("STOP AND ASK: AutoGluon fit failed on the raw data. Traceback saved to s1_autogluon_fit_traceback.txt. "
                 "Do not change the protocol without the user.")
    fit_seconds = round(time.time() - t0, 1)
    for name in ("autogluon", ""):
        logging.getLogger(name).removeHandler(handler)
    handler.lines = list(dict.fromkeys(handler.lines))  # handler sits on two loggers; drop duplicate lines
    with open(os.path.join(out_dir, "s1_autogluon_fit_log.txt"), "w") as f:
        f.write("\n".join(handler.lines))
    # warnings/errors/exceptions only (the settings dump contains "squared_error" and is NOT a failure)
    fail_lines = [ln for ln in handler.lines
                  if ln.startswith(("WARNING", "ERROR", "CRITICAL")) or "Exception" in ln or "failed" in ln.lower()]

    # ---------------- validation-only leaderboard (no test data passed)
    lb = predictor.leaderboard(silent=True).copy()
    lb["is_ensemble"] = lb["model"].str.startswith("WeightedEnsemble")
    ok_val = lb["score_val"].fillna(-np.inf)
    lb["best_overall_by_val"] = False
    lb.loc[ok_val.idxmax(), "best_overall_by_val"] = True
    lb["best_nonensemble_by_val"] = False
    ne = lb[~lb["is_ensemble"]]
    if not ne.empty:
        lb.loc[ne["score_val"].fillna(-np.inf).idxmax(), "best_nonensemble_by_val"] = True
    lb["best_flag_basis"] = "validation (out-of-fold) score on train rows only; NOT test"
    lb["result_label"] = RESULT_LABEL
    lb.to_csv(os.path.join(out_dir, "s1_autogluon_leaderboard.csv"), index=False)

    # ---------------- NaN handling notes (documentation)
    notes = ["AutoGluon NaN handling, Scenario 1 (raw CSV)",
             f"mode: {mode}; autogluon {agt.__version__}",
             "Known from the v2 experiment [C from v2; NOT re-verified in this run]: RandomForest/ExtraTrees-type models fill NaN "
             "with 0, neural networks use the median, gradient-boosting models handle NaN natively.",
             "Results below reflect AutoGluon's internal handling, not one manually specified common imputation, and are "
             "not equivalent to the sklearn/PyCaret NaN-native runs."]
    try:
        tf = predictor.transform_features(X_eval.head(50))
        notes.append(f"NaN cells in first 50 eval rows before AutoGluon's generic feature transform: "
                     f"{int(X_eval.head(50).isna().sum().sum())}; after: {int(tf.isna().sum().sum())} "
                     "(generic transform only; model-specific filling happens inside each model)")
    except Exception as e:  # noqa: BLE001
        notes.append(f"could not inspect transform_features: {one_line(e, 150)}")
    notes.append(f"log lines mentioning failures/errors during fit: {len(fail_lines)} (full log: s1_autogluon_fit_log.txt)")
    notes += ["  " + one_line(ln, 220) for ln in fail_lines[:15]]
    with open(os.path.join(out_dir, "s1_autogluon_nan_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")

    # ---------------- score every leaderboard model ONCE on the eval rows
    met_rows, conf_rows, pred_frames, status, tbs = [], [], [], [], {}
    for mname in lb["model"].tolist():
        try:
            proba = predictor.predict_proba(X_eval, model=mname)
            p1 = (proba[1] if 1 in proba.columns else proba.iloc[:, -1]).astype(float).to_numpy()
            y_pred = (p1 > 0.5).astype(int)  # threshold 0.5 on class-1 probability
            sc, (tn, fp, fn, tp) = score_all(y_eval, y_pred, p1)
            row = lb[lb["model"] == mname].iloc[0]
            met_rows.append({"model": mname, "result_label": RESULT_LABEL, "run_mode": mode,
                             "is_ensemble": bool(row["is_ensemble"]), "score_val": row["score_val"],
                             "best_overall_by_val": bool(row["best_overall_by_val"]),
                             "best_nonensemble_by_val": bool(row["best_nonensemble_by_val"]),
                             "fit_time": row.get("fit_time", np.nan), "pred_time_val": row.get("pred_time_val", np.nan),
                             **sc})
            conf_rows.append({"model": mname, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "n_eval": len(y_eval)})
            pred_frames.append(pd.DataFrame({"model": mname, "row_id": eval_idx, "y_true": y_eval, "y_pred": y_pred,
                                             "proba_class1": p1}))
            status.append((mname, "scored", ""))
        except Exception as e:  # noqa: BLE001
            tbs[mname] = traceback.format_exc()
            status.append((mname, "scoring failed", one_line(e, 200)))
    pd.DataFrame(met_rows).to_csv(os.path.join(out_dir, "s1_autogluon_metrics.csv"), index=False)
    pd.DataFrame(conf_rows).to_csv(os.path.join(out_dir, "s1_autogluon_confusion.csv"), index=False)
    (pd.concat(pred_frames) if pred_frames else
     pd.DataFrame(columns=["model", "row_id", "y_true", "y_pred", "proba_class1"])).to_csv(
        os.path.join(out_dir, "s1_autogluon_predictions.csv"), index=False)
    for mname, tb in tbs.items():
        with open(os.path.join(out_dir, f"s1_autogluon_traceback_{mname}.txt"), "w") as f:
            f.write(tb)

    info = {"mode": mode, "timestamp": datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv),
            "seed": args.seed, "fit_kwargs": {k: str(v) for k, v in fit_kw.items()}, "eval_metric": args.eval_metric,
            "eval_metric_note": "[P] v2 value not recorded in the handoff; user to confirm", "fit_seconds": fit_seconds,
            "protocol_deviations": deviations, "test_rows_used": test_rows_used,
            "versions": {"python": sys.version.split()[0], "autogluon.tabular": agt.__version__,
                         "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                         "platform": platform.platform()},
            "n_models_in_leaderboard": int(len(lb)),
            "determinism_note": "AutoGluon may have nondeterministic parts (threads, time limit); repeat equality is checked, not assumed"}
    with open(os.path.join(out_dir, "s1_autogluon_run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ---------------- console summary (sorted by validation score, NOT test score)
    print(f"\n=== S1 AutoGluon | mode={mode} | out={out_dir} ===")
    print("\n".join(rep))
    print(f"\nfit seconds: {fit_seconds}; models in leaderboard: {len(lb)}; scored on eval rows: "
          f"{sum(1 for s in status if s[1] == 'scored')}; scoring failures: {sum(1 for s in status if s[1] != 'scored')}")
    print(f"log lines mentioning failures/errors during fit: {len(fail_lines)} (see s1_autogluon_nan_notes.txt)")
    if met_rows:
        m = pd.DataFrame(met_rows).sort_values("score_val", ascending=False)
        cols = ["model", "is_ensemble", "score_val", "accuracy", "baseline_acc_majority", "roc_auc", "balanced_accuracy",
                "f1", "mcc", "best_overall_by_val", "best_nonensemble_by_val"]
        print("\nModels (sorted by VALIDATION score; test scores shown once, not used for selection):")
        print(m[cols].round(4).to_string(index=False))
    if args.smoke:
        print("\nSMOKE: not reportable. Test rows used: 0.")
    if args.compare_with:
        print("\nRun comparison vs", args.compare_with)
        for line in compare_dirs(out_dir, args.compare_with if not args.smoke else os.path.join(args.compare_with, "autogluon_smoke")):
            print("  ", line)
    if not args.keep_models:
        shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
