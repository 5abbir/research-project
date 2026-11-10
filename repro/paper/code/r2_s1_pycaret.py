#!/usr/bin/env python
"""
r2_s1_pycaret.py  --  REPRO Scenario 1 (S1): every PyCaret library model on the RAW CSV.

LABEL for every result: "reproduction results, not validated performance".

Protocol (see S1_PROTOCOL.md, P8/P10/P11):
  * Data = CSV exactly as given. PyCaret setup(imputation_type=None, normalize=False,
    fix_imbalance=False, feature_selection=False): no filling, scaling, resampling or selection.
  * Same official split as the sklearn script: stratified 80/20, random_state=42.
    setup(data=train_rows, test_data=test_rows)  -> PyCaret uses ALL train rows (avoids the 70% default).
  * Every model in PyCaret's library is tried ONE BY ONE with create_model (5-fold stratified CV on train rows).
    A failure is recorded with the exact error + FULL traceback, then the loop continues. Nothing is skipped silently.
  * Every runnable model is scored ONCE on the held-out test rows. No model is selected, tuned or rejected
    based on test performance. "Best" flags use CV (train-only) AUC and are marked validation-based.
  * Roles: Primary = xgboost, rf, ada (paper's three). Exploratory = every other runnable model
    (NOT pre-specified).

Modes:
  --smoke : TRAIN rows only (test rows removed right after the split, never used). Output -> <out>/pycaret_smoke/.
            Not reportable. Use --only rf,ada,lr first to probe quickly.
  default : full run on the official split; test rows scored once per runnable model.
"""
import argparse
import json
import os
import platform
import signal
import sys
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_s1_sklearn_models import score_all, sha256_file, split_indices  # noqa: E402

LABEL_OK = "Raw CSV / NaN-native"
LABEL_FAIL = "Not runnable"
PRIMARY = {"xgboost": "XGBoost", "rf": "RandomForest", "ada": "AdaBoost"}
CV_COLS = ["Accuracy", "AUC", "Recall", "Prec.", "F1", "Kappa", "MCC"]


class ModelTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ModelTimeout("per-model time limit reached")


def one_line(s, n=None):
    s = str(s).replace("\n", " | ")
    return s if n is None else s[:n]


def compare_dirs(new_dir, old_dir, tol=1e-9):
    out = []
    for fname, keys in (("s1_pycaret_test_metrics.csv", ["model_id"]),
                        ("s1_pycaret_predictions.csv", ["model_id", "row_id"])):
        a_p, b_p = os.path.join(new_dir, fname), os.path.join(old_dir, fname)
        if not (os.path.exists(a_p) and os.path.exists(b_p)):
            out.append(f"{fname}: cannot compare (file missing)")
            continue
        a, b = pd.read_csv(a_p), pd.read_csv(b_p)
        m = a.merge(b, on=keys, suffixes=("_new", "_old"), how="outer", indicator=True)
        if (m["_merge"] != "both").any():
            out.append(f"{fname}: DIFFERENT row sets between runs")
            continue
        exact, worst = True, 0.0
        for col in a.columns:
            if col in keys:
                continue
            x, y = m[f"{col}_new"], m[f"{col}_old"]
            if pd.api.types.is_numeric_dtype(x):
                if not ((x == y) | (x.isna() & y.isna())).all():
                    exact = False
                    d = (x - y).abs().max()
                    worst = max(worst, float(d) if pd.notna(d) else float("inf"))
            elif not (x.fillna("") == y.fillna("")).all():
                exact, worst = False, float("inf")
        out.append(f"{fname}: EXACT match" if exact else
                   (f"{fname}: APPROXIMATE match (max abs diff {worst:.3g})" if worst <= tol
                    else f"{fname}: DIFFERENT (max abs diff {worst:.3g})"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=500)
    ap.add_argument("--only", default=None, help="comma list of PyCaret model ids (probe), e.g. rf,ada,lr")
    ap.add_argument("--per-model-timeout", type=int, default=None,
                    help="seconds per model (default 300 full, 60 smoke). A timeout is recorded as a failure.")
    ap.add_argument("--compare-with", default=None, help="folder of a previous run of this script (exact/approx check)")
    ap.add_argument("--check-split-with", default=None,
                    help="path to results/s1/s1_predictions.csv from the sklearn run; test row ids must match")
    args = ap.parse_args()
    timeout = args.per_model_timeout or (60 if args.smoke else 300)

    try:
        import pycaret
        from pycaret.classification import ClassificationExperiment
    except ImportError as e:
        sys.exit(f"STOP: pycaret cannot be imported ({e}). Activate env_pycaret (prompt must show (env_pycaret)). "
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
    out_dir = os.path.join(args.out, "pycaret_smoke") if args.smoke else args.out
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

    train_df = df.loc[fit_idx].reset_index(drop=True)
    test_df = df.loc[eval_idx].reset_index(drop=True)
    feats = [c for c in df.columns if c != args.target]
    y_eval = test_df[args.target].to_numpy()
    nan_rows_fit = int(train_df[feats].isna().any(axis=1).sum())
    assert nan_rows_fit > 0, "No NaN in the training data: the data was altered or has no blanks"

    rep = [f"mode: {mode}" + ("  (smoke results are NOT reportable)" if args.smoke else ""),
           f"csv: {args.csv}  sha256: {sha256_file(args.csv)}",
           f"seed {args.seed}; stratified 80/20; official train {len(train_idx)}, official test {len(test_idx)}",
           f"fit rows {len(train_df)}; eval rows {len(test_df)}",
           f"fit rows with >=1 NaN: {nan_rows_fit}",
           ("TEST ROWS USED: 0 (removed from memory after the split; guards asserted)" if args.smoke else
            f"eval (test) rows with >=1 NaN: {int(test_df[feats].isna().any(axis=1).sum())}; "
            "TEST ROWS USED: scored once per runnable model"),
           "preprocessing requested: NONE (imputation_type=None, normalize=False, fix_imbalance=False, "
           "feature_selection=False)",
           f"per-model timeout: {timeout}s"]

    if (not args.smoke) and args.check_split_with:
        if os.path.exists(args.check_split_with):
            ref = set(pd.read_csv(args.check_split_with)["row_id"].tolist())
            same = ref == set(eval_idx.tolist())
            rep.append(f"split check vs sklearn run ({args.check_split_with}): {'IDENTICAL test rows' if same else 'MISMATCH'}")
            if not same:
                print("\n".join(rep))
                sys.exit("STOP: test rows differ from the sklearn run (library-version split difference?). Ask the user.")
        else:
            rep.append(f"split check skipped: {args.check_split_with} not found")
    with open(os.path.join(out_dir, "s1_pycaret_split_report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")

    # ---------------- PyCaret setup (one call; failure = stop and ask)
    setup_kw = dict(target=args.target, session_id=args.seed, fold=5, fold_strategy="stratifiedkfold",
                    imputation_type=None, normalize=False, fix_imbalance=False, feature_selection=False,
                    n_jobs=args.n_jobs, verbose=False, html=False, system_log=False,
                    index=False)  # train/test both have a 0..n-1 index; PyCaret needs index=False (bug fix, not a protocol change)
    exp = ClassificationExperiment()
    try:
        exp.setup(data=train_df, test_data=test_df, **setup_kw)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        with open(os.path.join(out_dir, "s1_pycaret_setup_traceback.txt"), "w") as f:
            f.write(tb)
        print("\n".join(rep))
        sys.exit("STOP AND ASK: PyCaret setup failed with imputation_type=None on NaN data. "
                 "Traceback saved to s1_pycaret_setup_traceback.txt. Do not change the protocol without the user.")

    checks = []
    try:
        n_tr = int(exp.get_config("X_train").shape[0])
        checks.append(f"PyCaret X_train rows = {n_tr} (expected {len(train_df)}): {'OK' if n_tr == len(train_df) else 'MISMATCH'}")
        if n_tr != len(train_df):
            print("\n".join(rep + checks))
            sys.exit("STOP: PyCaret did not use all supplied train rows (the 70% default bug?). Ask the user.")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        checks.append(f"could not read X_train size: {one_line(e, 120)}")
    try:
        nan_after = int(exp.get_config("X_train_transformed").isna().sum().sum())
        checks.append(f"NaN cells still present after PyCaret preprocessing (train): {nan_after}"
                      + ("  -> no imputation happened" if nan_after > 0 else "  -> WARNING: NaN gone, something imputed?"))
    except Exception as e:  # noqa: BLE001
        checks.append(f"could not read transformed X_train: {one_line(e, 120)}")
    rep += checks
    with open(os.path.join(out_dir, "s1_pycaret_split_report.txt"), "a") as f:
        f.write("\n".join(checks) + "\n")

    lib = exp.models()
    ids = list(lib.index)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [w for w in wanted if w not in ids]
        if unknown:
            sys.exit(f"STOP: unknown model ids {unknown}. Library ids: {ids}")
        ids = wanted
    names = {i: str(lib.loc[i].iloc[0]) for i in ids}

    X_test = test_df.drop(columns=[args.target])
    signal.signal(signal.SIGALRM, _alarm_handler)

    status, met_rows, conf_rows, pred_frames, tracebacks = [], [], [], [], {}
    for mid in ids:
        role = "Primary" if mid in PRIMARY else "Exploratory"
        rec = {"model_id": mid, "model_name": names[mid], "role": role, "status": "", "result_label": "",
               "error_type": "", "error_message": "", "warnings_snippet": "", "cv_suspect": "", "seconds": ""}
        t0 = time.time()
        stage = "create_model"
        model = None
        try:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                signal.alarm(timeout)
                try:
                    model = exp.create_model(mid, verbose=False)
                    grid = exp.pull()
                    folds = grid[~grid.index.isin(["Mean", "Std"])]
                    cv_mean = {c: float(pd.to_numeric(folds[c], errors="coerce").mean()) for c in CV_COLS if c in folds}
                    stage = "predict_model"
                    pred = exp.predict_model(model, data=X_test, raw_score=True, verbose=False)
                finally:
                    signal.alarm(0)
            rec["warnings_snippet"] = one_line("; ".join(sorted({str(w.message)[:80] for w in wlist})), 300)
            assert len(pred) == len(test_df), "prediction row count differs from test rows"
            y_pred = pred["prediction_label"].astype(int).to_numpy()
            pcol = [c for c in pred.columns if c == "prediction_score_1"]
            have_p = bool(pcol)
            p1 = pred[pcol[0]].astype(float).to_numpy() if have_p else y_pred.astype(float)
            sc, (tn, fp, fn, tp) = score_all(y_eval, y_pred, p1)
            if not have_p:
                sc["roc_auc"] = sc["log_loss"] = sc["brier"] = float("nan")
                sc["note"] = (sc["note"] + "; " if sc["note"] else "") + "no predicted probabilities from this model"
            auc_cv = cv_mean.get("AUC", float("nan"))
            if not (auc_cv > 0):
                rec["cv_suspect"] = "CV AUC is 0/NaN: possible swallowed error inside PyCaret CV; check warnings"
            rec.update(status="runnable", result_label=LABEL_OK)
            met_rows.append({"model_id": mid, "model_name": names[mid], "role": role, "result_label": LABEL_OK,
                             "run_mode": mode, "proba_available": have_p,
                             **{f"cv_{k.lower().replace('.', '')}": v for k, v in cv_mean.items()}, **sc})
            conf_rows.append({"model_id": mid, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "n_eval": len(y_eval)})
            pred_frames.append(pd.DataFrame({"model_id": mid, "row_id": eval_idx, "y_true": y_eval, "y_pred": y_pred,
                                             "proba_class1": p1 if have_p else np.nan}))
        except ModelTimeout:
            tracebacks[mid] = traceback.format_exc()
            rec.update(status="failed: timeout", result_label=LABEL_FAIL, error_type="ModelTimeout",
                       error_message=f"stopped after {timeout}s at stage {stage} (not an NaN error)")
        except Exception as e:  # noqa: BLE001  (record everything, hide nothing)
            tracebacks[mid] = traceback.format_exc()
            rec.update(status="not runnable" if stage == "create_model" else "trained, scoring failed",
                       result_label=LABEL_FAIL, error_type=type(e).__name__, error_message=one_line(e))
        rec["seconds"] = round(time.time() - t0, 2)
        status.append(rec)
        print(f"[{mid:>10}] {rec['role']:<11} {rec['status']:<24} {rec['seconds']:>7}s  {one_line(rec['error_message'], 70)}", flush=True)

    # ---------------- validation-based "best" flags (CV AUC only; never test)
    met = pd.DataFrame(met_rows)
    if not met.empty and "cv_auc" in met:
        ok = met["cv_auc"].fillna(-1)
        met["cv_best_overall"] = False
        met.loc[ok.idxmax(), "cv_best_overall"] = True
        prim = met[met["role"] == "Primary"]
        met["cv_best_primary"] = False
        if not prim.empty:
            met.loc[prim["cv_auc"].fillna(-1).idxmax(), "cv_best_primary"] = True
        met["best_flag_basis"] = "validation (CV AUC on train rows only); NOT test"
    met.to_csv(os.path.join(out_dir, "s1_pycaret_test_metrics.csv"), index=False)
    st = pd.DataFrame(status)
    if not met.empty:
        st = st.merge(met[["model_id", "cv_auc"]], on="model_id", how="left")
    st.to_csv(os.path.join(out_dir, "s1_pycaret_model_status.csv"), index=False)
    pd.DataFrame(conf_rows).to_csv(os.path.join(out_dir, "s1_pycaret_confusion.csv"), index=False)
    (pd.concat(pred_frames) if pred_frames else
     pd.DataFrame(columns=["model_id", "row_id", "y_true", "y_pred", "proba_class1"])).to_csv(
        os.path.join(out_dir, "s1_pycaret_predictions.csv"), index=False)
    for mid, tb in tracebacks.items():
        with open(os.path.join(out_dir, f"s1_pycaret_traceback_{mid}.txt"), "w") as f:
            f.write(tb)

    info = {"mode": mode, "timestamp": datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv),
            "seed": args.seed, "n_jobs": args.n_jobs, "per_model_timeout_s": timeout, "test_rows_used": test_rows_used,
            "setup_kwargs": {k: str(v) for k, v in setup_kw.items()},
            "versions": {"python": sys.version.split()[0], "pycaret": pycaret.__version__, "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
            "library_ids": list(lib.index), "ids_run": ids,
            "determinism_note": "seeds fixed; repeat equality is checked, not assumed (use --compare-with)"}
    with open(os.path.join(out_dir, "s1_pycaret_run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ---------------- console summary (sorted by role then id: NOT by test score)
    print(f"\n=== S1 PyCaret | mode={mode} | out={out_dir} ===")
    print("\n".join(rep))
    n_ok = int((st["status"] == "runnable").sum())
    print(f"\nRunnable: {n_ok} of {len(st)}.  Not runnable / failed: {len(st) - n_ok}")
    if not met.empty:
        cols = ["model_id", "role", "n_eval", "accuracy", "baseline_acc_majority", "roc_auc", "balanced_accuracy",
                "f1", "mcc", "cv_auc", "cv_best_overall", "cv_best_primary"]
        print("\nTest scores (sorted by role, id; best flags are CV-based):")
        print(met.sort_values(["role", "model_id"])[[c for c in cols if c in met]].round(4).to_string(index=False))
    if (st["cv_suspect"] != "").any():
        print("\nSUSPECT rows:\n", st.loc[st["cv_suspect"] != "", ["model_id", "cv_suspect"]].to_string(index=False))
    if args.smoke:
        print("\nSMOKE: not reportable. Test rows used: 0.")
    if args.compare_with:
        print("\nRun comparison vs", args.compare_with)
        for line in compare_dirs(out_dir, args.compare_with if not args.smoke else os.path.join(args.compare_with, "pycaret_smoke")):
            print("  ", line)


if __name__ == "__main__":
    main()
