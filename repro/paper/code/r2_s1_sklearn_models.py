#!/usr/bin/env python
"""
r2_s1_sklearn_models.py  --  REPRO Scenario 1 (S1): XGBoost, RandomForest, AdaBoost on the RAW CSV.

LABEL for every result: "reproduction results, not validated performance".

Protocol (see S1_PROTOCOL.md):
  * Data = the CSV exactly as given. No filling, no dropping of rows, no scaling,
    no resampling, no feature selection. Blank cells stay blank (NaN) and go straight to the models.
  * Split: stratified 80/20, random_state=42 (indices computed from the labels only).
  * Hyper-parameters: library defaults; only random_state (seed) and n_jobs are set.
  * Every model that can run is scored ONCE on the held-out test set. No model is selected,
    tuned, or rejected based on test performance.
  * A model that cannot run on NaN is reported as "Not runnable" with the exact error
    and the full traceback. It is never rescued with any filling.

Modes:
  --smoke : uses TRAIN rows only (test rows are removed right after the split and never used).
            Output goes to <out>/smoke/ and is NOT reportable.
  (default): full run on the official split; test scored once.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

LABEL_OK = "Raw CSV / NaN-native"
LABEL_FAIL = "Not runnable"
ROLE = "Primary"  # the paper reports these three models in Scenario 1
SEED_DEFAULT = 42

# Paper Table 5 (Scenario 1 / original data). REFERENCE ONLY, never mixed with our numbers.
PAPER_TABLE5 = [
    ("XGBoost",      63.70, 32.37, 89.07, 42.93, 67.98, ""),
    ("RandomForest", 63.97, 35.65, 88.10, 45.78, 68.60, "text Sect. 5.4 says 68.50; Table 5 / conclusion say 68.60"),
    ("AdaBoost",     48.97, 19.67, 87.86, 28.07, 62.50, ""),
]


# ----------------------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_indices(y, seed):
    """Official 80/20 stratified split. Uses ONLY the labels and the row count."""
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    return np.sort(train_idx), np.sort(test_idx)


def build_models(seed, n_jobs, XGBClassifier):
    # Library defaults. Only seed and n_jobs are set (both logged).
    return {
        "XGBoost": XGBClassifier(random_state=seed, n_jobs=n_jobs),
        "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=n_jobs),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
    }


def proba_pos(model, X):
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def score_all(y_true, y_pred, p1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    maj = int(pd.Series(y_true).value_counts().idxmax())
    note = ""
    if tp + fp == 0:
        note = "no positive predictions: precision set to 0 (zero_division=0)"
    return {
        "n_eval": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "baseline_acc_majority": float(np.mean(np.asarray(y_true) == maj)),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, p1),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, np.column_stack([1 - p1, p1]), labels=[0, 1]),
        "brier": brier_score_loss(y_true, p1),
        "note": note,
    }, (int(tn), int(fp), int(fn), int(tp))


def compare_runs(new_dir, old_metrics_path, tol=1e-9):
    """Report whether two runs match EXACTLY or only approximately (no bit-for-bit claim)."""
    old_dir = os.path.dirname(os.path.abspath(old_metrics_path))
    out = []
    for fname, keys in (("s1_sklearn_metrics.csv", ["model"]),
                        ("s1_predictions.csv", ["model", "row_id"])):
        a_path, b_path = os.path.join(new_dir, fname), os.path.join(old_dir, fname)
        if not (os.path.exists(a_path) and os.path.exists(b_path)):
            out.append(f"{fname}: cannot compare (file missing in one of the runs)")
            continue
        a, b = pd.read_csv(a_path), pd.read_csv(b_path)
        m = a.merge(b, on=keys, suffixes=("_new", "_old"), how="outer", indicator=True)
        if (m["_merge"] != "both").any():
            out.append(f"{fname}: DIFFERENT row sets between runs")
            continue
        worst, exact = 0.0, True
        for col in a.columns:
            if col in keys or col == "note":
                continue
            x, y = m[f"{col}_new"], m[f"{col}_old"]
            if pd.api.types.is_numeric_dtype(x):
                same = ((x == y) | (x.isna() & y.isna())).all()
                if not same:
                    exact = False
                    d = (x - y).abs().max()
                    worst = max(worst, float(d) if pd.notna(d) else float("inf"))
            else:
                if not (x.fillna("") == y.fillna("")).all():
                    exact = False
                    worst = float("inf")
        if exact:
            out.append(f"{fname}: EXACT match")
        elif worst <= tol:
            out.append(f"{fname}: APPROXIMATE match (max abs diff {worst:.3g} <= {tol})")
        else:
            out.append(f"{fname}: DIFFERENT (max abs diff {worst:.3g})")
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True, help="output folder (smoke mode writes to <out>/smoke)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="train-rows-only smoke test; not reportable")
    ap.add_argument("--smoke-n", type=int, default=500, help="rows in the smoke subsample (from TRAIN only)")
    ap.add_argument("--compare-with", default=None,
                    help="path to a previous s1_sklearn_metrics.csv; reports exact/approximate match")
    args = ap.parse_args()

    try:
        from xgboost import XGBClassifier
        import xgboost
    except ImportError as e:
        sys.exit(f"STOP: xgboost cannot be imported ({e}). Activate env_main (prompt must show (env_main)). "
                 "Nothing was run and no result was written.")

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        sys.exit(f"STOP: target column '{args.target}' not found. Columns: {list(df.columns)}")
    y_all = df[args.target].to_numpy()
    if set(np.unique(y_all)) != {0, 1}:
        sys.exit(f"STOP: target must contain exactly the labels 0 and 1, found {sorted(set(np.unique(y_all)))}")
    n_total = len(df)

    train_idx, test_idx = split_indices(y_all, args.seed)
    test_set = set(test_idx.tolist())

    mode = "smoke" if args.smoke else "full"
    out_dir = os.path.join(args.out, "smoke") if args.smoke else args.out
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke:
        # Test rows are removed from memory right here and are never used, scored or printed.
        df = df.iloc[train_idx]
        if args.smoke_n >= len(train_idx):
            sys.exit("STOP: --smoke-n must be smaller than the number of train rows")
        sub_idx, _ = train_test_split(train_idx, train_size=args.smoke_n, random_state=args.seed,
                                      stratify=y_all[train_idx])
        fit_idx, eval_idx = train_test_split(sub_idx, test_size=0.2, random_state=args.seed,
                                             stratify=y_all[sub_idx])
        fit_idx, eval_idx = np.sort(fit_idx), np.sort(eval_idx)
        used = set(fit_idx.tolist()) | set(eval_idx.tolist())
        assert not (used & test_set), "SMOKE GUARD FAILED: a test row is in the smoke data"
        assert set(df.index.tolist()).isdisjoint(test_set), "SMOKE GUARD FAILED: test rows still in memory"
        test_rows_used = 0
    else:
        fit_idx, eval_idx = train_idx, test_idx
        test_rows_used = len(test_idx)

    feats = [c for c in df.columns if c != args.target]
    X_fit, y_fit = df.loc[fit_idx, feats], df.loc[fit_idx, args.target].to_numpy()
    X_eval, y_eval = df.loc[eval_idx, feats], df.loc[eval_idx, args.target].to_numpy()

    # ---- protocol self-checks: data untouched (NaN must still be present in what the models receive)
    nan_rows_fit = int(X_fit.isna().any(axis=1).sum())
    assert nan_rows_fit > 0, "No NaN reached the models: the data was altered or the file has no blanks"
    assert not (set(fit_idx.tolist()) & set(eval_idx.tolist())), "fit/eval overlap"

    # ---- split report (counts are label/NaN counts only; in smoke mode nothing about test rows is computed)
    rep = []
    rep.append(f"mode: {mode}  (smoke results are NOT reportable)" if args.smoke else f"mode: {mode}")
    rep.append(f"csv: {args.csv}  sha256: {sha256_file(args.csv)}")
    rep.append(f"rows in file: {n_total}; columns: {df.shape[1]} (features {len(feats)} + target)")
    rep.append(f"seed: {args.seed}; split: stratified 80/20; official train rows {len(train_idx)}, official test rows {len(test_idx)}")
    cc = lambda a: {int(k): int(v) for k, v in pd.Series(a).value_counts().sort_index().items()}  # noqa: E731
    rep.append(f"fit rows: {len(fit_idx)}  (class counts {cc(y_fit)})")
    rep.append(f"eval rows: {len(eval_idx)}  (class counts {cc(y_eval)})")
    rep.append(f"fit rows containing >=1 NaN: {nan_rows_fit}; NaN cells per column in fit rows: "
               f"{ {c: int(v) for c, v in X_fit.isna().sum().items() if v} }")
    if args.smoke:
        rep.append("TEST ROWS USED: 0  (removed from memory after the split; guards asserted: "
                   "smoke rows disjoint from test indices)")
        rep.append("NaN counts for the test partition are NOT computed in smoke mode.")
        nan_rows_test = None
    else:
        nan_rows_test = int(X_eval.isna().any(axis=1).sum())
        rep.append(f"eval (test) rows containing >=1 NaN: {nan_rows_test}")
        rep.append(f"NaN-containing rows, whole file: {int(df[feats].isna().any(axis=1).sum())}")
        rep.append("TEST ROWS USED: scored once per runnable model, after fitting on train rows only")
    rep.append("preprocessing applied: NONE (no filling, no row dropping, no scaling, no resampling, no feature selection)")
    with open(os.path.join(out_dir, "s1_split_report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")

    # ---- fit and score
    models = build_models(args.seed, args.n_jobs, XGBClassifier)
    metrics_rows, conf_rows, status_rows, pred_frames = [], [], [], []
    tracebacks = {}
    for name, model in models.items():
        t0 = time.time()
        try:
            model.fit(X_fit, y_fit)
            p1 = proba_pos(model, X_eval)
            y_pred = model.predict(X_eval)
            sc, (tn, fp, fn, tp) = score_all(y_eval, y_pred, p1)
            metrics_rows.append({"model": name, "role": ROLE, "result_label": LABEL_OK, "run_mode": mode, **sc})
            conf_rows.append({"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "n_eval": len(y_eval)})
            pred_frames.append(pd.DataFrame({"model": name, "row_id": eval_idx, "y_true": y_eval,
                                             "y_pred": y_pred, "proba_class1": p1}))
            status_rows.append({"model": name, "role": ROLE, "status": "runnable", "result_label": LABEL_OK,
                                "error_type": "", "error_message": "", "fit_predict_seconds": round(time.time() - t0, 3)})
        except Exception as e:  # noqa: BLE001  (record every failure, never hide it)
            tracebacks[name] = traceback.format_exc()
            status_rows.append({"model": name, "role": ROLE, "status": "not runnable", "result_label": LABEL_FAIL,
                                "error_type": type(e).__name__, "error_message": str(e).replace("\n", " | "),
                                "fit_predict_seconds": round(time.time() - t0, 3)})
            metrics_rows.append({"model": name, "role": ROLE, "result_label": LABEL_FAIL, "run_mode": mode,
                                 "note": "Not runnable on raw CSV with NaN values; missing-value handling unspecified in paper"})

    pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, "s1_sklearn_metrics.csv"), index=False)
    pd.DataFrame(conf_rows).to_csv(os.path.join(out_dir, "s1_confusion.csv"), index=False)
    pd.DataFrame(status_rows).to_csv(os.path.join(out_dir, "s1_model_status.csv"), index=False)
    if pred_frames:
        pd.concat(pred_frames).to_csv(os.path.join(out_dir, "s1_predictions.csv"), index=False)
    else:
        pd.DataFrame(columns=["model", "row_id", "y_true", "y_pred", "proba_class1"]).to_csv(
            os.path.join(out_dir, "s1_predictions.csv"), index=False)
    for name, tb in tracebacks.items():
        with open(os.path.join(out_dir, f"s1_traceback_{name}.txt"), "w") as f:
            f.write(tb)
    pd.DataFrame(PAPER_TABLE5, columns=["model", "precision_pct", "sensitivity_pct", "specificity_pct", "f1_pct",
                                        "accuracy_pct", "note"]).assign(
        source="Shehab et al. 2025, Table 5 (original data)", use="REFERENCE ONLY, not our result").to_csv(
        os.path.join(out_dir, "s1_paper_reference.csv"), index=False)

    # ---- AdaBoost record (requested items)
    st = {r["model"]: r for r in status_rows}
    ada = st["AdaBoost"]
    L = []
    L.append("AdaBoost record, Scenario 1 (raw CSV)")
    L.append(f"mode: {mode}" + ("  (SMOKE: not reportable)" if args.smoke else ""))
    if ada["status"] == "not runnable":
        L.append('Reported as: "Not runnable on raw CSV with NaN values; missing-value handling unspecified in paper."')
        L.append("Our status: NOT REPRODUCIBLE for AdaBoost under the raw-CSV protocol.")
    else:
        L.append("UNEXPECTED: AdaBoost ran on NaN in this installed version. Labelled 'Raw CSV / NaN-native'. "
                 "Report this to the user; do not assume the earlier expectation.")
    L.append(f"exact error type: {ada['error_type']}")
    L.append(f"exact error message: {ada['error_message']}")
    L.append(f"scikit-learn version: {sklearn.__version__}")
    L.append(f"rows with >=1 NaN: fit/train rows used = {nan_rows_fit}; "
             + (f"test rows = {nan_rows_test}; whole file = {int(df[feats].isna().any(axis=1).sum())}"
                if not args.smoke else "test rows = not computed in smoke mode"))
    L.append(f"XGBoost ran on NaN: {st['XGBoost']['status']}; RandomForest ran on NaN: {st['RandomForest']['status']}")
    L.append("Paper's AdaBoost result (Table 5, reference only): precision 48.97, sensitivity 19.67, "
             "specificity 87.86, F1 28.07, accuracy 62.50 (%)")
    if "AdaBoost" in tracebacks:
        L.append("\nFull traceback:\n" + tracebacks["AdaBoost"])
    with open(os.path.join(out_dir, "s1_adaboost_record.txt"), "w") as f:
        f.write("\n".join(L) + "\n")

    # ---- run info (seeds, versions, parameters)
    info = {
        "mode": mode, "timestamp": datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv),
        "seed": args.seed, "n_jobs": args.n_jobs, "test_rows_used": test_rows_used,
        "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "platform": platform.platform()},
        "model_params": {k: {p: str(v) for p, v in m.get_params().items()} for k, m in models.items()},
        "determinism_note": "seeds fixed; exact repeat equality is checked, not assumed (use --compare-with)",
    }
    with open(os.path.join(out_dir, "s1_run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ---- console summary
    print(f"\n=== S1 sklearn models | mode={mode} | out={out_dir} ===")
    print("\n".join(rep))
    print("\nModel status:")
    short = pd.DataFrame(status_rows)[["model", "status", "result_label", "error_type", "error_message"]].copy()
    short["error_message"] = short["error_message"].str.slice(0, 90)  # console only; full text is in the files
    print(short.to_string(index=False))
    print("(full error text: s1_model_status.csv, s1_adaboost_record.txt, s1_traceback_*.txt)")
    ok = [r for r in metrics_rows if r["result_label"] == LABEL_OK]
    if ok:
        cols = ["model", "n_eval", "accuracy", "baseline_acc_majority", "roc_auc", "balanced_accuracy", "f1", "mcc"]
        print("\nScores (eval rows):")
        print(pd.DataFrame(ok)[cols].round(4).to_string(index=False))
    if args.smoke:
        print("\nSMOKE: these numbers are NOT reportable. Test rows used: 0.")
    print(f"\nProtocol check: NaN present in fit data ({nan_rows_fit} rows) -> nothing was filled or dropped.")
    if args.compare_with:
        print("\nRun comparison vs", args.compare_with)
        for line in compare_runs(out_dir, args.compare_with):
            print("  ", line)
    bad = [n for n in ("XGBoost", "RandomForest") if st[n]["status"] != "runnable"]
    if bad:
        print(f"\nSTOP AND ASK: {bad} could not run on NaN. Do not change the protocol without the user.")
        sys.exit(3)


if __name__ == "__main__":
    main()
