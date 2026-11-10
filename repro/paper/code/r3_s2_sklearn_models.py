#!/usr/bin/env python
"""
r3_s2_sklearn_models.py  --  Scenario 2, Track L (LEAK-FREE): XGBoost, RandomForest, AdaBoost on S2a and S2b.

LABEL: reproduction results, not validated performance.
  S2a = "Section 4.2 correlation-based imputation (Track L)"                       train = filled train rows (imbalanced)
  S2b = "Section 4.2 correlation-based imputation + SMOTE on train rows (Track L)"  train = filled + SMOTE train rows
Test = the same 656 ORIGINAL rows for both variants (filled by train-fitted models, never resampled).
Do NOT compare these numbers with Scenario 1 as if the missing-value treatment were identical (S1 = raw CSV).
S2a vs S2b share the same fill and the same test rows, so their difference isolates SMOTE.

Protocol (S2_PROTOCOL.md section 8): library defaults, seed 42, n_jobs fixed, single fit, no CV, no tuning, no scaling,
no selection. row_id and is_synthetic are never features; the target is never a feature. Every model is scored ONCE on
the test rows; none is selected, tuned or rejected on test performance. Predicted probabilities are saved.

Modes:
  --smoke : TRAIN rows only; the test file is NOT opened. Fits on <out>/smote_smoke/s2_smote_train_L.csv (S2b) and on its
            original rows (S2a); evaluates on --smoke-eval-n stratified ORIGINAL train rows that are NOT in the smoke SMOTE
            input (no synthetic neighbour of an evaluation row is in the fit). Output -> <out>/sklearn_smoke/. Not reportable.
  default : full run: S2a on the filled train file, S2b on the SMOTE train file, scored once on the filled test file.
"""
import argparse
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_s1_sklearn_models import proba_pos, score_all, sha256_file  # noqa: E402

LABELS = {"S2a": "Section 4.2 correlation-based imputation (Track L)",
          "S2b": "Section 4.2 correlation-based imputation + SMOTE on train rows (Track L)"}
MODELS = ["XGBoost", "RandomForest", "AdaBoost"]
# Paper Tables 6 and 7 (percent). REFERENCE ONLY. The paper does not say whether Table 6 includes SMOTE.
PAPER = [("Table 6 (phase 1, 'after filling null values')", "XGBoost", 94.58, 94.19, 96.86, 94.38, 95.88),
         ("Table 6 (phase 1, 'after filling null values')", "RandomForest", 87.98, 85.062, 93.25, 86.49, 90.24),
         ("Table 6 (phase 1, 'after filling null values')", "AdaBoost", 91.44, 84.23, 95.42, 87.68, 91.31),
         ("Table 7 (phase 4, 'after filling null values and apply fuzzy rough set')", "XGBoost", 98.30, 99.31, 98.39, 98.80, 98.83),
         ("Table 7 (phase 4, 'after filling null values and apply fuzzy rough set')", "RandomForest", 95.18, 94.86, 95.49, 95.02, 95.19),
         ("Table 7 (phase 4, 'after filling null values and apply fuzzy rough set')", "AdaBoost", 90.96, 93.15, 91.31, 92.04, 92.21)]


def build_models(seed, n_jobs, XGBClassifier):
    return {"XGBoost": XGBClassifier(random_state=seed, n_jobs=n_jobs),
            "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=n_jobs),
            "AdaBoost": AdaBoostClassifier(random_state=seed)}


def compare_dirs(new_dir, old_dir, tol=1e-9):
    out = []
    for fname, keys in (("s2_sklearn_metrics.csv", ["variant", "model"]),
                        ("s2_sklearn_predictions.csv", ["variant", "model", "row_id"])):
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
            if col in keys or col == "note":
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
    ap.add_argument("--s2-dir", required=True, help="results/s2 folder holding the filled/SMOTE/test files")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True, help="output folder (smoke writes to <out>/sklearn_smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-eval-n", type=int, default=300)
    ap.add_argument("--check-split-with", default=None, help="results/s1/s1_predictions.csv (test row ids must match)")
    ap.add_argument("--compare-with", default=None, help="folder of a previous run of this script")
    args = ap.parse_args()

    try:
        from xgboost import XGBClassifier
        import xgboost
    except ImportError as e:
        sys.exit(f"STOP: xgboost cannot be imported ({e}). Activate env_main (prompt must show (env_main)). Nothing was run.")

    T = args.target
    p_train = os.path.join(args.s2_dir, "s2_filled_train_L.csv")
    p_test = os.path.join(args.s2_dir, "s2_filled_test_L.csv")
    p_smote = os.path.join(args.s2_dir, "s2_smote_train_L.csv")
    p_smote_smoke = os.path.join(args.s2_dir, "smote_smoke", "s2_smote_train_L.csv")
    mode = "smoke" if args.smoke else "full"
    out_dir = os.path.join(args.out, "sklearn_smoke") if args.smoke else args.out
    os.makedirs(out_dir, exist_ok=True)

    checks, rep = [], []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    rt = dict(float_precision="round_trip")  # exact text -> float conversion (the default parser can be 1 ulp off)
    ft = pd.read_csv(p_train, **rt)
    sm = pd.read_csv(p_smote_smoke if args.smoke else p_smote, **rt)
    for df_, nm in ((ft, "filled train"), (sm, "SMOTE train")):
        for c in (T, "row_id"):
            if c not in df_.columns:
                sys.exit(f"STOP: column '{c}' missing in {nm} file")
    if "is_synthetic" not in sm.columns:
        sys.exit("STOP: 'is_synthetic' column missing in the SMOTE train file")
    feats = [c for c in ft.columns if c not in (T, "row_id")]
    chk("feature list excludes target, row_id and is_synthetic",
        T not in feats and "row_id" not in feats and "is_synthetic" not in feats, f"{len(feats)} features")
    chk("same feature columns in S2a and S2b train files", set(feats) == set(sm.columns) - {T, "row_id", "is_synthetic"})
    chk("no NaN in any training feature", not ft[feats].isna().any().any() and not sm[feats].isna().any().any())
    orig = sm[sm["is_synthetic"] == 0]
    syn = sm[sm["is_synthetic"] == 1]
    chk("synthetic rows have row_id -1 and originals have row_id >= 0", bool((syn["row_id"] == -1).all() and (orig["row_id"] >= 0).all()),
        f"{len(syn)} synthetic")

    if args.smoke:
        # S2a smoke train = the originals inside the smoke SMOTE file; eval = ORIGINAL train rows outside it. Test file NOT opened.
        fitA = orig
        fitB = sm
        rem = ft[~ft["row_id"].isin(orig["row_id"])]
        ev_idx, _ = train_test_split(rem.index.to_numpy(), train_size=args.smoke_eval_n, random_state=args.seed, stratify=rem[T])
        ev = ft.loc[np.sort(ev_idx)]
        chk("smoke eval rows are original train rows outside the smoke fit", not (set(ev["row_id"]) & set(orig["row_id"])), f"{len(ev)} rows")
        chk("smoke eval row ids are a subset of the filled train file", set(ev["row_id"]) <= set(ft["row_id"]))
        test_sha_before = test_sha_after = None
        rep.append("test file: NOT opened (smoke mode)")
    else:
        test_sha_before = sha256_file(p_test)
        te = pd.read_csv(p_test, **rt)
        fitA, fitB, ev = ft, sm, te
        same_shape = len(orig) == len(ft)
        rel = float(np.max(np.abs(orig[feats].to_numpy() - ft[feats].to_numpy()) / np.maximum(np.abs(ft[feats].to_numpy()), 1e-300))) if same_shape else float("inf")
        o_ok = (same_shape and list(orig["row_id"]) == list(ft["row_id"]) and np.array_equal(orig[T].to_numpy(), ft[T].to_numpy())
                and rel <= 1e-12)
        chk("S2b originals equal the S2a train rows (same ids, labels, order; feature values equal up to text round-off <= 1e-12 relative)",
            o_ok, f"{len(orig)} rows; max relative difference {rel:.2e}")
        cc = sm[T].value_counts()
        chk("S2b train classes balanced", cc.nunique() == 1, f"{ {int(k): int(v) for k, v in cc.sort_index().items()} }")
        chk("test columns contain all features", set(feats) <= set(te.columns))
        chk("no NaN in test features", not te[feats].isna().any().any(), f"{len(te)} rows")
        chk("test row ids disjoint from train row ids", not (set(te["row_id"]) & set(ft["row_id"])))
        if args.check_split_with and os.path.exists(args.check_split_with):
            ids = set(pd.read_csv(args.check_split_with)["row_id"])
            chk("test row ids equal the S1 test row ids", ids == set(te["row_id"]), f"{len(ids)} ids in S1 file")
        else:
            rep.append("split check vs S1 skipped (no --check-split-with file)")

    rep = [f"mode: {mode}" + ("  (NOT reportable)" if args.smoke else ""),
           f"filled train: {p_train}  sha256 {sha256_file(p_train)}",
           f"SMOTE train: {p_smote_smoke if args.smoke else p_smote}  sha256 {sha256_file(p_smote_smoke if args.smoke else p_smote)}"] + rep
    rep += [f"S2a fit rows {len(fitA)} {dict(fitA[T].value_counts().sort_index())}".replace("np.int64(", "").replace(")", ""),
            f"S2b fit rows {len(fitB)} {dict(fitB[T].value_counts().sort_index())}".replace("np.int64(", "").replace(")", ""),
            f"eval rows {len(ev)} {dict(ev[T].value_counts().sort_index())}".replace("np.int64(", "").replace(")", "")]
    if not args.smoke:
        rep.append(f"test file {p_test}  sha256 before {test_sha_before}")

    if not all(ok for _, ok, _ in checks):
        print("\n".join(rep))
        for n, ok, d in checks:
            print(f"  {'PASS' if ok else 'FAIL'}: {n} ({d})")
        sys.exit("STOP: a data check failed; nothing was fitted. Tell the user.")

    X_ev, y_ev, ev_ids = ev[feats], ev[T].to_numpy(), ev["row_id"].to_numpy()
    met, conf, preds, status, tbs = [], [], [], [], {}
    for var, fit in (("S2a", fitA), ("S2b", fitB)):
        Xf, yf = fit[feats], fit[T].to_numpy()
        models = build_models(args.seed, args.n_jobs, XGBClassifier)
        for name in MODELS:
            t0 = time.time()
            try:
                m = models[name]
                m.fit(Xf, yf)
                p1 = proba_pos(m, X_ev)
                yp = m.predict(X_ev)
                sc, (tn, fp, fn, tp) = score_all(y_ev, yp, p1)
                met.append({"variant": var, "result_label": LABELS[var], "model": name, "role": "Primary", "run_mode": mode,
                            "n_train": len(fit), **sc})
                conf.append({"variant": var, "model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "n_eval": len(y_ev)})
                preds.append(pd.DataFrame({"variant": var, "model": name, "row_id": ev_ids, "y_true": y_ev, "y_pred": yp, "proba_class1": p1}))
                status.append({"variant": var, "model": name, "status": "runnable", "error_type": "", "error_message": "",
                               "seconds": round(time.time() - t0, 2)})
            except Exception as e:  # noqa: BLE001
                tbs[(var, name)] = traceback.format_exc()
                status.append({"variant": var, "model": name, "status": "failed", "error_type": type(e).__name__,
                               "error_message": str(e).replace("\n", " | ")[:300], "seconds": round(time.time() - t0, 2)})

    if not args.smoke:
        test_sha_after = sha256_file(p_test)
        chk("test file unchanged by this script (sha256 before = after)", test_sha_before == test_sha_after)

    pd.DataFrame(met).to_csv(os.path.join(out_dir, "s2_sklearn_metrics.csv"), index=False)
    pd.DataFrame(conf).to_csv(os.path.join(out_dir, "s2_sklearn_confusion.csv"), index=False)
    pd.DataFrame(status).to_csv(os.path.join(out_dir, "s2_sklearn_status.csv"), index=False)
    (pd.concat(preds) if preds else pd.DataFrame(columns=["variant", "model", "row_id", "y_true", "y_pred", "proba_class1"])).to_csv(
        os.path.join(out_dir, "s2_sklearn_predictions.csv"), index=False)
    for (var, name), tb in tbs.items():
        with open(os.path.join(out_dir, f"s2_sklearn_traceback_{var}_{name}.txt"), "w") as f:
            f.write(tb)
    pd.DataFrame(PAPER, columns=["paper_table", "model", "precision_pct", "sensitivity_pct", "specificity_pct", "f1_pct",
                                 "accuracy_pct"]).assign(use="REFERENCE ONLY; the paper does not say whether Table 6 includes SMOTE; "
                                                         "split/seed/versions unknown").to_csv(
        os.path.join(out_dir, "s2_paper_reference.csv"), index=False)
    with open(os.path.join(out_dir, "s2_sklearn_report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n\nchecks:\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}: {n} ({d})" for n, ok, d in checks) + "\n")
    with open(os.path.join(out_dir, "s2_sklearn_run_info.json"), "w") as f:
        json.dump({"mode": mode, "track": "L", "timestamp": datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv),
                   "seed": args.seed, "n_jobs": args.n_jobs, "test_file_sha256_before": test_sha_before,
                   "test_file_sha256_after": test_sha_after, "labels": LABELS,
                   "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()}}, f, indent=2)

    print(f"=== S2 sklearn models (Track L) | mode={mode} | out={out_dir} ===")
    print("\n".join(rep))
    print("\nChecks:")
    for n, ok, d in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {n} ({d})")
    print("\nModel status:")
    print(pd.DataFrame(status)[["variant", "model", "status", "error_type", "seconds"]].to_string(index=False))
    if met:
        cols = ["variant", "model", "n_train", "n_eval", "accuracy", "baseline_acc_majority", "precision", "recall_sensitivity",
                "specificity", "f1", "roc_auc", "balanced_accuracy", "mcc"]
        print("\nScores on the eval rows (one scoring per model; nothing selected on them):")
        print(pd.DataFrame(met)[cols].round(4).to_string(index=False))
        print(f"\nOne split of {len(y_ev)} eval rows: accuracy standard error is about {100 * (0.61 * 0.39 / len(y_ev)) ** 0.5:.1f} points (rough guide).")
    if args.smoke:
        print("\nSMOKE: not reportable. Test file was not opened.")
    if args.compare_with:
        print("\nRun comparison vs", args.compare_with)
        for line in compare_dirs(out_dir, args.compare_with if not args.smoke else os.path.join(args.compare_with, "sklearn_smoke")):
            print("  ", line)
    bad = [s for s in status if s["status"] != "runnable"]
    if bad or not all(ok for _, ok, _ in checks):
        print(f"\nSTOP AND ASK: {len(bad)} model fit(s) failed or a check failed. Do not change the protocol without the user.")
        sys.exit(3)


if __name__ == "__main__":
    main()
