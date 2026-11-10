#!/usr/bin/env python
"""
r4_s3_sklearn_models.py  --  Scenario 3 (S3), Track L (LEAK-FREE): XGBoost, RandomForest, AdaBoost on fuzzy-rough-selected feature sets.

LABEL: reproduction results, not validated performance.  Plan = S3_PROTOCOL.md section 8 (approved 2026-11-10).
Train = the S2b train rows (filled + SMOTE, train only), restricted to a feature set. Test = the same 656 ORIGINAL filled test rows
with the same columns, never resampled. Everything except the feature set is identical to S2b, so S3 vs S2b-ref isolates feature selection.
Do NOT compare with Scenario 1 as if the missing-value treatment were identical (S1 = raw CSV).

Variants (5 feature sets x 3 models = 15 fits; library defaults, seed 42, n_jobs fixed, single fit, no CV, no tuning, no scaling):
  S2b-ref     all 9 features (re-fit; must reproduce the stored S2b numbers; = our strict FRS reduct)       role: Reference
  S3-paperM1  paper list [6,7,3,2,1,5,8] under M1 (Table 2 order, mapping [P])                              role: Primary
  S3-own7     first 7 features of OUR QuickReduct order (size 7 taken from the paper, not discovered)       role: Primary
  S3-paperM2  paper list under M2 (raw CSV order)                                                           role: Sensitivity
  S3-stab7    first 7 of the order found on the ORIGINAL (non-synthetic) train rows only                    role: Sensitivity
Feature lists are read BY NAME from s3_frs_selected_L.json (made by r4_frs_select.py from train rows only); columns are always passed
to the models in the train file's column order (so a different listing order cannot change a fit).
Each model is scored ONCE per variant on the test rows; no variant or model is selected, tuned or rejected on test performance.
After scoring: paired bootstrap of (S3 variant - S2b-ref) on the saved test predictions (no refitting, seed fixed): descriptive
uncertainty of the difference on THIS test split only (does not include training-set variability).

Modes:
  --smoke : TRAIN rows only; the test file is NOT opened. Fits on <s2-dir>/smote_smoke/s2_smote_train_L.csv, evaluates on --smoke-eval-n
            stratified ORIGINAL train rows outside that smoke input; feature lists from <s2-dir>/frs_smoke/. Output -> <out>/s3_sklearn_smoke/. Not reportable.
  default : S2b train file, feature lists from <s2-dir>/; scored once on the filled test file.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_s1_sklearn_models import proba_pos, score_all, sha256_file  # noqa: E402
from r3_s2_sklearn_models import MODELS, PAPER, build_models  # noqa: E402

BASE = "Section 4.2 fill + SMOTE (train rows only)"
VARIANTS = [  # (name, role, label)
    ("S2b-ref", "Reference", BASE + ", all 9 features (Track L)"),
    ("S3-paperM1", "Primary", BASE + " + paper's selected list, mapping M1 = Table 2 order [P] (Track L)"),
    ("S3-own7", "Primary", BASE + " + our fuzzy-rough QuickReduct, first 7 (size from paper) (Track L)"),
    ("S3-paperM2", "Sensitivity", BASE + " + paper's selected list, mapping M2 = raw CSV order (Track L)"),
    ("S3-stab7", "Sensitivity", BASE + " + our fuzzy-rough, first 7 of the original-rows-only order (Track L)"),
]


def feature_sets(sel, feats, k):
    return {"S2b-ref": list(feats), "S3-paperM1": sel["paper_list_M1"], "S3-own7": sel["our_prefix_k"],
            "S3-paperM2": sel["paper_list_M2"], "S3-stab7": sel["stability_order_original_rows"][:k]}


def paired_bootstrap(preds, n_boot, seed):
    """preds: DataFrame variant, model, row_id, y_true, proba_class1, y_pred (same test rows for every fit)."""
    rows = []
    rng = np.random.RandomState(seed)
    ids = np.sort(preds["row_id"].unique())
    n = len(ids)
    idx = rng.randint(0, n, size=(n_boot, n))  # the SAME resamples for every variant/model (paired)
    for model in MODELS:
        d = {}
        for var, _, _ in VARIANTS:
            s = preds[(preds["variant"] == var) & (preds["model"] == model)].set_index("row_id").loc[ids]
            d[var] = (s["y_true"].to_numpy(), s["y_pred"].to_numpy(), s["proba_class1"].to_numpy())
        if any(len(v[0]) != n for v in d.values()):
            continue
        acc, auc = {}, {}
        for var, (yt, yp, p1) in d.items():
            correct = (yt == yp).astype(float)
            acc[var] = correct[idx].mean(axis=1)
            auc[var] = np.array([roc_auc_score(yt[i], p1[i]) for i in idx])
        for var, _, _ in VARIANTS[1:]:
            for metric, store, pt in (("accuracy", acc, lambda v: float(np.mean(d[v][0] == d[v][1]))),
                                      ("roc_auc", auc, lambda v: float(roc_auc_score(d[v][0], d[v][2])))):
                diff = store[var] - store["S2b-ref"]
                rows.append({"model": model, "variant": var, "metric": metric, "diff_vs_S2b_ref": pt(var) - pt("S2b-ref"),
                             "boot_ci_low_2.5": float(np.percentile(diff, 2.5)), "boot_ci_high_97.5": float(np.percentile(diff, 97.5)),
                             "interval_includes_0": bool(np.percentile(diff, 2.5) <= 0 <= np.percentile(diff, 97.5)), "n_boot": n_boot})
    return pd.DataFrame(rows)


def compare_dirs(new_dir, old_dir, tol=1e-9):
    out = []
    for fname, keys in (("s3_sklearn_metrics.csv", ["variant", "model"]), ("s3_sklearn_predictions.csv", ["variant", "model", "row_id"])):
        a_p, b_p = os.path.join(new_dir, fname), os.path.join(old_dir, fname)
        if not (os.path.exists(a_p) and os.path.exists(b_p)):
            out.append(f"{fname}: cannot compare (file missing)")
            continue
        m = pd.read_csv(a_p).merge(pd.read_csv(b_p), on=keys, suffixes=("_new", "_old"), how="outer", indicator=True)
        if (m["_merge"] != "both").any():
            out.append(f"{fname}: DIFFERENT row sets between runs")
            continue
        exact, worst = True, 0.0
        for col in [c[:-4] for c in m.columns if c.endswith("_new")]:
            x, y = m[f"{col}_new"], m[f"{col}_old"]
            if pd.api.types.is_numeric_dtype(x):
                if not ((x == y) | (x.isna() & y.isna())).all():
                    exact = False
                    d = (x - y).abs().max()
                    worst = max(worst, float(d) if pd.notna(d) else float("inf"))
            elif not (x.fillna("") == y.fillna("")).all():
                exact, worst = False, float("inf")
        out.append(f"{fname}: EXACT match" if exact else (f"{fname}: APPROXIMATE match (max abs diff {worst:.3g})" if worst <= tol
                                                         else f"{fname}: DIFFERENT (max abs diff {worst:.3g})"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2-dir", required=True, help="results/s2 folder (filled train/test, SMOTE train, FRS outputs)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True, help="output folder (smoke writes to <out>/s3_sklearn_smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-eval-n", type=int, default=300)
    ap.add_argument("--smoke-n-boot", type=int, default=200)
    ap.add_argument("--check-split-with", default=None, help="results/s1/s1_predictions.csv (test row ids must match)")
    ap.add_argument("--s2-reference", default=None, help="results/s2 folder holding the stored s2_sklearn_predictions.csv (S2b-ref must reproduce it)")
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
    p_smote_full = os.path.join(args.s2_dir, "s2_smote_train_L.csv")
    p_smote = os.path.join(args.s2_dir, "smote_smoke", "s2_smote_train_L.csv") if args.smoke else p_smote_full
    frs_dir = os.path.join(args.s2_dir, "frs_smoke") if args.smoke else args.s2_dir
    p_sel, p_info = os.path.join(frs_dir, "s3_frs_selected_L.json"), os.path.join(frs_dir, "s3_frs_run_info_L.json")
    for p in (p_sel, p_info, p_smote):
        if not os.path.exists(p):
            sys.exit(f"STOP: required file not found: {p}. Run the earlier steps first. Nothing was run.")
    mode = "smoke" if args.smoke else "full"
    out_dir = os.path.join(args.out, "s3_sklearn_smoke") if args.smoke else args.out
    os.makedirs(out_dir, exist_ok=True)
    checks, rep = [], []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    rt = dict(float_precision="round_trip")
    ft = pd.read_csv(p_train, **rt)
    sm = pd.read_csv(p_smote, **rt)
    for df_, nm in ((ft, "filled train"), (sm, "SMOTE train")):
        for c in (T, "row_id"):
            if c not in df_.columns:
                sys.exit(f"STOP: column '{c}' missing in {nm} file")
    if "is_synthetic" not in sm.columns:
        sys.exit("STOP: 'is_synthetic' column missing in the SMOTE train file")
    feats = [c for c in ft.columns if c not in (T, "row_id")]
    sel = json.load(open(p_sel))
    info = json.load(open(p_info))
    k = int(sel["k_paper"])
    fsets = feature_sets(sel, feats, k)

    chk("feature list excludes target, row_id and is_synthetic", not ({T, "row_id", "is_synthetic"} & set(feats)), f"{len(feats)} features")
    chk("same feature columns in S2a and S2b train files", set(feats) == set(sm.columns) - {T, "row_id", "is_synthetic"})
    chk("no NaN in any training feature", not ft[feats].isna().any().any() and not sm[feats].isna().any().any())
    chk("FRS json mode matches this run's mode", sel.get("mode") == mode, f"json mode {sel.get('mode')}")
    chk("FRS run read the full SMOTE train file (sha256 in run info = sha256 of s2_smote_train_L.csv)",
        info.get("input_sha256") == sha256_file(p_smote_full), "provenance: selection was computed on these train rows")
    chk("FRS run info says test rows used = 0", info.get("test_rows_used") == 0)
    for name, cols in fsets.items():
        chk(f"{name}: valid feature names, no duplicates, size {9 if name == 'S2b-ref' else k}",
            len(cols) == len(set(cols)) == (9 if name == "S2b-ref" else k) and set(cols) <= set(feats) and not ({T, "row_id", "is_synthetic"} & set(cols)),
            ", ".join(cols))
    fsets = {n: [c for c in feats if c in set(cols)] for n, cols in fsets.items()}  # train-file column order for every fit

    orig = sm[sm["is_synthetic"] == 0]
    syn = sm[sm["is_synthetic"] == 1]
    chk("synthetic rows have row_id -1 and originals have row_id >= 0", bool((syn["row_id"] == -1).all() and (orig["row_id"] >= 0).all()), f"{len(syn)} synthetic")

    test_sha_before = test_sha_after = None
    if args.smoke:
        rem = ft[~ft["row_id"].isin(orig["row_id"])]
        ev_idx, _ = train_test_split(rem.index.to_numpy(), train_size=args.smoke_eval_n, random_state=args.seed, stratify=rem[T])
        ev = ft.loc[np.sort(ev_idx)]
        chk("smoke eval rows are original train rows outside the smoke fit", not (set(ev["row_id"]) & set(orig["row_id"])), f"{len(ev)} rows")
        rep.append("test file: NOT opened (smoke mode). Smoke feature lists come from a subsample of train rows (not reportable).")
    else:
        test_sha_before = sha256_file(p_test)
        te = pd.read_csv(p_test, **rt)
        ev = te
        cc = sm[T].value_counts()
        chk("S2b train classes balanced", cc.nunique() == 1, f"{ {int(a): int(b) for a, b in cc.sort_index().items()} }")
        chk("test columns contain all features", set(feats) <= set(te.columns))
        chk("no NaN in test features", not te[feats].isna().any().any(), f"{len(te)} rows")
        chk("test row ids disjoint from train row ids", not (set(te["row_id"]) & set(ft["row_id"])))
        if args.check_split_with and os.path.exists(args.check_split_with):
            ids = set(pd.read_csv(args.check_split_with)["row_id"])
            chk("test row ids equal the S1 test row ids", ids == set(te["row_id"]), f"{len(ids)} ids in S1 file")
        else:
            rep.append("split check vs S1 skipped (no --check-split-with file)")

    rep = [f"mode: {mode}" + ("  (NOT reportable)" if args.smoke else ""), f"SMOTE train: {p_smote}  sha256 {sha256_file(p_smote)}",
           f"FRS selection json: {p_sel}  sha256 {sha256_file(p_sel)}",
           f"fit rows {len(sm)} {dict(sm[T].value_counts().sort_index())}".replace("np.int64(", "").replace(")", ""),
           f"eval rows {len(ev)} {dict(ev[T].value_counts().sort_index())}".replace("np.int64(", "").replace(")", "")] + rep
    rep += [f"{n} ({len(c)} features): {', '.join(c)}" for n, c in fsets.items()]
    if not args.smoke:
        rep.append(f"test file {p_test}  sha256 before {test_sha_before}")

    if not all(ok for _, ok, _ in checks):
        print("\n".join(rep))
        for n, ok, d in checks:
            print(f"  {'PASS' if ok else 'FAIL'}: {n} ({d})")
        sys.exit("STOP: a data check failed; nothing was fitted. Tell the user.")

    y_fit = sm[T].to_numpy()
    X_ev_all, y_ev, ev_ids = ev, ev[T].to_numpy(), ev["row_id"].to_numpy()
    met, conf, preds, status, tbs = [], [], [], [], {}
    for var, role, label in VARIANTS:
        cols = fsets[var]
        models = build_models(args.seed, args.n_jobs, XGBClassifier)
        for name in MODELS:
            t0 = time.time()
            try:
                m = models[name]
                m.fit(sm[cols], y_fit)
                p1 = proba_pos(m, X_ev_all[cols])
                yp = m.predict(X_ev_all[cols])
                sc, (tn, fp, fn, tp) = score_all(y_ev, yp, p1)
                met.append({"variant": var, "role": role, "result_label": label, "model": name, "run_mode": mode, "n_train": len(sm),
                            "n_features": len(cols), "features": ";".join(cols), **sc})
                conf.append({"variant": var, "model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "n_eval": len(y_ev)})
                preds.append(pd.DataFrame({"variant": var, "model": name, "row_id": ev_ids, "y_true": y_ev, "y_pred": yp, "proba_class1": p1}))
                status.append({"variant": var, "model": name, "status": "runnable", "error_type": "", "error_message": "", "seconds": round(time.time() - t0, 2)})
            except Exception as e:  # noqa: BLE001
                tbs[(var, name)] = traceback.format_exc()
                status.append({"variant": var, "model": name, "status": "failed", "error_type": type(e).__name__,
                               "error_message": str(e).replace("\n", " | ")[:300], "seconds": round(time.time() - t0, 2)})

    if not args.smoke:
        test_sha_after = sha256_file(p_test)
        chk("test file unchanged by this script (sha256 before = after)", test_sha_before == test_sha_after)
        if args.s2_reference:
            pr = os.path.join(args.s2_reference, "s2_sklearn_predictions.csv")
            if os.path.exists(pr) and preds:
                old = pd.read_csv(pr)
                old = old[old["variant"] == "S2b"].rename(columns={"proba_class1": "p_old"})[["model", "row_id", "p_old"]]
                new = pd.concat(preds)
                new = new[new["variant"] == "S2b-ref"][["model", "row_id", "proba_class1"]]
                mm = new.merge(old, on=["model", "row_id"], how="outer", indicator=True)
                if (mm["_merge"] != "both").any():
                    chk("S2b-ref reproduces the stored S2b probabilities", False, "row sets differ")
                else:
                    dmax = float((mm["proba_class1"] - mm["p_old"]).abs().max())
                    chk("S2b-ref reproduces the stored S2b probabilities (tolerance 1e-9)", dmax <= 1e-9, f"max abs diff {dmax:.2e}")
            else:
                rep.append("S2b-ref reproduction check skipped (stored S2 predictions not found)")

    P = pd.concat(preds) if preds else pd.DataFrame(columns=["variant", "model", "row_id", "y_true", "y_pred", "proba_class1"])
    boot = pd.DataFrame()
    if len(status) and all(s["status"] == "runnable" for s in status):
        boot = paired_bootstrap(P, args.smoke_n_boot if args.smoke else args.n_boot, args.seed)

    pd.DataFrame(met).to_csv(os.path.join(out_dir, "s3_sklearn_metrics.csv"), index=False)
    pd.DataFrame(conf).to_csv(os.path.join(out_dir, "s3_sklearn_confusion.csv"), index=False)
    pd.DataFrame(status).to_csv(os.path.join(out_dir, "s3_sklearn_status.csv"), index=False)
    P.to_csv(os.path.join(out_dir, "s3_sklearn_predictions.csv"), index=False)
    boot.to_csv(os.path.join(out_dir, "s3_bootstrap_diffs.csv"), index=False)
    for (var, name), tb in tbs.items():
        with open(os.path.join(out_dir, f"s3_sklearn_traceback_{var}_{name}.txt"), "w") as f:
            f.write(tb)
    pd.DataFrame(PAPER[3:], columns=["paper_table", "model", "precision_pct", "sensitivity_pct", "specificity_pct", "f1_pct", "accuracy_pct"]).assign(
        use="REFERENCE ONLY (paper Table 7); split/seed/versions/selection details unknown").to_csv(os.path.join(out_dir, "s3_paper_reference.csv"), index=False)
    with open(os.path.join(out_dir, "s3_sklearn_report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n\nchecks:\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}: {n} ({d})" for n, ok, d in checks) + "\n")
    with open(os.path.join(out_dir, "s3_sklearn_run_info.json"), "w") as f:
        json.dump({"mode": mode, "track": "L", "timestamp": datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv),
                   "seed": args.seed, "n_jobs": args.n_jobs, "n_boot": args.smoke_n_boot if args.smoke else args.n_boot,
                   "test_file_sha256_before": test_sha_before, "test_file_sha256_after": test_sha_after,
                   "feature_sets": fsets, "frs_json_sha256": sha256_file(p_sel),
                   "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()}}, f, indent=2)

    print(f"=== S3 sklearn models (Track L) | mode={mode} | out={out_dir} ===")
    print("\n".join(rep))
    print("\nChecks:")
    for n, ok, d in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {n} ({d})")
    print("\nModel status:")
    print(pd.DataFrame(status)[["variant", "model", "status", "error_type", "seconds"]].to_string(index=False))
    if met:
        cols = ["variant", "role", "model", "n_features", "accuracy", "baseline_acc_majority", "precision", "recall_sensitivity",
                "specificity", "f1", "roc_auc", "balanced_accuracy", "mcc"]
        print("\nScores on the eval rows (one scoring per fit; nothing selected on them):")
        print(pd.DataFrame(met)[cols].round(4).to_string(index=False))
        print(f"\nOne split of {len(y_ev)} eval rows: accuracy standard error is about {100 * (0.61 * 0.39 / len(y_ev)) ** 0.5:.1f} points (rough guide).")
    if len(boot):
        print(f"\nPaired bootstrap of (variant - S2b-ref), {int(boot['n_boot'].iloc[0])} resamples of the eval rows, no refitting "
              "(descriptive; this split only; not used for selection):")
        print(boot.drop(columns=["n_boot"]).round(4).to_string(index=False))
    if args.smoke:
        print("\nSMOKE: not reportable. Test file was not opened.")
    if args.compare_with:
        print("\nRun comparison vs", args.compare_with, "(the first run of this same script, inputs and settings)")
        for line in compare_dirs(out_dir, args.compare_with if not args.smoke else os.path.join(args.compare_with, "s3_sklearn_smoke")):
            print("  ", line)
    bad = [s for s in status if s["status"] != "runnable"]
    if bad or not all(ok for _, ok, _ in checks):
        print(f"\nSTOP AND ASK: {len(bad)} model fit(s) failed or a check failed. Do not change the protocol without the user.")
        sys.exit(3)


if __name__ == "__main__":
    main()
