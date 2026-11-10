#!/usr/bin/env python
"""
r3_make_smote_trackL.py  --  Scenario 2b, Track L (LEAK-FREE): SMOTE on the FILLED TRAIN rows only.

LABEL: "Section 4.2 correlation-based imputation + SMOTE (train rows only)", Track L.
Reproduction results, not validated performance.

Input : results/s2/s2_filled_train_L.csv (written by r3_fill_trackL.py; train rows only).
Output: results/s2/s2_smote_train_L.csv = the original filled train rows (unchanged, first) + synthetic rows.
        Columns: 9 features, Potability, row_id (original row id; -1 for synthetic), is_synthetic (0/1).
SMOTE settings (decision B12): imblearn defaults, k_neighbors=5, sampling_strategy='auto' (minority raised to the
majority count), random_state=42, NO scaling, NO feature selection. Only the 9 feature columns go into SMOTE;
row_id and is_synthetic are never features.

S2a (fill only) needs no file from this script: the tools read s2_filled_train_L.csv directly.
The TEST file (s2_filled_test_L.csv) is never resampled and never used to fit anything. In full mode it is only
checked (hash before and after, columns, NaN, row ids disjoint from train); in smoke mode it is NOT opened at all.

Modes: --smoke : 500-row stratified subsample of the filled TRAIN file; test file not opened; output -> <out>/smote_smoke/.
       default : full train file; test file checked read-only.
"""
import argparse
import json
import os
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_s1_sklearn_models import sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", required=True, help="s2_filled_train_L.csv")
    ap.add_argument("--test-csv", default=None, help="s2_filled_test_L.csv (full mode only; read-only check)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-neighbors", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=500)
    ap.add_argument("--compare-with", default=None, help="path of a previous s2_smote_train_L.csv (exact/approx check)")
    args = ap.parse_args()

    try:
        from imblearn.over_sampling import SMOTE
        import imblearn
    except ImportError as e:
        sys.exit(f"STOP: imblearn cannot be imported ({e}). Activate env_main (prompt must show (env_main)). Nothing was run.")

    tr = pd.read_csv(args.train_csv)
    for c in (args.target, "row_id"):
        if c not in tr.columns:
            sys.exit(f"STOP: column '{c}' not found in {args.train_csv}. Columns: {list(tr.columns)}")
    feats = [c for c in tr.columns if c not in (args.target, "row_id")]
    if tr[feats].isna().any().any():
        sys.exit("STOP: NaN in the filled train file. Run r3_fill_trackL.py first (SMOTE cannot handle NaN).")
    if set(np.unique(tr[args.target])) != {0, 1}:
        sys.exit("STOP: target must contain exactly the labels 0 and 1")
    if tr["row_id"].duplicated().any() or (tr["row_id"] < 0).any():
        sys.exit("STOP: row_id must be unique and non-negative in the train file")

    mode = "smoke" if args.smoke else "full"
    out_dir = os.path.join(args.out, "smote_smoke") if args.smoke else args.out
    os.makedirs(out_dir, exist_ok=True)
    train_sha = sha256_file(args.train_csv)

    if args.smoke:
        parts = []
        for lab in sorted(tr[args.target].unique()):  # explicit loop: groupby.apply behaves differently across pandas versions
            g = tr[tr[args.target] == lab]
            parts.append(g.sample(n=max(args.k_neighbors + 2, round(args.smoke_n * len(g) / len(tr))), random_state=args.seed))
        tr = pd.concat(parts).sort_values("row_id").reset_index(drop=True)
    X, y = tr[feats], tr[args.target]
    counts = y.value_counts().sort_index()
    maj_label, min_label = int(counts.idxmax()), int(counts.idxmin())
    n_maj, n_min = int(counts.max()), int(counts.min())
    if n_maj == n_min:
        sys.exit("STOP: classes already balanced; nothing to resample. Ask the user.")

    # ---- test file: read-only checks (full mode only; smoke does not open it)
    test_report = ["test file: NOT opened (smoke mode)"]
    test_sha_before = test_sha_after = None
    if not args.smoke:
        if not args.test_csv or not os.path.exists(args.test_csv):
            sys.exit("STOP: full mode needs --test-csv (read-only check). Nothing was run.")
        test_sha_before = sha256_file(args.test_csv)
        te = pd.read_csv(args.test_csv)
        problems = []
        if list(te.columns) != list(pd.read_csv(args.train_csv, nrows=0).columns):
            problems.append("test columns differ from train columns")
        if te[feats].isna().any().any():
            problems.append("NaN in test features")
        if set(te["row_id"]) & set(tr["row_id"]):
            problems.append("row_id overlap between train and test")
        if problems:
            sys.exit("STOP: " + "; ".join(problems))
        test_report = [f"test file (read-only check): {args.test_csv}", f"  rows {len(te)}, class counts "
                       f"{ {int(k): int(v) for k, v in te[args.target].value_counts().sort_index().items()} }, NaN 0, "
                       f"row ids disjoint from train; never resampled, never used for fitting",
                       f"  sha256 before: {test_sha_before}"]
        del te

    # ---- SMOTE (train rows only)
    sm = SMOTE(random_state=args.seed, k_neighbors=args.k_neighbors)  # sampling_strategy='auto' (default)
    Xr, yr = sm.fit_resample(X, y)
    Xr = pd.DataFrame(Xr, columns=feats).reset_index(drop=True)
    yr = pd.Series(yr, name=args.target).reset_index(drop=True)
    n0 = len(X)
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    chk("originals come first and are unchanged", np.array_equal(Xr.iloc[:n0].to_numpy(), X.to_numpy())
        and np.array_equal(yr.iloc[:n0].to_numpy(), y.to_numpy()), f"{n0} rows")
    syn = Xr.iloc[n0:].reset_index(drop=True)
    ysyn = yr.iloc[n0:].reset_index(drop=True)
    chk("number of synthetic rows = majority - minority", len(syn) == n_maj - n_min, f"{len(syn)} = {n_maj} - {n_min}")
    chk("all synthetic rows belong to the minority class", bool((ysyn == min_label).all()), f"label {min_label}")
    rc = yr.value_counts()
    chk("classes balanced after SMOTE", rc.nunique() == 1, f"{ {int(k): int(v) for k, v in rc.sort_index().items()} }")
    chk("no NaN in the resampled features", not Xr.isna().any().any())
    mn = X[y == min_label].min()
    mx = X[y == min_label].max()
    tol = 1e-9 * (mx - mn).abs().max()
    inside = ((syn >= mn - tol) & (syn <= mx + tol)).all().all()
    chk("synthetic values lie inside the original minority-class range of every feature (interpolation property)", inside)
    orig_keys = set(map(tuple, np.round(X.to_numpy(), 9)))
    n_dup = int(sum(tuple(r) in orig_keys for r in np.round(syn.to_numpy(), 9)))
    chk("synthetic rows that exactly copy an original row (informational, expected about 0)", True, f"{n_dup} of {len(syn)}")

    out = Xr.copy()
    out[args.target] = yr.to_numpy()
    out["row_id"] = np.concatenate([tr["row_id"].to_numpy(), np.full(len(syn), -1)])
    out["is_synthetic"] = np.concatenate([np.zeros(n0, dtype=int), np.ones(len(syn), dtype=int)])
    out_path = os.path.join(out_dir, "s2_smote_train_L.csv")
    out.to_csv(out_path, index=False)

    if not args.smoke:
        test_sha_after = sha256_file(args.test_csv)
        chk("test file unchanged by this script (sha256 before = after)", test_sha_before == test_sha_after)
        test_report.append(f"  sha256 after:  {test_sha_after}")

    # ---- distance-dominance diagnostic (verifies the unscaled-distance limitation from the data)
    var = X[y == min_label].var()
    share = (var / var.sum() * 100).round(2)
    diag = pd.DataFrame({"sd_minority_class": X[y == min_label].std().round(4), "share_of_squared_distance_variance_pct": share})

    out_sha = sha256_file(out_path)
    rep = [f"mode: {mode}" + ("  (NOT reportable)" if args.smoke else ""),
           f"train file: {args.train_csv}  sha256: {train_sha}",
           f"train rows given to SMOTE: {n0}; class counts {{0: {int(counts.get(0, 0))}, 1: {int(counts.get(1, 0))}}}; "
           f"majority label {maj_label} ({n_maj}), minority label {min_label} ({n_min})",
           f"SMOTE: imblearn {imblearn.__version__}, defaults (k_neighbors={args.k_neighbors}, sampling_strategy='auto'), "
           f"random_state={args.seed}, no scaling; features used: {feats}",
           f"output: {out_path}  rows {len(out)} ({n0} original + {len(syn)} synthetic)  sha256: {out_sha}"] + test_report
    with open(os.path.join(out_dir, "s2_smote_report_L.txt"), "w") as f:
        f.write("\n".join(rep) + "\n\nchecks:\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}: {n} ({d})" for n, ok, d in checks)
                + "\n\ndistance dominance (minority class):\n" + diag.to_string() + "\n")
    with open(os.path.join(out_dir, "s2_smote_run_info_L.json"), "w") as f:
        json.dump({"mode": mode, "track": "L (leak-free, primary), S2b", "timestamp": datetime.now().isoformat(timespec="seconds"),
                   "command": " ".join(sys.argv), "seed": args.seed, "k_neighbors": args.k_neighbors,
                   "train_sha256": train_sha, "output_sha256": out_sha, "test_sha256_before": test_sha_before,
                   "test_sha256_after": test_sha_after, "n_original": n0, "n_synthetic": int(len(syn)),
                   "versions": {"python": sys.version.split()[0], "imblearn": imblearn.__version__, "sklearn": sklearn.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()}}, f, indent=2)

    print(f"=== S2b SMOTE (Track L) | mode={mode} | out={out_dir} ===")
    print("\n".join(rep))
    print("\nChecks:")
    for n, ok, d in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {n} ({d})")
    print("\nDistance dominance among minority-class rows (SMOTE uses unscaled Euclidean distance):")
    print(diag.to_string())
    top = diag["share_of_squared_distance_variance_pct"].idxmax()
    print(f"-> '{top}' carries {diag.loc[top, 'share_of_squared_distance_variance_pct']}% of the squared-distance variance.")
    if args.compare_with:
        if os.path.exists(args.compare_with):
            same = sha256_file(args.compare_with) == out_sha
            print(f"\nRun comparison vs {args.compare_with}: {'EXACT match (identical file)' if same else 'DIFFERENT file'}")
        else:
            print(f"\nRun comparison skipped: {args.compare_with} not found")
    if args.smoke:
        print("\nSMOKE: not reportable. Test file was not opened.")
    if not all(ok for _, ok, _ in checks):
        print("\nWARNING: at least one check FAILED. Do not use this file; tell the user.")
        sys.exit(4)


if __name__ == "__main__":
    main()
