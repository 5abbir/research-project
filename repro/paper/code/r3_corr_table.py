#!/usr/bin/env python
"""
r3_corr_table.py  --  Scenario 2 step 2 of Section 4.2 ("build a correlation chart"), TRAIN ROWS ONLY.

Read-only, descriptive. No model, no filling, no test rows, and the target (Potability) is NOT included
(decision B4: the label is not a predictor). It only shows the numbers needed to choose the rule for
"positive co-related" (decision B3). It does NOT choose the rule.

Uses the official split (stratified 80/20, random_state=42) and only its TRAIN rows.
Pearson r is computed on pairwise-complete rows (rows where both columns are present).
Outputs (in --out): s2_corr_matrix_train.csv, s2_corr_candidates_train.csv, s2_corr_report.txt, s2_corr_heatmap_train.png (if matplotlib exists)
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_s1_sklearn_models import split_indices, sha256_file  # noqa: E402


def pearson_pairwise(a, b):
    m = a.notna() & b.notna()
    n = int(m.sum())
    if n < 4:
        return float("nan"), n, float("nan"), float("nan")
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    rc = min(max(r, -0.999999), 0.999999)
    z, se = math.atanh(rc), 1 / math.sqrt(n - 3)
    return r, n, math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        sys.exit(f"STOP: target '{args.target}' not found. Columns: {list(df.columns)}")
    y = df[args.target].to_numpy()
    train_idx, test_idx = split_indices(y, args.seed)
    train = df.iloc[train_idx]
    del df  # test rows leave memory here; nothing below can use them
    feats = [c for c in train.columns if c != args.target]
    X = train[feats]
    miss = X.isna().sum()
    incomplete = [c for c in feats if miss[c] > 0]
    complete = [c for c in feats if miss[c] == 0]
    os.makedirs(args.out, exist_ok=True)

    mat = pd.DataFrame(index=feats, columns=feats, dtype=float)
    for a in feats:
        for b in feats:
            mat.loc[a, b] = 1.0 if a == b else pearson_pairwise(X[a], X[b])[0]
    mat.to_csv(os.path.join(args.out, "s2_corr_matrix_train.csv"))

    rows = []
    for tgt in incomplete:
        for c in feats:
            if c == tgt:
                continue
            r, n, lo, hi = pearson_pairwise(X[tgt], X[c])
            rows.append({"column_to_fill": tgt, "candidate_predictor": c,
                         "candidate_is_complete_column": c in complete, "n_pairs": n, "pearson_r": round(r, 4),
                         "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                         "r_gt_0": bool(r > 0), "r_gt_0.1": bool(r > 0.1),
                         "ci_excludes_0": bool(lo > 0 or hi < 0)})
    cand = pd.DataFrame(rows)
    cand.to_csv(os.path.join(args.out, "s2_corr_candidates_train.csv"), index=False)

    L = ["Scenario 2, Section 4.2 step 2: correlation table (TRAIN ROWS ONLY; test rows not used; target not included)",
         f"csv sha256: {sha256_file(args.csv)}",
         f"official split seed {args.seed}: train rows {len(train)} (test rows {len(test_idx)} not used)",
         f"missing cells in train rows: { {c: int(miss[c]) for c in incomplete} }; complete columns: {complete}",
         "Pearson r on pairwise-complete rows; 95% CI by Fisher z (approximate).",
         "This report does NOT choose the 'positive co-related' rule (decision B3 is the user's).", ""]
    for tgt in incomplete:
        sub = cand[cand.column_to_fill == tgt].sort_values("pearson_r", ascending=False)
        L.append(f"== column to fill: {tgt} ({int(miss[tgt])} missing of {len(train)} train rows)")
        L.append(sub.drop(columns=["column_to_fill"]).to_string(index=False))
        L.append(f"   r > 0: {int(sub['r_gt_0'].sum())} candidates; r > 0.1: {int(sub['r_gt_0.1'].sum())}; "
                 f"CI excludes 0: {int(sub['ci_excludes_0'].sum())}; largest |r| = {sub['pearson_r'].abs().max():.4f}")
        L.append("")
    L.append("Reading guide: with this many rows an |r| below about 0.05 is hard to tell from 0; a rule 'r > 0' would then "
             "admit columns by noise. If a rule leaves NO candidate for a column, the script of step 3 must stop and ask (never a silent fallback).")
    txt = "\n".join(L)
    with open(os.path.join(args.out, "s2_corr_report.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right")
        ax.set_yticks(range(len(feats)), feats)
        for i in range(len(feats)):
            for j in range(len(feats)):
                ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Pearson r, train rows only (pairwise complete)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "s2_corr_heatmap_train.png"), dpi=150)
        print("\nWrote heat map s2_corr_heatmap_train.png")
    except Exception as e:  # noqa: BLE001
        print(f"\nHeat map skipped ({type(e).__name__}: {str(e)[:100]}). Tables were written.")


if __name__ == "__main__":
    main()
