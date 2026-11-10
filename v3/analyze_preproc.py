"""Summaries, factor effects and paired statistics for the v3 study.
  python analyze_preproc.py --name water
Reads whichever of screen_raw.csv / pycaret_raw.csv / autogluon_raw.csv exist.
Stats: Nadeau-Bengio corrected resampled t-test (repeated splits overlap), bootstrap 95% CI, Holm correction.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common as c

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--base", default=c.BASE_COMBO, help="combo that everything is compared against")
args = p.parse_args()
name = args.name
d = c.out_dir(name)
METRICS = ["test_auc", "test_acc", "test_bal_acc", "test_f1", "test_mcc"]
ratio = c.TEST_SIZE / (1 - c.TEST_SIZE)
rng = np.random.default_rng(0)


def paired(piv, base, metric):
    rows = []
    for combo in piv.columns:
        if combo == base:
            continue
        x = (piv[combo] - piv[base]).dropna().values
        J = len(x)
        if J < 3 or x.std(ddof=1) == 0:
            continue
        t = x.mean() / np.sqrt((1 / J + ratio) * x.var(ddof=1))
        pr = 2 * stats.t.sf(abs(t), J - 1)
        boots = [rng.choice(x, J).mean() for _ in range(3000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"combo": combo, "metric": metric, "mean_diff_vs_base": x.mean(),
                     "ci95_low": lo, "ci95_high": hi, "p_raw": pr, "n": J})
    if not rows:
        return pd.DataFrame()
    r = pd.DataFrame(rows)
    order = r["p_raw"].argsort().values  # Holm step-down
    adj, run = np.empty(len(r)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(r) - rank) * r["p_raw"].iloc[i])
        adj[i] = min(1.0, run)
    r["p_holm"] = adj
    return r


for fname, label in [("screen_raw.csv", "SCREENING (Random Forest)"), ("pycaret_raw.csv", "PYCARET"),
                     ("autogluon_raw.csv", "AUTOGLUON")]:
    path = os.path.join(d, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    tag = fname.replace("_raw.csv", "")
    print("\n" + "=" * 78 + f"\n{label}: {df['combo'].nunique()} combos x {df['repeat'].nunique()} repeats\n" + "=" * 78)

    cols = [m for m in (["cv_auc"] + METRICS) if m in df.columns]
    g = df.groupby("combo")[cols]
    table = (g.mean().round(4).astype(str) + " +/- " + g.std().round(4).astype(str))
    table["_sort"] = g.mean()["test_auc"]
    table = table.sort_values("_sort", ascending=False).drop(columns="_sort")
    print("\n--- every combination (sorted by mean test AUC) ---")
    print(table.to_string())
    table.to_csv(os.path.join(d, f"{tag}_combo_table.csv"))

    print("\n--- factor effects (mean over all other factors) ---")
    for factor in ["imputer", "scaler", "resampler"]:
        print(df.groupby(factor)[METRICS].mean().round(4).to_string(), "\n")

    if "best_model" in df.columns and tag != "screen":
        print("--- winning model counts ---")
        print(df.groupby("best_model").size().sort_values(ascending=False).head(8).to_string())

    base = args.base
    allp = []
    for m in ["test_auc", "test_acc", "test_bal_acc", "test_f1"]:
        piv = df.pivot_table(index="repeat", columns="combo", values=m)
        if base in piv.columns:
            allp.append(paired(piv, base, m))
    allp = [x for x in allp if len(x)]
    if allp:
        res = pd.concat(allp)
        res.to_csv(os.path.join(d, f"{tag}_paired_vs_base.csv"), index=False)
        print(f"\n--- paired difference vs base '{base}' (positive = better than base) ---")
        print("(showing the 8 largest mean test-AUC and test-accuracy differences)")
        for m in ["test_auc", "test_acc"]:
            sub = res[res["metric"] == m].sort_values("mean_diff_vs_base", ascending=False).head(8)
            print(sub.round(4).to_string(index=False), "\n")

    # figure: AUC and balanced accuracy for up to 12 best combos (+ base)
    best = df.groupby("combo")["test_auc"].median().sort_values(ascending=False).index[:12].tolist()
    if base in df["combo"].values and base not in best:
        best.append(base)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, m in zip(axes, ["test_auc", "test_bal_acc"]):
        ax.boxplot([df[df["combo"] == k][m] for k in best], showmeans=True)
        ax.set_xticks(range(1, len(best) + 1))
        ax.set_xticklabels(best, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{label}: {m}")
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"{tag}_boxplot.png"), dpi=200)
    plt.close()

print(f"\nSaved tables and figures in {d}/")
