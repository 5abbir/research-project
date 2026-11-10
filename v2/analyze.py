"""Summaries + paired statistics. Run in any env that has pandas/scipy/matplotlib:
  python analyze.py --name water
Statistics: Nadeau-Bengio corrected resampled t-test (repeated random splits overlap, so the
plain t-test is too optimistic), bootstrap 95% CI of the mean difference, Holm correction.
"""
import argparse
import itertools
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
name = p.parse_args().name
d = c.out_dir(name)
df = pd.read_csv(os.path.join(d, c.RAW))
df["method"] = np.where(df["condition"] == "-", df["tool"], df["tool"] + "|" + df["condition"])

metrics = ["cv_auc", "test_auc", "test_f1", "test_bal_acc", "test_mcc",
           "test_logloss", "test_brier", "seconds"]
g = df.groupby("method")[metrics]
summary = (g.mean().round(4).astype(str) + " +/- " + g.std().round(4).astype(str))
summary["n_repeats"] = df.groupby("method")["repeat"].nunique()
print("\n=== Mean +/- std over repeats ===")
print(summary.to_string())
summary.to_csv(os.path.join(d, "summary_table.csv"))

print("\n=== Winning model per repeat ===")
wins = df[df["tool"].isin(["pycaret", "autogluon"])].groupby(["method", "best_model"]).size()
print(wins.to_string())

# ---------------- paired tests on test AUC ----------------
piv = df.pivot_table(index="repeat", columns="method", values="test_auc")
ratio = c.TEST_SIZE / (1 - c.TEST_SIZE)  # n_test / n_train
rng = np.random.default_rng(0)


def compare(a, b):
    x = (piv[a] - piv[b]).dropna().values
    J = len(x)
    if J < 3 or x.std(ddof=1) == 0:
        return None
    t = x.mean() / np.sqrt((1 / J + ratio) * x.var(ddof=1))
    pval = 2 * stats.t.sf(abs(t), J - 1)
    boots = [rng.choice(x, J).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"A": a, "B": b, "mean_diff_AUC": x.mean(), "ci95_low": lo, "ci95_high": hi,
            "NB_t": t, "p_raw": pval, "n": J}


pairs = [("pycaret|A_all", "autogluon|A_all"), ("pycaret|B_trees", "autogluon|B_trees")]
for m in piv.columns:
    if m.startswith(("pycaret", "autogluon")):
        pairs.append((m, "baseline_rf_default"))
        pairs.append((m, "baseline_logreg"))
rows = [r for a, b in pairs if a in piv and b in piv and (r := compare(a, b))]
if rows:
    res = pd.DataFrame(rows)
    order = res["p_raw"].argsort().values  # Holm step-down
    m_ = len(res)
    adj = np.empty(m_)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m_ - rank) * res["p_raw"].iloc[i])
        adj[i] = min(1.0, running)
    res["p_holm"] = adj
    print("\n=== Paired comparisons on test AUC (positive diff = A better) ===")
    print(res.round(4).to_string(index=False))
    res.to_csv(os.path.join(d, "paired_tests.csv"), index=False)

# ---------------- figure ----------------
order = piv.median().sort_values(ascending=False).index
plt.figure(figsize=(9, 4.5))
plt.boxplot([piv[m].dropna() for m in order], showmeans=True)
plt.xticks(range(1, len(order) + 1), order, rotation=35, ha="right")
plt.ylabel("Test ROC-AUC")
plt.title(f"{name}: test AUC over {piv.shape[0]} repeated splits")
plt.tight_layout()
plt.savefig(os.path.join(d, "auc_boxplot.png"), dpi=200)
print(f"\nSaved tables and figure in {d}/")
