"""Like-for-like AutoML comparison on the SAME 10 splits (seeds 0-9):
   AutoGluon from v2 (5-fold bagging, all training rows)  vs  PyCaret from v3 (fixed, all training rows, mean fill, no fix).
Run from ~/research/v3:  python compare_tools.py
"""
import numpy as np
import pandas as pd
from scipy import stats

ag = pd.read_csv("../v2/results/water/phase1_raw.csv")
ag = ag[(ag["tool"] == "autogluon") & (ag["condition"] == "A_all")].set_index("repeat")["test_auc"]
pc = pd.read_csv("results/water3/pycaret_raw.csv")
pc = pc[pc["combo"] == "mean|none|none"].set_index("repeat")["test_auc"]

d = (pc - ag).dropna()                      # positive = PyCaret better
J, ratio = len(d), 0.2 / 0.8                # 10 splits; test/train size ratio of an 80/20 split
t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
p = 2 * stats.t.sf(abs(t), J - 1)
rng = np.random.default_rng(0)
lo, hi = np.percentile([rng.choice(d.values, J).mean() for _ in range(5000)], [2.5, 97.5])
print(f"splits compared: {J}")
print(f"AutoGluon (v2) test AUC: {ag.loc[d.index].mean():.4f} +/- {ag.loc[d.index].std():.4f}")
print(f"PyCaret   (v3) test AUC: {pc.loc[d.index].mean():.4f} +/- {pc.loc[d.index].std():.4f}")
print(f"mean difference (PyCaret - AutoGluon): {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"corrected resampled t-test p-value: {p:.3f}")
print("Reading: if the CI includes 0 and p > 0.05, the two tools are statistically tied.")
