"""LESSON 0 - Look at the missing values BEFORE choosing how to fix them.
Uses only the TRAINING file, so the test set stays untouched.
Run:  python lesson0_look_at_missing.py --train ../data_cls/train.csv --target Potability
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---- 1. read the file and separate the inputs (X) from the answer (y)
p = argparse.ArgumentParser()
p.add_argument("--train", default="../data_cls/train.csv")
p.add_argument("--target", default="Potability")
a = p.parse_args()
df = pd.read_csv(a.train)
X, y = df.drop(columns=[a.target]), df[a.target]
miss = X.isna()                      # True where a cell is blank

# ---- 2. how much is missing, and where?
print("=== 1. Missing values per column ===")
summary = pd.DataFrame({"n_missing": miss.sum(), "percent": (miss.mean() * 100).round(1)})
print(summary[summary.n_missing > 0].sort_values("percent", ascending=False).to_string())
rows_any = int(miss.any(axis=1).sum())
print(f"\nRows with at least one blank: {rows_any} of {len(df)} ({rows_any / len(df):.0%})")
print(f"Complete rows (what 'drop rows' would keep): {len(df) - rows_any}")

# ---- 3. is 'being blank' linked to the answer?  (a hint, not a proof)
print("\n=== 2. Does a blank relate to the answer? ===")
print("column            rate(answer=1) when blank | when present | chi-square p-value")
for col in miss.columns[miss.any()]:
    table = pd.crosstab(miss[col], y)
    p_val = chi2_contingency(table)[1] if table.shape == (2, 2) else np.nan
    r_blank, r_present = y[miss[col]].mean(), y[~miss[col]].mean()
    print(f"{col:17s} {r_blank:.3f}                     | {r_present:.3f}        | {p_val:.3f}")
print("(small p-value, e.g. < 0.05, would mean 'blank' carries information about the answer)")

# ---- 4. do blanks happen together?
print("\n=== 3. Do blanks in different columns happen together? (correlation of blank flags) ===")
cols = miss.columns[miss.any()]
print(miss[cols].astype(int).corr().round(3).to_string())

# ---- 5. shape of the numbers (mean vs median question)
print("\n=== 4. Skewness of columns that have blanks (near 0 = symmetric) ===")
print(X[cols].skew().round(2).to_string())
