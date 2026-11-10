"""R1: describe the data and compare with numbers stated in the paper.
Read-only: no modelling, no splitting, no changes to the data.
Paper: Shehab, Darwish & Hassanien (2025), doi:10.1007/s10668-023-03916-4
"""
import argparse, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--target", default="Potability")
ap.add_argument("--out", default="results/r1_data_check.txt")
a = ap.parse_args()

df = pd.read_csv(a.csv)
if a.target not in df.columns:
    sys.exit(f"ERROR: target column '{a.target}' not found. Columns: {list(df.columns)}")

lines = []
def say(s=""):
    print(s); lines.append(str(s))

say(f"File: {a.csv}")
say(f"Shape (rows, cols): {df.shape}   | paper: 3276 rows, 9 features + target")
say(f"Columns: {list(df.columns)}")
say(f"Duplicate rows: {int(df.duplicated().sum())}")

vc = df[a.target].value_counts(dropna=False).sort_index()
say("\nClass counts:")
for k, v in vc.items():
    say(f"  class {k}: {v}")
say("  paper (Sect. 5.2): class 0 = 1998, class 1 = 1277")

paper_missing = {"Sulfate": 23.84, "ph": 14.99, "Trihalomethanes": 4.95}
say("\nMissing values (% of rows)   [paper Table 4 in brackets]")
miss = df.isna().mean() * 100
for c in df.columns:
    ref = f"[{paper_missing[c]}%]" if c in paper_missing else ""
    say(f"  {c:16s} {miss[c]:6.2f}%  {ref}")
say(f"Rows with at least one missing value: {int(df.isna().any(axis=1).sum())}")
say(f"Complete rows: {int(df.notna().all(axis=1).sum())}")

say("\nMin / max per column (paper Table 2 looks garbled, so only print ours)")
d = df.describe().T[["min", "max"]]
for c, r in d.iterrows():
    say(f"  {c:16s} {r['min']:.4f}  to  {r['max']:.4f}")

import os
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write("\n".join(lines) + "\n")
say(f"\nSaved report to {a.out}")
