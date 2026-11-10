"""
Phase 0: inspect the CSV, flag problems, and make ONE fixed train/test split.

Usage examples:
  python phase0_prepare.py --csv data/raw.csv --target label --task classification
  python phase0_prepare.py --csv data/raw.csv --target price --task regression --drop id,customer_id

Outputs (in ./data): train.csv, test.csv, phase0_report.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True, help="name of the target column")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.add_argument("--drop", default="", help="comma-separated columns to drop (IDs, etc.)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    df = pd.read_csv(a.csv)
    if a.target not in df.columns:
        sys.exit(f"Target '{a.target}' not found. Columns: {list(df.columns)}")

    log("=== 1. BASIC INFO ===")
    log(f"Shape (rows, cols): {df.shape}")
    log(f"Column types:\n{df.dtypes.value_counts()}")
    log(f"\nFirst rows:\n{df.head()}")

    # ---- drop user-specified columns (IDs etc.)
    drop_cols = [c.strip() for c in a.drop.split(",") if c.strip()]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log(f"\nDropped columns: {drop_cols}")

    log("\n=== 2. MISSING VALUES ===")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    log(miss.apply(lambda x: f"{x:.1%}").to_string() if len(miss) else "None")

    log("\n=== 3. DUPLICATES ===")
    n_dup = int(df.duplicated().sum())
    log(f"Exact duplicate rows: {n_dup}")
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        log("-> removed (duplicates across train/test would leak information)")

    # rows with missing target are unusable
    n_before = len(df)
    df = df.dropna(subset=[a.target]).reset_index(drop=True)
    if len(df) != n_before:
        log(f"Removed {n_before - len(df)} rows with missing target")

    features = [c for c in df.columns if c != a.target]

    log("\n=== 4. TARGET ===")
    if a.task == "classification":
        vc = df[a.target].value_counts()
        log(f"Class counts:\n{vc}")
        log(f"Class proportions:\n{(vc / len(df)).round(3)}")
        if vc.min() / vc.max() < 0.2:
            log("WARNING: imbalanced classes -> use F1 / AUC / balanced accuracy, not plain accuracy.")
        if vc.min() < 10:
            log("WARNING: a class has <10 samples; consider merging or removing it.")
    else:
        log(df[a.target].describe().to_string())

    log("\n=== 5. SUSPICIOUS COLUMNS (possible leakage / useless) ===")
    flagged = False
    for c in features:
        nun = df[c].nunique(dropna=True)
        if nun <= 1:
            log(f"- '{c}': constant column (useless)")
            flagged = True
        elif nun == len(df):
            log(f"- '{c}': all values unique (ID-like? consider dropping)")
            flagged = True
        elif df[c].dtype == object and nun > 0.5 * len(df):
            log(f"- '{c}': high-cardinality text/category ({nun} unique)")
            flagged = True
    # numeric features that almost perfectly predict the target
    y_num = df[a.target]
    if a.task == "classification":
        y_num = y_num.astype("category").cat.codes
    num = df[features].select_dtypes(include=np.number)
    for c in num.columns:
        try:
            r = abs(np.corrcoef(num[c].fillna(num[c].median()), y_num)[0, 1])
        except Exception:
            continue
        if r > 0.95:
            log(f"- '{c}': |corr| with target = {r:.3f} -> LEAKAGE SUSPECT, ask what this column means")
            flagged = True
    if not flagged:
        log("None found by automatic checks (still check column meanings manually!)")

    log("\n=== 6. TRAIN / TEST SPLIT ===")
    strat = None
    if a.task == "classification" and df[a.target].value_counts().min() >= 2:
        strat = df[a.target]
    train, test = train_test_split(
        df, test_size=a.test_size, random_state=a.seed, stratify=strat
    )
    train_path = os.path.join(a.out, "train.csv")
    test_path = os.path.join(a.out, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    log(f"Seed={a.seed}, test_size={a.test_size}, stratified={strat is not None}")
    log(f"Train: {train.shape}  -> {train_path}")
    log(f"Test:  {test.shape}  -> {test_path}")
    if a.task == "classification":
        log(f"Train class proportions:\n{train[a.target].value_counts(normalize=True).round(3)}")
        log(f"Test class proportions:\n{test[a.target].value_counts(normalize=True).round(3)}")

    with open(os.path.join(a.out, "phase0_report.txt"), "w") as f:
        f.write("\n".join(lines))
    log("\nDONE. Report saved. From now on, NEVER touch test.csv until the final evaluation.")


if __name__ == "__main__":
    main()
