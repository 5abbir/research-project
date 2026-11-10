"""
Phase 1c: summarise both tools. Run after both runners have finished:
  python phase1_compare.py
"""
import pandas as pd

pc = pd.read_csv("results/pycaret_results.csv")
ag = pd.read_csv("results/autogluon_results.csv")
df = pd.concat([pc, ag])

metrics = ["cv_auc", "test_auc", "test_f1", "test_bal_acc", "test_acc", "seconds"]
g = df.groupby("tool")[metrics]
summary = g.mean().round(4).astype(str) + " +/- " + g.std().round(4).astype(str)
print("\n=== Mean +/- std over seeds ===")
print(summary.T.to_string())

print("\n=== Which model won, per seed ===")
print(df.groupby(["tool", "best_model"]).size().to_string())

print("\n=== Top-5 models by mean CV AUC (PyCaret) ===")
lb = pd.read_csv("results/pycaret_leaderboards.csv")
print(lb.groupby("Model")["AUC"].agg(["mean", "std"]).sort_values("mean", ascending=False).head(5).round(4))

print("\n=== Top-5 single models by mean val AUC (AutoGluon) ===")
lb = pd.read_csv("results/autogluon_leaderboards.csv")
lb = lb[~lb["model"].str.startswith("WeightedEnsemble")]
print(lb.groupby("model")["score_val"].agg(["mean", "std"]).sort_values("mean", ascending=False).head(5).round(4))

summary.T.to_csv("results/phase1_summary.csv")
print("\nSaved results/phase1_summary.csv")
