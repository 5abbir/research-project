#!/usr/bin/env python
"""
r2_s1_report.py  --  REPRO Scenario 1: ONE labelled side-by-side table from the SAVED result files.

Reads only CSV/JSON/TXT files written by r2_s1_sklearn_models.py, r2_s1_pycaret.py, r2_s1_autogluon.py.
Fits nothing, scores nothing, reads no raw data. Refuses smoke outputs.

Label for everything: "reproduction results, not validated performance".
The paper's Table 5 is shown in a SEPARATE reference block and is never merged with our numbers.

Outputs (in --out): s1_results_table.csv, s1_results_table.md
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

BASELINE_NOTE = "majority-class baseline on the test rows"
NAN_HANDLING = {
    "sklearn": "model's own NaN support (XGBoost, RandomForest native); AdaBoost rejects NaN",
    "pycaret": "PyCaret imputation_type=None; each model's own NaN support",
    "autogluon": "AutoGluon internal, model-specific handling (default preprocessing untouched)",
}


def need(path):
    if not os.path.exists(path):
        sys.exit(f"STOP: missing file {path}. Run the corresponding S1 script in full mode first.")
    return path


def read_json(path):
    with open(need(path)) as f:
        return json.load(f)


def pct(x):
    return float("nan") if pd.isna(x) else round(100 * float(x), 2)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("" if (isinstance(v, float) and math.isnan(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results folder holding the full-mode S1 outputs (e.g. ../results/s1)")
    args = ap.parse_args()
    d = args.out
    j = lambda f: os.path.join(d, f)  # noqa: E731

    # ---------------- load + guards
    sk_info, pc_info, ag_info = (read_json(j("s1_run_info.json")), read_json(j("s1_pycaret_run_info.json")),
                                 read_json(j("s1_autogluon_run_info.json")))
    for name, info in (("sklearn", sk_info), ("pycaret", pc_info), ("autogluon", ag_info)):
        if info.get("mode") != "full":
            sys.exit(f"STOP: {name} run_info mode is '{info.get('mode')}', not 'full'. Smoke results are not reportable.")
    sk = pd.read_csv(need(j("s1_sklearn_metrics.csv")))
    sk_stat = pd.read_csv(need(j("s1_model_status.csv")))
    sk_conf = pd.read_csv(need(j("s1_confusion.csv")))
    pc = pd.read_csv(need(j("s1_pycaret_test_metrics.csv")))
    pc_stat = pd.read_csv(need(j("s1_pycaret_model_status.csv")))
    pc_conf = pd.read_csv(need(j("s1_pycaret_confusion.csv")))
    ag = pd.read_csv(need(j("s1_autogluon_metrics.csv")))
    ag_conf = pd.read_csv(need(j("s1_autogluon_confusion.csv")))
    paper = pd.read_csv(need(j("s1_paper_reference.csv")))
    sk_pred = pd.read_csv(need(j("s1_predictions.csv")))
    pc_pred = pd.read_csv(need(j("s1_pycaret_predictions.csv")))
    ag_pred = pd.read_csv(need(j("s1_autogluon_predictions.csv")))
    split_txt = open(need(j("s1_split_report.txt"))).read()
    csv_line = [ln for ln in split_txt.splitlines() if ln.startswith("csv:")]
    nan_line = [ln for ln in split_txt.splitlines() if "rows containing >=1 NaN" in ln]

    # ---------------- consistency checks
    checks = []
    rows_sk, rows_pc, rows_ag = (set(sk_pred.row_id), set(pc_pred.row_id), set(ag_pred.row_id))
    same_rows = rows_sk == rows_pc == rows_ag
    checks.append(("same test rows in all three tools", same_rows, f"{len(rows_sk)} / {len(rows_pc)} / {len(rows_ag)} rows"))
    for tool, met, conf, key in (("sklearn", sk[sk.result_label != "Not runnable"], sk_conf, "model"),
                                 ("pycaret", pc, pc_conf, "model_id"), ("autogluon", ag, ag_conf, "model")):
        m = met.merge(conf, on=key, suffixes=("", "_c"))
        acc_c = (m["TN"] + m["TP"]) / (m["TN"] + m["FP"] + m["FN"] + m["TP"])
        ok = bool(np.allclose(acc_c, m["accuracy"], atol=1e-9)) and len(m) == len(met)
        checks.append((f"{tool}: accuracy equals accuracy recomputed from confusion counts", ok, f"{len(m)} models"))
    n_eval = int(sk.dropna(subset=["n_eval"])["n_eval"].iloc[0])
    base = float(sk.dropna(subset=["baseline_acc_majority"])["baseline_acc_majority"].iloc[0])
    se_acc = math.sqrt(base * (1 - base) / n_eval)  # SE of an accuracy near the baseline; rough guide only

    # ---------------- unified rows
    out = []

    def add(tool, model, role, label, status, r=None, cause="", val_auc=np.nan, val_basis="", flags="", version=""):
        row = {"tool": tool, "model": model, "role": role, "result_label": label, "status": status,
               "nan_handling": NAN_HANDLING[tool], "library_versions": version}
        if r is not None:
            row.update({"n_test": int(r["n_eval"]), "accuracy_pct": pct(r["accuracy"]),
                        "baseline_pct": pct(r["baseline_acc_majority"]),
                        "above_baseline_pts": round(100 * (r["accuracy"] - r["baseline_acc_majority"]), 2),
                        "precision_pct": pct(r["precision"]), "recall_pct": pct(r["recall_sensitivity"]),
                        "specificity_pct": pct(r["specificity"]), "f1_pct": pct(r["f1"]),
                        "roc_auc": round(float(r["roc_auc"]), 4), "balanced_acc": round(float(r["balanced_accuracy"]), 4),
                        "mcc": round(float(r["mcc"]), 4), "log_loss": round(float(r["log_loss"]), 4),
                        "brier": round(float(r["brier"]), 4)})
        row.update({"validation_auc": val_auc, "validation_basis": val_basis, "best_flags_validation_based": flags,
                    "cause_if_not_runnable": cause})
        out.append(row)

    v_sk = f"sklearn {sk_info['versions']['sklearn']}, xgboost {sk_info['versions']['xgboost']}"
    for _, r in sk.iterrows():
        if r["result_label"] == "Not runnable":
            st = sk_stat[sk_stat.model == r["model"]].iloc[0]
            add("sklearn", r["model"], r["role"], "Not runnable", "not runnable",
                cause=f"{st['error_type']}: {str(st['error_message'])[:60]}...", version=v_sk)
        else:
            add("sklearn", r["model"], r["role"], r["result_label"], "runnable", r, version=v_sk)

    v_pc = f"pycaret {pc_info['versions']['pycaret']}, sklearn {pc_info['versions']['sklearn']}"
    pc_ok = set(pc["model_id"])
    for _, r in pc.sort_values(["role", "model_id"]).iterrows():
        fl = ", ".join(n for n, c in (("CV-best overall", r.get("cv_best_overall")), ("CV-best primary", r.get("cv_best_primary"))) if c is True or str(c) == "True")
        add("pycaret", r["model_id"], r["role"], r["result_label"], "runnable", r, val_auc=round(float(r["cv_auc"]), 4),
            val_basis="5-fold CV on train rows", flags=fl, version=v_pc)
    for _, r in pc_stat[~pc_stat.model_id.isin(pc_ok)].sort_values(["role", "model_id"]).iterrows():
        msg = str(r["error_message"])
        cause = "Input X contains NaN (NaN not accepted)" if "contains NaN" in msg else f"UNVERIFIED cause: {msg[:80]}"
        add("pycaret", r["model_id"], r["role"], "Not runnable", str(r["status"]), cause=cause, version=v_pc)

    v_ag = f"autogluon {ag_info['versions']['autogluon.tabular']}"
    for _, r in ag.sort_values("score_val", ascending=False).iterrows():
        fl = ", ".join(n for n, c in (("best overall by validation", r["best_overall_by_val"]),
                                      ("best non-ensemble by validation", r["best_nonensemble_by_val"])) if c is True or str(c) == "True")
        add("autogluon", r["model"], "AutoGluon leaderboard", r["result_label"], "runnable", r,
            val_auc=round(float(r["score_val"]), 4), val_basis="out-of-fold (train rows only)", flags=fl, version=v_ag)

    tab = pd.DataFrame(out)
    tab.to_csv(j("s1_results_table.csv"), index=False)

    # ---------------- markdown
    show = ["model", "role", "result_label", "accuracy_pct", "above_baseline_pts", "precision_pct", "recall_pct",
            "specificity_pct", "f1_pct", "roc_auc", "balanced_acc", "mcc", "validation_auc", "best_flags_validation_based"]
    md = []
    md.append("# Scenario 1 results: raw CSV (no filling, no dropping, no scaling, no resampling, no feature selection)")
    md.append("**Reproduction results, not validated performance.** Single stratified 80/20 split, seed 42, library defaults "
              "(AutoGluon: project settings carried over from v2). Every runnable model was scored once on the test rows; "
              "no model was selected, tuned or rejected on test performance. Do NOT compare these numbers with Scenario 2/3 "
              "(different missing-value treatment).\n")
    md.append(f"- Data: {csv_line[0] if csv_line else 'csv line not found'}")
    md.append(f"- Test rows: {n_eval}; {nan_line[-1].strip() if nan_line else ''}")
    md.append(f"- Majority-class baseline on test rows: **{round(100 * base, 2)}%** ({BASELINE_NOTE}).")
    md.append(f"- Rough noise guide: standard error of one accuracy near the baseline = about {round(100 * se_acc, 1)} points "
              "(one test split; differences of 1-2 points are not meaningful).\n")
    md.append("## Consistency checks (computed from the saved files)")
    for name, ok, detail in checks:
        md.append(f"- {'PASS' if ok else 'FAIL'}: {name} ({detail})")
    for tool, title, extra in (
            ("sklearn", "A. sklearn models (env_main)", "AdaBoost is reported as: Not runnable on raw CSV with NaN values; "
             "missing-value handling unspecified in paper. Our status: NOT REPRODUCIBLE for AdaBoost under the raw-CSV protocol."),
            ("pycaret", "B. PyCaret library models (env_pycaret)", "Primary = xgboost, rf, ada (the paper's three). Exploratory = "
             "every other runnable model; they were NOT pre-specified. 'CV-best' flags use train-only 5-fold CV AUC, never test."),
            ("autogluon", "C. AutoGluon leaderboard (env_main)", "Label: Raw CSV / NaN-native (tool-internal handling), project "
             "settings carried over from v2 (medium_quality, 5 bag folds, 0 stack levels, 300 s, 8 CPUs; not AutoGluon defaults). "
             "Results reflect AutoGluon's internal handling, not a common imputation. Sorted by validation score, not test score.")):
        sub = tab[tab.tool == tool]
        md.append(f"\n## {title}")
        md.append(f"Library versions: {sub['library_versions'].iloc[0]}. NaN handling: {NAN_HANDLING[tool]}.")
        md.append(extra + "\n")
        run = sub[sub.status == "runnable"][show]
        md.append(md_table(run))
        nr = sub[sub.status != "runnable"]
        if len(nr):
            md.append("\n**Not runnable:**\n")
            md.append(md_table(nr[["model", "role", "result_label", "cause_if_not_runnable"]]))
    md.append("\n## D. Paper's Scenario 1 (Table 5) - REFERENCE ONLY, not our results, not merged")
    md.append("Source: Shehab et al. 2025, Table 5 (original data). The paper's split, seed, hyper-parameters and library "
              "versions are unknown, and its missing-value handling for this scenario is unspecified.\n")
    md.append(md_table(paper[["model", "precision_pct", "sensitivity_pct", "specificity_pct", "f1_pct", "accuracy_pct", "note"]]))
    md.append("\n## E. Limits and honest reading")
    md.append("- One split of 656 test rows: accuracy standard error about 1.9 points, AUC about 0.02; ranking models on it would be over-reading.")
    md.append("- sklearn/xgboost versions differ between tools (see versions above); PyCaret's rf and the env_main RandomForest are different builds.")
    md.append("- AutoGluon's validation (out-of-fold) AUC is optimistic compared with test AUC; AutoGluon's bagged models are not the same configuration as the default sklearn models.")
    md.append("- The paper's Scenario 1 numbers are neither reproduced nor refuted by this single run; AdaBoost cannot be reproduced under the raw-CSV protocol.")
    md.append("- Scenario 1 uses a different missing-value treatment from Scenarios 2/3, so it must not be compared with them.")
    with open(j("s1_results_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nWrote {j('s1_results_table.csv')} and {j('s1_results_table.md')}")
    if not all(ok for _, ok, _ in checks):
        print("\nWARNING: at least one consistency check FAILED. Do not report these tables; tell the user.")
        sys.exit(4)


if __name__ == "__main__":
    main()
