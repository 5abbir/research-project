#!/usr/bin/env python
"""
r4_frs_select.py  --  Scenario 3 (S3), Track L (LEAK-FREE): fuzzy-rough feature selection (QuickReduct) on TRAIN rows only.

LABEL: "Section 4.2 correlation-based imputation + SMOTE (train rows only) + fuzzy-rough selection (our implementation)".
Reproduction results, not validated performance.  See S3_PROTOCOL.md (decisions C1-C4, all "accept recommended").

Input : results/s2/s2_smote_train_L.csv (filled + SMOTE train rows; columns: 9 features, Potability, row_id, is_synthetic).
        THE TEST FILE IS NOT AN INPUT AND IS NEVER OPENED ("test rows used: 0").
Method (Jensen-Shen style fuzzy-rough dependency, relation-based Eqs. 4,5,9,10 of the paper; our choices, the paper leaves them UNSPECIFIED):
  * per-attribute similarity  R_a(x,y) = 1 - |a(x)-a(y)| / (max_a - min_a)   (range from the rows given = TRAIN rows)
  * subset similarity         R_P(x,y) = min over a in P of R_a(x,y)          (min t-norm)
  * implicator                Kleene-Dienes I(a,b) = max(1-a, b)              (Lukasiewicz gives identical numbers for crisp classes; self-test checks both)
  * lower approximation of class c at x = min_y I(R_P(x,y), [y in c]);  positive region mu_POS(x) = max_c lower_c(x);  gamma_P = mean_x mu_POS(x)
  * closed form used for speed (verified against the brute force in the built-in self-test, which must pass or the script stops):
        mu_POS(x) = 1 - max over y of the OTHER class of R_P(x,y)
  * search: QuickReduct, greedy forward; add the feature giving the highest gamma; ties -> earlier column; the full order is recorded.
    Reduct = shortest prefix of the order whose gamma equals gamma(all 9 features) (tolerance 1e-12).  Strict rule, no invented threshold.
  * ALSO reported: the first K features of the order (K=7, the size of the paper's list; taken from the paper, not discovered) -- labelled as such.
  * Paper list [6,7,3,2,1,5,8] under two index maps (M1 = paper Table 2 order, primary [P]; M2 = raw CSV order, sensitivity): their gamma and overlap with ours.
  * Stability check: the same search on the ORIGINAL filled rows only (is_synthetic == 0 in the same file).
Features never include Potability, row_id or is_synthetic.

Modes: --smoke : stratified subsample (--smoke-n, default 600) of the given file; output -> <out>/frs_smoke/ ; NOT reportable.
       default : the full file; output -> <out>.
Exit codes: 0 ok; 2 input problem (STOP message); 4 self-test or check failed.
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

# Paper Sect. 5.3 list (indices) and the two index maps (decision C1). Kept list is in the paper's order.
PAPER_LIST = [6, 7, 3, 2, 1, 5, 8]
MAPS = {
    "M1 (paper Table 2 order, primary [P])": ["Hardness", "Solids", "Chloramines", "Conductivity", "Organic_carbon",
                                              "Turbidity", "Sulfate", "Trihalomethanes", "ph"],
    "M2 (raw CSV column order, sensitivity)": ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity",
                                               "Organic_carbon", "Trihalomethanes", "Turbidity"],
}
TIE_TOL = 1e-12


def stop(msg, code=2):
    print("STOP: " + msg, file=sys.stderr)
    sys.exit(code)


# ----------------------------------------------------------------------------- fuzzy-rough core
class FRS:
    """Cross-class similarity matrices per attribute (only pairs from different classes are needed)."""

    def __init__(self, X, y):
        self.feats = list(X.columns)
        Xv = X.to_numpy(dtype=np.float64)
        yv = np.asarray(y)
        self.i0 = np.where(yv == 0)[0]
        self.i1 = np.where(yv == 1)[0]
        self.n = len(yv)
        self.S = {}
        for j, f in enumerate(self.feats):
            col = Xv[:, j]
            rng = col.max() - col.min()
            if rng <= 0:
                stop(f"feature '{f}' is constant on the rows given; similarity range is 0. Ask the user.")
            self.S[f] = 1.0 - np.abs(col[self.i0][:, None] - col[self.i1][None, :]) / rng  # (n0, n1)

    def _gamma_from_matrix(self, M):
        # mu_POS for class-0 rows = 1 - max over class-1 columns ; for class-1 rows = 1 - max over class-0 rows
        mu0 = 1.0 - M.max(axis=1)
        mu1 = 1.0 - M.max(axis=0)
        return (mu0.sum() + mu1.sum()) / self.n

    def gamma(self, subset):
        subset = list(subset)
        if not subset:
            return 0.0
        M = self.S[subset[0]]
        for f in subset[1:]:
            M = np.minimum(M, self.S[f])
        return float(self._gamma_from_matrix(M))

    def quickreduct(self):
        """Greedy forward selection; returns the full order with gamma after each addition."""
        remaining = list(self.feats)
        chosen, path = [], []
        M = None
        while remaining:
            best_f, best_g, best_M = None, -1.0, None
            for f in remaining:  # column order breaks ties (strictly greater needed to replace)
                Mc = self.S[f] if M is None else np.minimum(M, self.S[f])
                g = float(self._gamma_from_matrix(Mc))
                if g > best_g + TIE_TOL:
                    best_f, best_g, best_M = f, g, Mc
            chosen.append(best_f)
            remaining.remove(best_f)
            M = best_M
            path.append((best_f, best_g))
        return chosen, path


def brute_force_gamma(Xv, yv, cols, implicator):
    """Direct evaluation of the definitions on ALL pairs (n x n). Only for the self-test (small n)."""
    n = len(yv)
    R = np.ones((n, n))
    for j in cols:
        rng = Xv[:, j].max() - Xv[:, j].min()
        R = np.minimum(R, 1.0 - np.abs(Xv[:, j][:, None] - Xv[:, j][None, :]) / rng)
    mu_pos = np.zeros(n)
    for c in (0, 1):
        ind = (yv == c).astype(float)[None, :]                    # A(y) crisp
        lower = implicator(R, ind).min(axis=1)                    # min_y I(R(x,y), A(y))
        mu_pos = np.maximum(mu_pos, lower)                        # sup over decision classes
    return mu_pos.sum() / n


def self_test(X, y, seed, n_sub=80, n_subsets=6):
    rng = np.random.RandomState(seed)
    parts = []
    for lab in (0, 1):
        idx = np.where(np.asarray(y) == lab)[0]
        parts.append(rng.choice(idx, size=min(n_sub // 2, len(idx)), replace=False))
    sel = np.sort(np.concatenate(parts))
    Xs, ys = X.iloc[sel].reset_index(drop=True), np.asarray(y)[sel]
    frs = FRS(Xs, ys)
    Xv = Xs.to_numpy(dtype=np.float64)
    kd = lambda a, b: np.maximum(1.0 - a, b)
    lk = lambda a, b: np.minimum(1.0, 1.0 - a + b)
    worst, rows = 0.0, []
    for k in range(n_subsets):
        size = rng.randint(1, len(Xs.columns) + 1)
        cols = sorted(rng.choice(len(Xs.columns), size=size, replace=False))
        names = [Xs.columns[j] for j in cols]
        g_fast = frs.gamma(names)
        d1 = abs(g_fast - brute_force_gamma(Xv, ys, cols, kd))
        d2 = abs(g_fast - brute_force_gamma(Xv, ys, cols, lk))
        worst = max(worst, d1, d2)
        rows.append(f"  subset of {size}: closed form {g_fast:.10f}; max |diff| to brute force (Kleene-Dienes, Lukasiewicz) = {max(d1, d2):.2e}")
    # monotonicity along the greedy path on the subsample
    _, path = frs.quickreduct()
    gs = [g for _, g in path]
    mono = all(gs[i + 1] >= gs[i] - 1e-12 for i in range(len(gs) - 1))
    return worst < 1e-10, mono, worst, rows, len(sel)


def run_search(X, y, k_paper):
    frs = FRS(X, y)
    g_all = frs.gamma(frs.feats)
    singles = {f: frs.gamma([f]) for f in frs.feats}
    order, path = frs.quickreduct()
    gs = [g for _, g in path]
    reduct_len = next(i + 1 for i, g in enumerate(gs) if g >= g_all - TIE_TOL)
    return {"frs": frs, "g_all": g_all, "singles": singles, "order": order, "gammas": gs,
            "reduct": order[:reduct_len], "prefix_k": order[:k_paper]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", required=True, help="s2_smote_train_L.csv (train rows only)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-paper", type=int, default=7, help="size of the paper's list (taken from the paper, not discovered)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=600)
    ap.add_argument("--compare-with", default=None, help="previous s3_frs_order_L.csv of THIS script and input (identical-file check)")
    args = ap.parse_args()

    if "test" in os.path.basename(args.train_csv).lower():
        stop("the input file name contains 'test'. This script reads TRAIN rows only. Nothing was run.")
    df = pd.read_csv(args.train_csv, float_precision="round_trip")
    for c in (args.target, "row_id", "is_synthetic"):
        if c not in df.columns:
            stop(f"column '{c}' not found in {args.train_csv}. Columns: {list(df.columns)}")
    feats = [c for c in df.columns if c not in (args.target, "row_id", "is_synthetic")]
    all_names = set(MAPS["M1 (paper Table 2 order, primary [P])"])
    if set(feats) != all_names or len(feats) != 9:
        stop(f"expected the 9 water-quality feature columns {sorted(all_names)}; found {feats}")
    if df[feats].isna().any().any():
        stop("NaN in the input. Run the fill and SMOTE scripts first.")
    if set(np.unique(df[args.target])) != {0, 1}:
        stop("target must contain exactly the labels 0 and 1")
    orig_in = df[df["is_synthetic"] == 0]
    if (orig_in["row_id"] < 0).any() or (df.loc[df["is_synthetic"] == 1, "row_id"] != -1).any():
        stop("row_id / is_synthetic are inconsistent (originals need row_id >= 0, synthetic -1)")

    mode = "smoke" if args.smoke else "full"
    out_dir = os.path.join(args.out, "frs_smoke") if args.smoke else args.out
    os.makedirs(out_dir, exist_ok=True)
    in_sha = sha256_file(args.train_csv)

    if args.smoke and len(df) > args.smoke_n:
        parts = []
        for lab in (0, 1):  # explicit loop; keeps the original/synthetic mix
            g = df[df[args.target] == lab]
            parts.append(g.sample(n=round(args.smoke_n * len(g) / len(df)), random_state=args.seed))
        df = pd.concat(parts).sort_index().reset_index(drop=True)
    X, y = df[feats], df[args.target]
    Xo, yo = df.loc[df["is_synthetic"] == 0, feats], df.loc[df["is_synthetic"] == 0, args.target]
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ---- built-in self-test of the closed form (train rows only, small subsample)
    ok_cf, ok_mono, worst, st_rows, n_st = self_test(X, y, args.seed)
    chk("closed form = brute-force definitions (Kleene-Dienes and Lukasiewicz), 6 random subsets", ok_cf, f"worst diff {worst:.2e} on {n_st} rows")
    chk("gamma non-decreasing along the greedy path (self-test subsample)", ok_mono)

    # ---- main search (all given train rows: originals + synthetic, as in paper Sect. 5.3) and stability search (originals only)
    main_res = run_search(X, y, args.k_paper)
    stab_res = run_search(Xo, yo, args.k_paper)
    gs = main_res["gammas"]
    chk("gamma non-decreasing along the greedy path (main search)", all(gs[i + 1] >= gs[i] - 1e-12 for i in range(len(gs) - 1)))
    chk("gamma after all 9 features equals gamma(all)", abs(gs[-1] - main_res["g_all"]) < 1e-9, f"{gs[-1]:.10f}")
    chk("order contains each feature exactly once", sorted(main_res["order"]) == sorted(feats))
    chk("Potability / row_id / is_synthetic never used as features", not ({args.target, "row_id", "is_synthetic"} & set(feats)))

    # ---- paper list under the two maps
    frs = main_res["frs"]
    maps_out = {}
    for name, cols in MAPS.items():
        kept = [cols[i] for i in PAPER_LIST]
        dropped = [cols[i] for i in range(9) if i not in PAPER_LIST]
        maps_out[name] = {"kept_in_paper_order": kept, "dropped": dropped, "gamma": frs.gamma(kept),
                          "overlap_with_our_reduct": len(set(kept) & set(main_res["reduct"])),
                          "overlap_with_our_prefix_k": len(set(kept) & set(main_res["prefix_k"]))}

    # ---- files
    order_df = pd.DataFrame({"step": range(1, 10), "feature_added": main_res["order"], "gamma_after": np.round(gs, 10),
                             "gamma_gain": np.round(np.diff([0.0] + gs), 10),
                             "in_reduct": [f in main_res["reduct"] for f in main_res["order"]],
                             "in_prefix_k": [f in main_res["prefix_k"] for f in main_res["order"]]})
    order_path = os.path.join(out_dir, "s3_frs_order_L.csv")
    order_df.to_csv(order_path, index=False)
    sel = {"track": "L (leak-free, primary), S3", "label": "Section 4.2 fill + SMOTE (train rows only) + fuzzy-rough selection (our implementation)",
           "mode": mode, "n_rows_used": int(len(X)), "n_original_rows": int(len(Xo)),
           "our_reduct_strict": main_res["reduct"], "our_prefix_k": main_res["prefix_k"], "k_paper": args.k_paper,
           "paper_list_indices": PAPER_LIST,
           "paper_list_M1": maps_out["M1 (paper Table 2 order, primary [P])"]["kept_in_paper_order"],
           "paper_list_M2": maps_out["M2 (raw CSV column order, sensitivity)"]["kept_in_paper_order"],
           "gamma_all": main_res["g_all"], "gamma_path": gs, "order": main_res["order"],
           "stability_order_original_rows": stab_res["order"], "stability_reduct_original_rows": stab_res["reduct"]}
    sel_path = os.path.join(out_dir, "s3_frs_selected_L.json")
    with open(sel_path, "w") as f:
        json.dump(sel, f, indent=2)
    order_sha = sha256_file(order_path)

    # ---- report
    L = [f"mode: {mode}" + ("  (NOT reportable)" if args.smoke else ""),
         f"input: {args.train_csv}  sha256: {in_sha}",
         f"rows used by the main search: {len(X)} (original {len(Xo)} + synthetic {len(X) - len(Xo)}); class counts "
         f"{ {int(k): int(v) for k, v in y.value_counts().sort_index().items()} }",
         "test rows used: 0 (the test file is not an input of this script)",
         "settings: linear similarity (train range), min t-norm, Kleene-Dienes, relation-based Eqs. 4-5/9/10, QuickReduct strict stop, ties -> earlier column",
         f"gamma(all 9 features) = {main_res['g_all']:.6f}", "",
         "single-feature gamma (information only): " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(main_res['singles'].items(), key=lambda kv: -kv[1])), "",
         "QuickReduct path (main search):"]
    for _, r in order_df.iterrows():
        L.append(f"  {int(r.step)}. +{r.feature_added:<16} gamma {r.gamma_after:.6f} (gain {r.gamma_gain:+.6f})"
                 + ("   <- reduct ends here" if r.feature_added == main_res["reduct"][-1] and r.in_reduct and len(main_res["reduct"]) == int(r.step) else ""))
    L += ["", f"OUR REDUCT (strict gamma = gamma(all)): {len(main_res['reduct'])} features: {main_res['reduct']}",
          f"first {args.k_paper} of our order (size taken from the paper, NOT discovered): {main_res['prefix_k']}",
          f"  gamma of that prefix = {gs[args.k_paper - 1]:.6f}", "", "Paper list [6,7,3,2,1,5,8] under the two index maps:"]
    for name, m in maps_out.items():
        L += [f"  {name}", f"    kept   : {m['kept_in_paper_order']}", f"    dropped: {m['dropped']}",
              f"    gamma = {m['gamma']:.6f}; overlap with our reduct {m['overlap_with_our_reduct']}/{len(main_res['reduct'])}, "
              f"with our first-{args.k_paper} {m['overlap_with_our_prefix_k']}/{args.k_paper}"]
    same_set = set(stab_res["prefix_k"]) == set(main_res["prefix_k"])
    L += ["", f"Stability check on the {len(Xo)} ORIGINAL filled rows only (no synthetic): order {stab_res['order']}",
          f"  reduct {stab_res['reduct']}; gamma(all) = {stab_res['g_all']:.6f}",
          f"  same reduct as main search: {stab_res['reduct'] == main_res['reduct']}; same first-{args.k_paper} SET: {same_set}; same full order: {stab_res['order'] == main_res['order']}",
          "", "Self-test of the closed form:"] + st_rows
    L += ["", "checks:"] + [f"{'PASS' if ok else 'FAIL'}: {n} ({d})" if d else f"{'PASS' if ok else 'FAIL'}: {n}" for n, ok, d in checks]
    L += ["", "Notes: similarity uses each feature's own range, so Solids' large scale does NOT dominate here (it does dominate SMOTE's distances). "
              "Synthetic rows lie between real minority rows and can change similarities (limitation, not fixed). "
              "The paper's FRS details are unspecified; this is our implementation, not a reproduction of theirs."]
    with open(os.path.join(out_dir, "s3_frs_report_L.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "s3_frs_run_info_L.json"), "w") as f:
        json.dump({"mode": mode, "timestamp": datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv), "seed": args.seed,
                   "input_sha256": in_sha, "order_csv_sha256": order_sha, "test_rows_used": 0,
                   "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                                "sklearn": sklearn.__version__, "platform": platform.platform()}}, f, indent=2)

    print(f"=== S3-1 fuzzy-rough selection (Track L) | mode={mode} | out={out_dir} ===")
    print("\n".join(L))
    if args.compare_with:
        if os.path.exists(args.compare_with):
            same = sha256_file(args.compare_with) == order_sha
            print(f"\nRun comparison vs {args.compare_with} (the first run of this same script, input and settings): "
                  f"{'EXACT match (identical order/gamma file)' if same else 'DIFFERENT file'}")
        else:
            print(f"\nRun comparison skipped: {args.compare_with} not found")
    if args.smoke:
        print("\nSMOKE: not reportable. Test rows used: 0.")
    print(f"\nFiles written to {out_dir}: s3_frs_order_L.csv, s3_frs_selected_L.json, s3_frs_report_L.txt, s3_frs_run_info_L.json")
    if not all(ok for _, ok, _ in checks):
        print("\nWARNING: at least one check FAILED. Do not use these outputs; tell the user.")
        sys.exit(4)


if __name__ == "__main__":
    main()
