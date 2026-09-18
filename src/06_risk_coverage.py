#!/usr/bin/env python3
"""Step 6: risk-coverage / selective-prediction audit on top of Step-3's LOCO
genome-wide predictions.

Ties two things from memory/plan.md together instead of shipping a bare
accuracy number:

  1. "Dry-run veto" money chart -- at a fixed false-accept budget (the risk
     budget), how much of the genome-wide candidate pool can be auto-accepted
     without orthogonal validation (Sanger/ddPCR/long-read)?  The delta
     between the `all` (with UMI molecule-linkage features) and `no_molecule`
     arms is the feature's dollar value, expressed as "% of validation saved"
     instead of a raw PR-AUC delta.
  2. Abstention-boundary hit-rate -- the tail this harness REFUSES to
     auto-accept (the abstained/low-confidence region) should be enriched in
     already-known-hard GIAB stratification regions (low-mappability/segdup,
     "all difficult"). If it isn't, the confidence score is not actually
     tracking genomic difficulty and the "harness catches the hard cases"
     story does not hold -- that null result is reported, not hidden.

Ground truth is GIAB (real orthogonal-consensus truth), never another
predictor -- see AI4S plan.md's "in-silico proxy ground truth" trap.

Candidate-pool trap (see plan.md "纠正 1"): computing risk-coverage on the
RAW candidate pool is dominated by trivially-rejectable noise (label
prevalence ~0.5%), so any confidence threshold "saves" close to 100% for free
and the number is meaningless. `--min-alt-reads`/`--min-vaf` restrict the
pool to candidates that would actually incur a validation cost.

Inputs
------
  --genomewide-dir   root with loco_<genome>/chr*_<arm>/test_predictions.tsv
                      (from 03_train_eval.py's genome-wide LOCO CV).
  --genomes          comma list, default hg002,hg004.
  --arms             comma list; first = primary (e.g. `all`), rest are
                      compared against it. Default all,no_molecule.
  --strat-bed        GIAB stratification BED(s) (.bed / .bed.gz, 0-based
                      half-open). Default: strat/lowmap_segdup.bed.gz and
                      strat/alldiff.bed.gz next to this repo's `out/`.
  --min-alt-reads / --min-vaf   candidate-pool prefilter (default 2 / 0.05).
  --threshold        decision threshold for pred=1 (default 0.5).
  --risk-budgets     comma list of false-accept budgets, default 0.01,0.02,0.05.
                     Saturates to 100% coverage fast once the achievable
                     error rate clears budget -- kept for the "what's my max
                     coverage under an error cap" question, but --abstain-fracs
                     is the more informative axis for comparing arms/policies.
  --abstain-fracs    comma list of FIXED validation-cost fractions (same cost
                     for every arm/policy compared), default
                     0.01,0.02,0.05,0.1,0.15,0.2,0.3. Preferred money-chart
                     axis -- see metrics_at_abstain_frac() docstring. Reports
                     FNR/FPR on the covered side plus FN/FP recovery (what %
                     of ALL missed-true-variants / wrongly-accepted-noise in
                     the whole pool get sent to validation).
  --vaf-bins         comma list of bin edges, default 0,0.05,0.1,0.2,0.35,0.5,1.0
  --out              output JSON report path (also writes <out>.curves.tsv,
                      <out>.vaf_strata.tsv, and <out>.abstain_fracs.tsv).

Usage
-----
  python 06_risk_coverage.py --genomewide-dir out/genomewide \
    --strat-bed strat/lowmap_segdup.bed.gz strat/alldiff.bed.gz \
    --out out/risk_coverage/report.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

PRED_COLS = ["chrom", "pos", "label", "vaf", "alt_reads", "dp", "p_true", "p_uncertainty"]


def load_bed(path):
    """0-based half-open BED -> {chrom: (sorted starts, sorted ends)}."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                      names=["chrom", "start", "end"], comment="#",
                      dtype={"start": np.int64, "end": np.int64})
    intervals = {}
    for chrom, g in df.groupby("chrom", sort=False):
        g = g.sort_values("start")
        intervals[chrom] = (g["start"].to_numpy(), g["end"].to_numpy())
    return intervals


def flag_in_regions(chrom_arr, pos1_arr, intervals):
    """1-based `pos` -> bool array: does the site fall in any BED interval."""
    pos0 = pos1_arr - 1
    out = np.zeros(len(pos1_arr), dtype=bool)
    idxframe = pd.DataFrame({"chrom": chrom_arr, "pos0": pos0})
    for chrom, g in idxframe.groupby("chrom", sort=False):
        se = intervals.get(chrom)
        if se is None:
            continue
        starts, ends = se
        p = g["pos0"].to_numpy()
        idx = np.searchsorted(starts, p, side="right") - 1
        valid = idx >= 0
        res = np.zeros(len(p), dtype=bool)
        if valid.any():
            res[valid] = p[valid] < ends[idx[valid]]
        out[g.index.to_numpy()] = res
    return out


def load_arm(genomewide_dir, genome, arm, chroms=None):
    pattern = os.path.join(genomewide_dir, f"loco_{genome}", f"chr*_{arm}", "test_predictions.tsv")
    files = sorted(glob.glob(pattern))
    if chroms:
        wanted = set(chroms)
        files = [f for f in files if os.path.basename(os.path.dirname(f)).split("_")[0] in wanted]
    if not files:
        raise FileNotFoundError(f"no test_predictions.tsv matched {pattern}")
    dfs = [pd.read_csv(f, sep="\t", usecols=PRED_COLS) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["chrom"] = df["chrom"].astype(str)
    df["label"] = df["label"].astype(np.int8)
    df["vaf"] = df["vaf"].astype(np.float32)
    df["p_true"] = df["p_true"].astype(np.float32)
    df["p_uncertainty"] = df["p_uncertainty"].astype(np.float32)
    df["alt_reads"] = df["alt_reads"].astype(np.int32)
    df["genome"] = genome
    return df


def apply_pool_filter(df, min_alt_reads, min_vaf):
    mask = (df["alt_reads"] >= min_alt_reads) & (df["vaf"] >= min_vaf)
    return df[mask].reset_index(drop=True)


def selective_curve(df, threshold, rank_by="confidence"):
    """Sort descending by the ranking score; return coverage[], risk[], and
    `order` (sorted position -> original row index) so callers can recover
    which original rows are covered/abstained at any coverage cut."""
    p = df["p_true"].to_numpy()
    label = df["label"].to_numpy()
    pred = (p >= threshold).astype(np.int8)
    correct = (pred == label)
    if rank_by == "confidence":
        score = np.abs(p - threshold)
    elif rank_by == "uncertainty":
        score = -df["p_uncertainty"].to_numpy()
    else:
        raise ValueError(rank_by)
    order = np.argsort(-score, kind="mergesort")  # stable: ties keep original order
    n = len(df)
    cum_correct = np.cumsum(correct[order])
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - cum_correct / np.arange(1, n + 1)
    return coverage, risk, order


def coverage_index_at_budget(risk, budget):
    """Largest index (0-based) whose cumulative risk is <= budget, or -1."""
    ok = np.where(risk <= budget)[0]
    return int(ok[-1]) if len(ok) else -1


def sanitize_nan(obj):
    """Recursively replace float NaN with None so json.dump emits valid JSON
    (NaN can happen when a coverage cut leaves an empty covered/abstained
    side, e.g. a risk budget so loose that 100% coverage is achievable)."""
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    return obj


def curve_auc(coverage, risk):
    return float(np.trapz(risk, coverage))


def grid_sample(coverage, risk, grid):
    idx = np.clip((grid * len(coverage)).astype(int) - 1, 0, len(coverage) - 1)
    return risk[idx]


def _rate(mask_num, mask_den):
    n = int(mask_den.sum())
    return (float(mask_num[mask_den].mean()), n) if n else (float("nan"), 0)


def metrics_at_abstain_frac(df, order, frac):
    """Fixed-cost money-chart row: at abstain fraction `frac` (= validation
    cost, independent of any risk-budget threshold), report FNR/FPR on the
    covered (auto-accepted) side plus FN/FP recovery -- what fraction of ALL
    missed-true-variants (resp. wrongly-accepted-noise) in the whole pool get
    caught by sending them to validation.

    Preferred over `--risk-budgets` for comparing arms/policies: a risk
    budget's covered fraction saturates to 100% as soon as the achievable
    error rate is already under budget (observed on real data -- budgets
    >=0.02 hit 100% coverage for both `all` and `no_molecule` on hg002/hg004,
    which is uninformative), while a fixed abstain fraction is comparable at
    any cost level and is what a fixed validation budget actually looks like
    in practice.
    """
    n = len(df)
    n_abstain = int(round(frac * n))
    k = n - n_abstain
    covered = np.zeros(n, dtype=bool)
    covered[order[:k]] = True
    abstained = ~covered

    p = df["p_true"].to_numpy()
    label = df["label"].to_numpy().astype(bool)
    pred = p >= 0.5
    fn = label & ~pred
    fp = ~label & pred

    fnr_cov, n_pos_cov = _rate(fn, label & covered)
    fpr_cov, n_neg_cov = _rate(fp, ~label & covered)
    total_fn, total_fp = int(fn.sum()), int(fp.sum())
    fn_recovery = float((fn & abstained).sum()) / total_fn if total_fn else float("nan")
    fp_recovery = float((fp & abstained).sum()) / total_fp if total_fp else float("nan")

    return {
        "abstain_frac": frac, "n_abstain": n_abstain, "coverage": round(k / n, 4),
        "FNR_covered": round(fnr_cov, 4) if n_pos_cov else None,
        "FPR_covered": round(fpr_cov, 4) if n_neg_cov else None,
        "total_FN": total_fn, "total_FP": total_fp,
        "FN_recovery_pct": round(100 * fn_recovery, 2) if total_fn else None,
        "FP_recovery_pct": round(100 * fp_recovery, 2) if total_fp else None,
    }


def region_hit_rate(df, in_region_cols, order, k):
    """At coverage cut k (covered = order[:k], abstained = order[k:]),
    compare difficult-region rate + FNR/FPR inside vs outside, and
    covered-vs-abstained, split by region.

    A blended error rate (FN+FP together) is NOT reported: with heavy class
    imbalance (label prevalence ~2-4% after the pool filter) it is dominated
    by FPR and can hide an FNR effect pointing the opposite way -- confirmed
    on real data (AI4S plan.md 2026-09-15 Step 0): GIAB difficult regions
    here have LOWER FPR (candidate generation already restricts to GIAB
    confident regions, so noise inside difficult regions is pre-filtered)
    but HIGHER FNR (true variants are genuinely harder to call there), and
    a blended rate averaged those into a misleading "difficult regions
    aren't risky" reading. Confidence-based abstention (|p_true-threshold|)
    also structurally cannot see the FNR side: a confidently-wrong miss
    (p_true near 0 on a true variant) looks identical to a confidently-
    correct true negative, so it enriches abstained-FPR by ~100-300x while
    leaving FNR roughly flat -- report both sides so that blind spot is
    visible instead of averaged away.
    """
    covered_idx, abstained_idx = order[:k], order[k:]
    p_true = df["p_true"].to_numpy()
    label = df["label"].to_numpy().astype(bool)
    pred = p_true >= 0.5
    fn = label & ~pred
    fp = ~label & pred
    out = {}
    for col in in_region_cols:
        region = df[col].to_numpy()
        overall_rate = float(region.mean())
        covered_rate = float(region[covered_idx].mean()) if k > 0 else float("nan")
        abstained_rate = float(region[abstained_idx].mean()) if k < len(df) else float("nan")
        block = {
            "overall_region_rate": round(overall_rate, 4),
            "covered_region_rate": round(covered_rate, 4),
            "abstained_region_rate": round(abstained_rate, 4),
            "abstained_enrichment": round(abstained_rate / overall_rate, 3) if overall_rate > 0 else None,
        }
        for side_name, side_mask in (("in_region", region), ("out_region", ~region)):
            prevalence, n_side = _rate(label, side_mask)
            fnr, n_pos = _rate(fn, label & side_mask)
            fpr, n_neg = _rate(fp, ~label & side_mask)
            block[side_name] = {"n": n_side, "prevalence": round(prevalence, 4) if n_side else None,
                                "FNR": round(fnr, 4) if n_pos else None, "n_pos": n_pos,
                                "FPR": round(fpr, 4) if n_neg else None, "n_neg": n_neg}
            for cov_name, cov_mask in (("covered", covered_idx), ("abstained", abstained_idx)):
                m = np.zeros(len(df), dtype=bool)
                m[cov_mask] = True
                m &= side_mask
                fnr_c, n_pos_c = _rate(fn, label & m)
                fpr_c, n_neg_c = _rate(fp, ~label & m)
                block[side_name][f"{cov_name}_FNR"] = round(fnr_c, 4) if n_pos_c else None
                block[side_name][f"{cov_name}_FPR"] = round(fpr_c, 4) if n_neg_c else None
        out[col] = block
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomewide-dir", required=True)
    ap.add_argument("--genomes", default="hg002,hg004")
    ap.add_argument("--arms", default="all,no_molecule",
                    help="first arm is primary; deltas = primary - each other arm")
    ap.add_argument("--strat-bed", nargs="*", default=None,
                    help="GIAB stratification BED(s); default strat/lowmap_segdup.bed.gz "
                         "and strat/alldiff.bed.gz relative to the repo root")
    ap.add_argument("--min-alt-reads", type=int, default=2)
    ap.add_argument("--min-vaf", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--risk-budgets", default="0.01,0.02,0.05")
    ap.add_argument("--abstain-fracs", default="0.01,0.02,0.05,0.1,0.15,0.2,0.3",
                    help="fixed validation-cost axis (preferred money-chart axis over "
                         "--risk-budgets, which saturates to 100%% coverage quickly)")
    ap.add_argument("--vaf-bins", default="0,0.05,0.1,0.2,0.35,0.5,1.0")
    ap.add_argument("--chroms", nargs="*", default=None, help="restrict to these chroms (debug)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    strat_paths = args.strat_bed or [
        os.path.join(repo_root, "strat", "lowmap_segdup.bed.gz"),
        os.path.join(repo_root, "strat", "alldiff.bed.gz"),
    ]
    genomes = args.genomes.split(",")
    arms = args.arms.split(",")
    primary_arm = arms[0]
    budgets = [float(x) for x in args.risk_budgets.split(",")]
    abstain_fracs = [float(x) for x in args.abstain_fracs.split(",")]
    vaf_edges = [float(x) for x in args.vaf_bins.split(",")]

    sys.stderr.write(f"[06] loading strat BEDs: {strat_paths}\n")
    strat = {}
    for path in strat_paths:
        if not os.path.exists(path):
            sys.stderr.write(f"[06] WARN strat bed not found, skipping: {path}\n")
            continue
        name = os.path.basename(path).replace(".bed.gz", "").replace(".bed", "")
        strat[name] = load_bed(path)

    # ---- load + filter each (genome, arm) ----
    pooled = {arm: [] for arm in arms}
    for genome in genomes:
        for arm in arms:
            sys.stderr.write(f"[06] loading {genome}/{arm}\n")
            df = load_arm(args.genomewide_dir, genome, arm, args.chroms)
            df = apply_pool_filter(df, args.min_alt_reads, args.min_vaf)
            pooled[arm].append(df)
    pooled = {arm: pd.concat(dfs, ignore_index=True) for arm, dfs in pooled.items()}
    n_pool = len(pooled[primary_arm])
    sys.stderr.write(f"[06] filtered candidate pool: {n_pool} rows "
                     f"(alt_reads>={args.min_alt_reads}, vaf>={args.min_vaf}), "
                     f"label prevalence={pooled[primary_arm]['label'].mean():.4f}\n")

    # region flags computed once on the primary arm; rows are identical
    # across arms (same candidate pool, only feature-set/score differs) --
    # verified by construction (01/02 emit one row per candidate before 03
    # branches on --feature-set).
    region_cols = []
    for name, intervals in strat.items():
        col = f"in_{name}"
        pooled[primary_arm][col] = flag_in_regions(
            pooled[primary_arm]["chrom"].to_numpy(), pooled[primary_arm]["pos"].to_numpy(), intervals)
        region_cols.append(col)
    if region_cols:
        union = np.zeros(n_pool, dtype=bool)
        for col in region_cols:
            union |= pooled[primary_arm][col].to_numpy()
        pooled[primary_arm]["in_any_difficult"] = union
        region_cols.append("in_any_difficult")

    report = {
        "n_pool": n_pool, "genomes": genomes, "arms": arms,
        "min_alt_reads": args.min_alt_reads, "min_vaf": args.min_vaf,
        "threshold": args.threshold, "risk_budgets": budgets,
    }

    # ---- 1. risk-coverage curves + coverage@budget per arm ----
    curves_rows = []
    arm_curves = {}
    for arm in arms:
        cov, risk, order = selective_curve(pooled[arm], args.threshold, rank_by="confidence")
        arm_curves[arm] = (cov, risk, order)
        grid = np.linspace(0.01, 1.0, 100)
        for c, r in zip(grid, grid_sample(cov, risk, grid)):
            curves_rows.append({"arm": arm, "rank_by": "confidence", "coverage": round(float(c), 4),
                                "risk": round(float(r), 5)})
    pd.DataFrame(curves_rows).to_csv(args.out.replace(".json", "") + ".curves.tsv", sep="\t", index=False)

    coverage_at_budget = {}
    for arm in arms:
        cov, risk, order = arm_curves[arm]
        coverage_at_budget[arm] = {}
        for b in budgets:
            k = coverage_index_at_budget(risk, b) + 1
            coverage_at_budget[arm][b] = round(k / len(cov), 4)
    report["coverage_at_budget"] = coverage_at_budget

    savings_delta = {}
    for other in arms[1:]:
        savings_delta[f"{primary_arm}_minus_{other}"] = {
            b: round(coverage_at_budget[primary_arm][b] - coverage_at_budget[other][b], 4)
            for b in budgets
        }
    report["savings_delta_pct_validation_saved"] = savings_delta

    # ---- 1b. fixed-cost money chart: metrics at fixed abstain fraction ----
    # (see metrics_at_abstain_frac docstring for why this axis is preferred
    # over --risk-budgets once budgets start saturating to 100% coverage)
    abstain_rows = []
    for arm in arms:
        _, _, order = arm_curves[arm]
        for frac in abstain_fracs:
            row = {"arm": arm, **metrics_at_abstain_frac(pooled[arm], order, frac)}
            abstain_rows.append(row)
    abstain_df = pd.DataFrame(abstain_rows)
    abstain_df.to_csv(args.out.replace(".json", "") + ".abstain_fracs.tsv", sep="\t", index=False)

    fn_recovery_delta = {}
    for other in arms[1:]:
        fn_recovery_delta[f"{primary_arm}_minus_{other}"] = {}
        for frac in abstain_fracs:
            a = abstain_df[(abstain_df.arm == primary_arm) & (abstain_df.abstain_frac == frac)].iloc[0]
            b = abstain_df[(abstain_df.arm == other) & (abstain_df.abstain_frac == frac)].iloc[0]
            d = (None if pd.isna(a["FN_recovery_pct"]) or pd.isna(b["FN_recovery_pct"])
                 else round(a["FN_recovery_pct"] - b["FN_recovery_pct"], 2))
            fn_recovery_delta[f"{primary_arm}_minus_{other}"][frac] = d
    report["metrics_at_abstain_frac"] = abstain_rows
    report["fn_recovery_delta_pct_points"] = fn_recovery_delta

    # ---- 2. VAF-stratified delta (inverted-U check) ----
    vaf_rows = []
    for lo, hi in zip(vaf_edges[:-1], vaf_edges[1:]):
        cov_at_b = {}
        for arm in arms:
            sub = pooled[arm][(pooled[arm]["vaf"] >= lo) & (pooled[arm]["vaf"] < hi)]
            if len(sub) < 50:
                cov_at_b[arm] = {b: None for b in budgets}
                continue
            cov, risk, _ = selective_curve(sub, args.threshold, rank_by="confidence")
            cov_at_b[arm] = {b: round((coverage_index_at_budget(risk, b) + 1) / len(cov), 4) for b in budgets}
        row = {"vaf_lo": lo, "vaf_hi": hi, "n": int(((pooled[primary_arm]["vaf"] >= lo) &
                                                     (pooled[primary_arm]["vaf"] < hi)).sum())}
        for b in budgets:
            row[f"coverage_{primary_arm}@{b}"] = cov_at_b[primary_arm][b]
            for other in arms[1:]:
                d = (None if cov_at_b[primary_arm][b] is None or cov_at_b[other][b] is None
                     else round(cov_at_b[primary_arm][b] - cov_at_b[other][b], 4))
                row[f"delta_{primary_arm}_minus_{other}@{b}"] = d
        vaf_rows.append(row)
    pd.DataFrame(vaf_rows).to_csv(args.out.replace(".json", "") + ".vaf_strata.tsv", sep="\t", index=False)
    report["vaf_stratified"] = vaf_rows

    # ---- 3. p_true vs p_uncertainty ranking ----
    unc_var = float(pooled[primary_arm]["p_uncertainty"].var())
    unc_block = {"p_uncertainty_variance": unc_var}
    if unc_var <= 0:
        unc_block["degenerate"] = True
        unc_block["note"] = ("p_uncertainty is uniformly constant in this run (n_bag=1 in the "
                             "03_train_eval.py runs that produced these predictions) -- ranking by "
                             "uncertainty degenerates to an arbitrary stable tie-break, not a real "
                             "comparison. Re-run 03 with --n-bag>1 to make this comparison meaningful.")
    else:
        cov_c, risk_c, _ = arm_curves[primary_arm]
        cov_u, risk_u, _ = selective_curve(pooled[primary_arm], args.threshold, rank_by="uncertainty")
        unc_block["degenerate"] = False
        unc_block["auc_risk_confidence_ranked"] = round(curve_auc(cov_c, risk_c), 6)
        unc_block["auc_risk_uncertainty_ranked"] = round(curve_auc(cov_u, risk_u), 6)
        unc_block["confidence_ranking_better"] = curve_auc(cov_c, risk_c) < curve_auc(cov_u, risk_u)
    report["p_true_vs_p_uncertainty"] = unc_block

    # ---- 4. abstention-boundary hit-rate against GIAB difficult regions ----
    # reported on BOTH axes: --risk-budgets (saturates fast, kept for
    # backward compat) and --abstain-fracs (preferred -- stays informative
    # across the whole cost range, see metrics_at_abstain_frac docstring).
    if region_cols:
        cov, risk, order = arm_curves[primary_arm]
        hit_rate = {}
        for b in budgets:
            k = coverage_index_at_budget(risk, b) + 1
            hit_rate[b] = region_hit_rate(pooled[primary_arm], region_cols, order, k)
        hit_rate_by_frac = {}
        for frac in abstain_fracs:
            n = len(pooled[primary_arm])
            k = n - int(round(frac * n))
            hit_rate_by_frac[frac] = region_hit_rate(pooled[primary_arm], region_cols, order, k)
        report["abstention_boundary_hit_rate"] = {
            "arm": primary_arm,
            "by_budget": hit_rate,
            "by_abstain_frac": hit_rate_by_frac,
        }
    else:
        report["abstention_boundary_hit_rate"] = {"note": "no strat BEDs loaded"}

    report = sanitize_nan(report)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    sys.stderr.write(f"[06] n_pool={n_pool}  coverage_at_budget={coverage_at_budget}\n")
    sys.stderr.write(f"[06] savings_delta={savings_delta}\n")
    sys.stderr.write(f"[06] fn_recovery_delta_pct_points={fn_recovery_delta}\n")
    if region_cols:
        top_budget = budgets[0]
        sys.stderr.write(f"[06] abstention hit-rate @ budget={top_budget}: "
                         f"{hit_rate[top_budget]}\n")
    sys.stderr.write(f"[06] -> {args.out}\n")
    print(json.dumps({k: report[k] for k in
                      ("n_pool", "coverage_at_budget", "savings_delta_pct_validation_saved",
                       "metrics_at_abstain_frac", "fn_recovery_delta_pct_points",
                       "p_true_vs_p_uncertainty", "abstention_boundary_hit_rate")}, indent=2))


if __name__ == "__main__":
    main()
