#!/usr/bin/env python3
"""Step 3 (Claim A): train + evaluate a true-vs-error variant confidence model.

Discriminates true variants from sequencing/mapping errors using molecule-linkage
features. Honest protocol (lessons from the leaky sqanti3.py):
  * chromosome-held-out test split (no position leakage)
  * NO scaling/resampling before the split; class imbalance -> scale_pos_weight
  * probability calibration fit on a held-out slice of TRAIN, evaluated on TEST
  * metrics stratified by VAF (where the low-freq story lives)

Options:
  --model {lightgbm,xgboost}  same GBDT family; xgboost is a robustness check to
                              show the result is not model-dependent.
  --n-bag N                   seed-bagged ensemble of N models; prediction MEAN is
                              the score, prediction STD is a per-SNP uncertainty
                              (flag uncertain calls in repeats / low-support sites).

Ablation (money chart): run with --feature-set all vs baseline; the PR-AUC delta
(largest among higher-VAF errors = mapping artifacts) is the molecule-feature value.

Inputs
------
  --features        feature TSV from 02_extract_features.py.
  --out-dir         output dir (metrics.json, feature_importance.csv,
                    calibration.csv, strata_counts_*.csv, test_predictions.tsv, model.*).
  --test-chroms     chrom(s) held out for test, e.g. `chr20` (rest = train).
  --feature-set     all | baseline | no_molecule | molecule_only (UMI ablation:
                    UMI value = PR-AUC(all) - PR-AUC(no_molecule)).
  --model           lightgbm (default) | xgboost (robustness check).
  --n-bag           (default 1) seed-bagged models; >1 also emits per-SNP uncertainty.
  --calib-frac      (default 0.2) TRAIN fraction held out to fit calibration.
  --threshold       (default 0.5) decision threshold for the confusion matrix.
  --min-hard-count  (default 200) warn if a hard class (high-VAF error / low-VAF
                    true) is thinner than this.
  --threads         (default 8) cap CPU threads for GBDT + OMP/BLAS. -1 = all cores.
  --seed            (default 42).

Resources (this step is light; 01/02 pileup are the heavy CPU steps)
  CPU:  GBDT grabs ALL cores by default -> on a big shared server (e.g. 128-core)
        cap with --threads N (default 8; also caps OMP/BLAS). --n-bag N runs N
        trainings sequentially -> ~N x time.
  MEM:  dominated by loading features.tsv into pandas, NOT the model.
        rule of thumb ~2-5 GB per ~10M candidate rows.
        chr1 train + chr20 test (~100k-1M rows) -> 4-8 GB is safe.
        genome-wide (10M+ rows) -> 16 GB comfortable. If OOM, train on fewer chroms.

Usage
-----
  # money chart: 3-stage ablation isolating the UMI contribution
  for FS in baseline no_molecule all; do
    python 03_train_eval.py --features out/features.tsv \
      --out-dir out/claimA_$FS --test-chroms chr20 --feature-set $FS
  done   # UMI value = PR-AUC(claimA_all) - PR-AUC(claimA_no_molecule)

  # uncertainty (seed-bagging) + xgboost robustness check
  python 03_train_eval.py --features out/features.tsv --out-dir out/bag \
    --test-chroms chr20 --n-bag 10
  python 03_train_eval.py --features out/features.tsv --out-dir out/xgb \
    --test-chroms chr20 --model xgboost
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             roc_auc_score)

ALL_FEATURES = [
    "dp", "alt_reads", "ref_reads", "vaf",
    "n_mol_total", "n_mol_alt", "n_mol_ref", "mol_alt_fraction",
    "alt_reads_per_alt_mol_mean", "within_mol_alt_agreement_mean",
    "alt_bq_mean", "alt_bq_min", "ref_bq_mean",
    "alt_mapq_mean", "alt_mapq_min", "ref_mapq_mean",
    "alt_strand_balance", "alt_softclip_frac_mean",
    "alt_readpos_fromend_mean", "alt_readpos_fromend_min",
    "alt_indel_near_frac", "alt_nm_mean", "homopolymer_run",
    "alt_clip_frac_mean", "alt_supplementary_frac",
]
# feature groups for ablation. UMI/molecule features REQUIRE grouping reads by
# barcode/UMI (done in 02); per-read/context features (incl. CIGAR-robustness:
# alt_indel_near_frac / alt_nm_mean / homopolymer_run) do not.
BASELINE = {"dp", "alt_reads", "ref_reads", "vaf"}                 # plain pileup
MOLECULE = {"n_mol_total", "n_mol_alt", "n_mol_ref", "mol_alt_fraction",
            "alt_reads_per_alt_mol_mean", "within_mol_alt_agreement_mean"}  # UMI
# everything else = per-read mapping/quality (mapq, softclip, strand, bq, readpos)


def feature_set(name):
    """Ordered subset of ALL_FEATURES. UMI contribution = all - no_molecule."""
    if name == "all":
        return list(ALL_FEATURES)
    if name == "baseline":
        return [f for f in ALL_FEATURES if f in BASELINE]
    if name == "no_molecule":  # baseline + per-read, NO UMI
        return [f for f in ALL_FEATURES if f not in MOLECULE]
    if name == "molecule_only":  # pileup + UMI, NO per-read mapping/quality
        return [f for f in ALL_FEATURES if f in BASELINE or f in MOLECULE]
    if name == "no_abs":  # all MINUS absolute molecule counts (keep ratio molecule feats)
        return [f for f in ALL_FEATURES if f not in {"n_mol_total", "n_mol_alt", "n_mol_ref"}]
    raise ValueError(name)


VAF_BINS = [(0, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)]


def stratify_counts(frame):
    """label x VAF-bin counts -> DataFrame. Binding constraint is the minority
    hard cells (high-VAF errors = mapping artifacts; low-VAF true = low support),
    not the total sample count."""
    rows = []
    for lo, hi in VAF_BINS:
        sub = frame[(frame["vaf"] >= lo) & (frame["vaf"] < hi)]
        rows.append({"vaf_bin": f"{lo}-{hi}",
                     "error(0)": int((sub["label"] == 0).sum()),
                     "true(1)": int((sub["label"] == 1).sum())})
    return pd.DataFrame(rows)


def build_model(kind, seed, scale_pos_weight, n_jobs):
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, verbose=-1,
                              n_jobs=n_jobs,
                              scale_pos_weight=scale_pos_weight, random_state=seed)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             eval_metric="logloss", n_jobs=n_jobs,
                             scale_pos_weight=scale_pos_weight, random_state=seed)
    raise ValueError(kind)


def feature_gain(model, kind, feats):
    if kind == "lightgbm":
        vals = model.booster_.feature_importance(importance_type="gain")
        return {f: float(g) for f, g in zip(feats, vals)}
    d = model.get_booster().get_score(importance_type="gain")
    return {f: float(d.get(f, 0.0)) for f in feats}


def save_model(model, kind, out_dir):
    if kind == "lightgbm":
        model.booster_.save_model(f"{out_dir}/model.txt")
    else:
        model.get_booster().save_model(f"{out_dir}/model.json")


def run_cv(df, feats, args):
    """K-fold CV for single-chromosome data. GroupKFold by --cv-block bp so
    neighbouring (correlated) sites stay in the same fold -> no position leakage.
    Reports per-fold + mean/std ROC-AUC, PR-AUC, Brier."""
    from sklearn.model_selection import GroupKFold
    from sklearn.isotonic import IsotonicRegression
    os.makedirs(args.out_dir, exist_ok=True)
    X, y = df[feats].reset_index(drop=True), df["label"].values
    groups = (df["pos"] // args.cv_block).values
    n_groups = len(set(groups))
    k = min(args.cv, n_groups)
    if k < 2:
        sys.exit(f"ERROR: only {n_groups} position blocks; need >=2 for CV.")
    sys.stderr.write(f"[cv] {k}-fold GroupKFold, {n_groups} blocks of {args.cv_block}bp, "
                     f"n={len(df)}, pos={int(y.sum())}, neg={int((y==0).sum())}\n")

    gkf = GroupKFold(n_splits=k)
    folds, gains = [], {f: 0.0 for f in feats}
    for i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ytr = y[tr]
        spw = max(1, int((ytr == 0).sum())) / max(1, int(ytr.sum()))
        rng = np.random.default_rng(args.seed + i)
        cal = rng.random(len(tr)) < args.calib_frac
        fit_i, cal_i = tr[~cal], tr[cal]
        model = build_model(args.model, args.seed + i, spw, args.threads)
        model.fit(X.iloc[fit_i], y[fit_i])
        calib = None
        if len(cal_i) > 50 and len(set(y[cal_i])) == 2:
            pc = model.predict_proba(X.iloc[cal_i])[:, 1]
            calib = IsotonicRegression(out_of_bounds="clip").fit(pc, y[cal_i])
        praw = model.predict_proba(X.iloc[te])[:, 1]
        pt = calib.predict(praw) if calib is not None else praw
        yte = y[te]
        two = len(set(yte)) == 2
        folds.append({
            "fold": i, "n_test": int(len(te)),
            "test_pos": int(yte.sum()), "test_neg": int((yte == 0).sum()),
            "roc_auc": float(roc_auc_score(yte, pt)) if two else None,
            "pr_auc": float(average_precision_score(yte, pt)) if two else None,
            "brier": float(brier_score_loss(yte, pt)) if two else None,
        })
        for f, g in feature_gain(model, args.model, feats).items():
            gains[f] += g / k
        sys.stderr.write(f"[cv] fold {i}: pr_auc={folds[-1]['pr_auc']} "
                         f"roc_auc={folds[-1]['roc_auc']} test_pos={folds[-1]['test_pos']}\n")

    def agg(key):
        v = [m[key] for m in folds if m[key] is not None]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None

    summary = {
        "mode": "cv", "cv": k, "cv_block": args.cv_block, "n_groups": n_groups,
        "feature_set": args.feature_set, "n_features": len(feats), "model": args.model,
        "n_total": int(len(df)), "total_pos": int(y.sum()), "total_neg": int((y == 0).sum()),
        "roc_auc": agg("roc_auc"), "pr_auc": agg("pr_auc"), "brier": agg("brier"),
        "folds": folds,
    }
    pd.DataFrame([{"feature": f, "gain": gains[f]} for f in feats]
                 ).sort_values("gain", ascending=False).to_csv(
        f"{args.out_dir}/feature_importance.csv", index=False)
    with open(f"{args.out_dir}/cv_metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({"cv": k, "roc_auc": summary["roc_auc"],
                      "pr_auc": summary["pr_auc"], "brier": summary["brier"],
                      "n_groups": n_groups}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-chroms", nargs="+", default=None,
                    help="chrom(s) held out for test (chrom-holdout mode). "
                         "Omit when using --cv.")
    ap.add_argument("--cv", type=int, default=0,
                    help="if >1, k-fold cross-validation instead of chrom-holdout. "
                         "Use for single-chromosome data (e.g. only chr22).")
    ap.add_argument("--cv-block", type=int, default=1_000_000,
                    help="GroupKFold block size in bp: sites within a block stay in "
                         "the same fold (prevents position leakage). default 1Mb.")
    ap.add_argument("--feature-set",
                    choices=["all", "baseline", "no_molecule", "molecule_only", "no_abs"],
                    default="all",
                    help="baseline=pileup; no_molecule=+per-read (NO UMI); "
                         "molecule_only=pileup+UMI; all=everything. "
                         "UMI contribution = PR-AUC(all) - PR-AUC(no_molecule).")
    ap.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    ap.add_argument("--n-bag", type=int, default=1,
                    help="N seed-bagged models; >1 also yields per-SNP uncertainty (pred std)")
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-hard-count", type=int, default=200,
                    help="warn if a hard class (high-VAF error / low-VAF true) is thinner than this")
    ap.add_argument("--threads", type=int, default=8,
                    help="max CPU threads for LightGBM/XGBoost (default 8; prevents "
                         "grabbing all cores on a big shared server). -1 = all.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # cap BLAS/OpenMP threads (pandas/numpy/sklearn) too, before heavy work
    if args.threads and args.threads > 0:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(v, str(args.threads))

    os.makedirs(args.out_dir, exist_ok=True)
    feats = feature_set(args.feature_set)

    df = pd.read_csv(args.features, sep="\t")
    df["label"] = df["label"].astype(int)

    if args.cv and args.cv > 1:
        run_cv(df, feats, args)
        return
    if not args.test_chroms:
        sys.exit("ERROR: pass --test-chroms (chrom-holdout) or --cv K (cross-val).")

    test_df = df[df["chrom"].isin(args.test_chroms)].copy()
    train_df = df[~df["chrom"].isin(args.test_chroms)].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        sys.exit("ERROR: empty train or test split; check --test-chroms")

    # pre-flight: is there enough of each class x VAF stratum to learn from?
    strata_tr, strata_te = stratify_counts(train_df), stratify_counts(test_df)
    strata_tr.to_csv(f"{args.out_dir}/strata_counts_train.csv", index=False)
    strata_te.to_csv(f"{args.out_dir}/strata_counts_test.csv", index=False)
    sys.stderr.write("[strata] TRAIN label x VAF:\n" + strata_tr.to_string(index=False) + "\n")
    sys.stderr.write("[strata] TEST  label x VAF:\n" + strata_te.to_string(index=False) + "\n")
    hard = {
        "high-VAF errors (vaf>=0.25, mapping artifacts)":
            int(train_df[(train_df.label == 0) & (train_df.vaf >= 0.25)].shape[0]),
        "low-VAF true (vaf<0.1, low support)":
            int(train_df[(train_df.label == 1) & (train_df.vaf < 0.1)].shape[0]),
    }
    for name, n in hard.items():
        if n < args.min_hard_count:
            sys.stderr.write(
                f"[strata][WARN] TRAIN hard class thin: {name} = {n} (< {args.min_hard_count}). "
                f"Add chromosomes/samples; for low-VAF true, use a GIAB titration (Claim B).\n")

    rng = np.random.default_rng(args.seed)
    is_calib = rng.random(len(train_df)) < args.calib_frac
    fit_df, calib_df = train_df[~is_calib], train_df[is_calib]

    # seed-bagged ensemble (n_bag=1 -> single model, no variance)
    models = []
    for b in range(args.n_bag):
        sample = fit_df if args.n_bag == 1 else fit_df.sample(
            frac=1.0, replace=True, random_state=args.seed + b)
        y = sample["label"].values
        n_pos, n_neg = max(1, int(y.sum())), max(1, int((y == 0).sum()))
        m = build_model(args.model, args.seed + b, n_neg / n_pos, args.threads)
        m.fit(sample[feats], y)
        models.append(m)

    def raw_pred(frame):
        P = np.column_stack([m.predict_proba(frame[feats])[:, 1] for m in models])
        return P.mean(axis=1), P.std(axis=1)

    calibrator = None
    if len(calib_df) > 50 and calib_df["label"].nunique() == 2:
        mean_calib, _ = raw_pred(calib_df)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(mean_calib, calib_df["label"].values)

    mean_test, unc_test = raw_pred(test_df)
    test_df["p_true"] = calibrator.predict(mean_test) if calibrator is not None else mean_test
    test_df["p_uncertainty"] = unc_test  # bagged std (0 when n_bag=1)
    yt, pt = test_df["label"].values, test_df["p_true"].values
    two = len(set(yt)) == 2
    pred = (pt >= args.threshold).astype(int)

    metrics = {
        "model": args.model, "n_bag": args.n_bag, "feature_set": args.feature_set,
        "n_features": len(feats), "n_fit": int(len(fit_df)), "n_calib": int(len(calib_df)),
        "n_test": int(len(test_df)), "test_pos": int(yt.sum()), "test_neg": int((yt == 0).sum()),
        "roc_auc": float(roc_auc_score(yt, pt)) if two else None,
        "pr_auc": float(average_precision_score(yt, pt)) if two else None,
        "brier": float(brier_score_loss(yt, pt)) if two else None,
        "confusion_matrix@thr": confusion_matrix(yt, pred).tolist(),
        "report@thr": classification_report(yt, pred, output_dict=True, zero_division=0),
        "vaf_strata_pr_auc": {},
    }
    if args.n_bag > 1:
        correct = pred == yt
        metrics["mean_uncertainty"] = float(unc_test.mean())
        metrics["uncertainty_correct"] = float(unc_test[correct].mean()) if correct.any() else None
        metrics["uncertainty_wrong"] = float(unc_test[~correct].mean()) if (~correct).any() else None
    for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)]:
        sub = test_df[(test_df["vaf"] >= lo) & (test_df["vaf"] < hi)]
        if sub["label"].nunique() == 2:
            metrics["vaf_strata_pr_auc"][f"{lo}-{hi}"] = {
                "n": int(len(sub)),
                "pr_auc": float(average_precision_score(sub["label"], sub["p_true"])),
            }

    gains = {f: float(np.mean([feature_gain(m, args.model, feats)[f] for m in models]))
             for f in feats}
    pd.DataFrame({"feature": feats, "gain": [gains[f] for f in feats]}
                 ).sort_values("gain", ascending=False).to_csv(
        f"{args.out_dir}/feature_importance.csv", index=False)

    order = np.argsort(pt)
    pd.DataFrame([{"mean_pred": float(pt[b].mean()), "frac_true": float(yt[b].mean()),
                   "n": int(len(b))} for b in np.array_split(order, 10) if len(b)]
                 ).to_csv(f"{args.out_dir}/calibration.csv", index=False)

    test_df.to_csv(f"{args.out_dir}/test_predictions.tsv", sep="\t", index=False)
    with open(f"{args.out_dir}/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    save_model(models[0], args.model, args.out_dir)

    keys = ["model", "n_bag", "feature_set", "roc_auc", "pr_auc", "brier",
            "test_pos", "test_neg", "vaf_strata_pr_auc"]
    if args.n_bag > 1:
        keys += ["mean_uncertainty", "uncertainty_correct", "uncertainty_wrong"]
    print(json.dumps({k: metrics[k] for k in keys if k in metrics}, indent=2))


if __name__ == "__main__":
    main()