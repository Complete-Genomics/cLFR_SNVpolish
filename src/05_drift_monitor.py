#!/usr/bin/env python3
"""Step 5: data-drift monitor for the Step-1 true-vs-error re-scorer.

MOST routine samples have NO ERCC (cost), so drift detection is primarily
UNSUPERVISED — it compares the incoming batch's feature distributions against the
training/reference distributions. No labels needed. ERCC, when present, is an
OPTIONAL labeled check (any alt in ERCC = error).

What it reports
  1. Per-feature PSI (Population Stability Index) reference vs batch, + NaN-rate
     drift + schema check.  PSI bands: <0.1 stable, 0.1-0.25 watch, >0.25 alert.
     Drift on HIGH-importance features (from 03's feature_importance.csv) matters
     most, so those are weighted / called out.
  2. Prediction drift (if --model): apply the model to reference & batch, compare
     the P(true) score distribution + KEEP rate at --threshold (a jump = something
     changed even if no single feature screams).
  3. ERCC check (optional --ercc-features): false-KEEP rate + mean p_true + Brier
     against all-error labels (a labeled slice when you happen to have it).
  4. Verdict: PASS / WATCH / RETRAIN with reasons -> the "detect drift" node of the
     agentic retrain loop.

Inputs
------
  --reference           training/reference features.tsv (defines baseline dists).
  --batch               incoming batch features.tsv (from 02; label ignored/optional).
  --model               (optional) model.txt from 03, for prediction drift.
  --feature-importance  (optional) feature_importance.csv from 03; weights drift by gain.
  --ercc-features       (optional) features.tsv from ERCC reads (all alt = error).
  --out                 output JSON report (also writes <out>.per_feature.tsv).
  --psi-warn 0.1 --psi-alert 0.25 --keep-delta 0.1 --top-k 5 --ref-sample 200000
  --threshold 0.5 --threads 8

Usage
-----
  python 05_drift_monitor.py --reference out/features.tsv --batch batchB/features.tsv \
    --model out/claimA_all/model.txt --feature-importance out/claimA_all/feature_importance.csv \
    --out drift/batchB.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

FEATURES = [
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


def psi(ref, batch, bins=10, eps=1e-6):
    """PSI on quantile bins defined by the reference. NaNs excluded (tracked
    separately). Returns nan for near-constant features."""
    ref = np.asarray(ref, float)
    batch = np.asarray(batch, float)
    ref = ref[~np.isnan(ref)]
    batch = batch[~np.isnan(batch)]
    if len(ref) < 2 or len(batch) < 1:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    b = np.clip(np.histogram(batch, edges)[0] / len(batch), eps, None)
    return float(np.sum((b - r) * np.log(b / r)))


def band(v, warn, alert):
    if np.isnan(v):
        return "na"
    return "alert" if v > alert else ("watch" if v > warn else "stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--feature-importance", default=None)
    ap.add_argument("--ercc-features", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--psi-warn", type=float, default=0.1)
    ap.add_argument("--psi-alert", type=float, default=0.25)
    ap.add_argument("--keep-delta", type=float, default=0.1,
                    help="KEEP-rate abs shift that alerts")
    ap.add_argument("--top-k", type=int, default=5,
                    help="a top-k importance feature in the alert band forces RETRAIN")
    ap.add_argument("--n-alert", type=int, default=3,
                    help="this many alert-band features forces RETRAIN")
    ap.add_argument("--ref-sample", type=int, default=200000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    if args.threads and args.threads > 0:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, str(args.threads))

    ref = pd.read_csv(args.reference, sep="\t")
    batch = pd.read_csv(args.batch, sep="\t")
    if len(ref) > args.ref_sample:
        ref = ref.sample(args.ref_sample, random_state=0)

    present = [f for f in FEATURES if f in ref.columns and f in batch.columns]
    missing = [f for f in FEATURES if f not in batch.columns]

    importance = {}
    top_feats = set()
    if args.feature_importance and os.path.exists(args.feature_importance):
        imp = pd.read_csv(args.feature_importance)
        importance = dict(zip(imp["feature"], imp["gain"]))
        top_feats = set(imp.sort_values("gain", ascending=False)["feature"].head(args.top_k))

    # ---- per-feature drift ----
    rows = []
    for f in present:
        p = psi(ref[f].values, batch[f].values)
        rows.append({
            "feature": f,
            "psi": round(p, 4) if not np.isnan(p) else None,
            "band": band(p, args.psi_warn, args.psi_alert),
            "gain": round(float(importance.get(f, 0.0)), 1) if importance else None,
            "top_k": f in top_feats,
            "ref_nan_rate": round(float(ref[f].isna().mean()), 4),
            "batch_nan_rate": round(float(batch[f].isna().mean()), 4),
        })
    pf = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
    pf.to_csv(args.out.replace(".json", "") + ".per_feature.tsv", sep="\t", index=False)

    alert_feats = pf[pf["band"] == "alert"]["feature"].tolist()
    watch_feats = pf[pf["band"] == "watch"]["feature"].tolist()
    top_alert = [f for f in alert_feats if f in top_feats]
    nan_spikes = pf[(pf["batch_nan_rate"] - pf["ref_nan_rate"]) > 0.1]["feature"].tolist()

    report = {
        "n_reference": int(len(ref)), "n_batch": int(len(batch)),
        "missing_features": missing,
        "n_alert_features": len(alert_feats), "alert_features": alert_feats,
        "watch_features": watch_feats,
        "top_importance_alert": top_alert,
        "nan_rate_spikes": nan_spikes,
    }

    reasons = []
    if missing:
        reasons.append(f"schema: missing features {missing}")
    if nan_spikes:
        reasons.append(f"NaN-rate spike in {nan_spikes}")
    if top_alert:
        reasons.append(f"top-{args.top_k} importance feature(s) in alert band: {top_alert}")
    if len(alert_feats) >= args.n_alert:
        reasons.append(f"{len(alert_feats)} features in alert band (>= {args.n_alert})")

    # ---- prediction drift (optional) ----
    if args.model and os.path.exists(args.model) and not missing:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=args.model)
        rp = booster.predict(ref[FEATURES])
        bp = booster.predict(batch[FEATURES])
        keep_ref = float((rp >= args.threshold).mean())
        keep_batch = float((bp >= args.threshold).mean())
        report["prediction_drift"] = {
            "keep_rate_ref": round(keep_ref, 4), "keep_rate_batch": round(keep_batch, 4),
            "keep_rate_delta": round(keep_batch - keep_ref, 4),
            "mean_score_ref": round(float(rp.mean()), 4),
            "mean_score_batch": round(float(bp.mean()), 4),
            "score_psi": round(psi(rp, bp), 4),
        }
        if abs(keep_batch - keep_ref) > args.keep_delta:
            reasons.append(f"KEEP-rate shifted {keep_batch - keep_ref:+.3f} (> {args.keep_delta})")

    # ---- ERCC labeled check (optional; most samples lack it) ----
    if args.ercc_features and os.path.exists(args.ercc_features) and args.model and not missing:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=args.model)
        e = pd.read_csv(args.ercc_features, sep="\t")
        ep = booster.predict(e[FEATURES])
        false_keep = float((ep >= args.threshold).mean())  # all ERCC alt are errors
        report["ercc_check"] = {
            "n": int(len(e)),
            "false_keep_rate": round(false_keep, 4),      # want ~0
            "mean_p_true": round(float(ep.mean()), 4),    # want low
            "brier": round(float(np.mean(ep ** 2)), 4),   # vs all-zero labels
        }
        if false_keep > 0.05:
            reasons.append(f"ERCC false-KEEP rate {false_keep:.3f} > 0.05 (labeled degradation)")

    # ---- verdict ----
    verdict = "RETRAIN" if reasons else ("WATCH" if watch_feats else "PASS")
    # watch-only reasons shouldn't force RETRAIN; downgrade if only soft signals
    hard = [r for r in reasons if not r.startswith("schema")] or reasons
    if verdict == "RETRAIN" and not hard:
        verdict = "WATCH"
    report["verdict"] = verdict
    report["reasons"] = reasons or (["watch-band features present"] if watch_feats else ["no drift"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    sys.stderr.write(f"[drift] verdict={verdict}  alert_features={len(alert_feats)}  "
                     f"top_alert={top_alert}  -> {args.out}\n")
    print(json.dumps({k: report[k] for k in
                      ("verdict", "reasons", "n_alert_features", "top_importance_alert",
                       "prediction_drift") if k in report}, indent=2))


if __name__ == "__main__":
    main()