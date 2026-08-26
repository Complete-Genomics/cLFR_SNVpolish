#!/usr/bin/env python3
"""Step 4 (application): re-score candidate SNPs in cLFR consensus isoforms to
produce an isoform with CORRECT SNPs.

Input features come from 02_extract_features.py run on the reads that support the
consensus isoforms, where candidate sites = positions the consensus disagrees with
the reference. This applies the trained Claim A model and decides per SNP:

  KEEP    p_true >= threshold                 -> real transcript SNP, keep it
  REVERT  p_true <  threshold                 -> sequencing/mapping error, drop it
  EDIT    A>G / T>C at a known RNA-editing site -> RNA editing (real biology),
          (optional --editing-bed, REDIportal)    keep & annotate; never revert

The corrected isoform FASTA is then produced from the KEEP set:
  bcftools sort out.pass.vcf -Oz -o out.pass.vcf.gz && tabix -p vcf out.pass.vcf.gz
  bcftools consensus -f <ref_or_transcriptome>.fa out.pass.vcf.gz > corrected_isoforms.fa

Inputs
------
  --features     02_extract_features.py output, run on the CONSENSUS-supporting
                 reads (candidates = positions where consensus != reference).
  --model        model.txt from 03_train_eval.py (the trained lightgbm model;
                 feature columns must match 03's `all` set).
  --out-prefix   prefix for outputs:
                   <prefix>.rescored.tsv  per-SNP p_true + decision
                   <prefix>.pass.vcf      KEEP SNPs (for bcftools consensus)
                   <prefix>.rna_edits.tsv EDIT-annotated RNA-editing sites
  --threshold    (default 0.5) KEEP if p_true >= threshold, else REVERT.
  --editing-bed  (optional) REDIportal-style BED of RNA-editing sites; A>G / T>C
                 hits are labeled EDIT (kept & annotated, never reverted).

Usage
-----
  # feature-extract on consensus reads first (02), then re-score
  python 04_apply_rescore.py \
    --features out/consensus_features.tsv \
    --model    out/claimA_all/model.txt \
    --out-prefix out/corrected --threshold 0.5 \
    --editing-bed REDIportal.hg38.bed
  # then: bcftools sort out/corrected.pass.vcf -Oz -o out/corrected.pass.vcf.gz
  #       tabix -p vcf out/corrected.pass.vcf.gz
  #       bcftools consensus -f REF.fa out/corrected.pass.vcf.gz > corrected_isoforms.fa
"""
import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

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


def load_editing(bed):
    sites = set()
    if bed:
        with open(bed) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.split("\t")
                sites.add((f[0], int(f[2])))  # chrom, 1-based pos (BED end)
    return sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="02_extract_features.py output on consensus-supporting reads")
    ap.add_argument("--model", required=True, help="model.txt from 03_train_eval.py")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--editing-bed", default=None,
                    help="REDIportal-style BED of RNA-editing sites (optional)")
    args = ap.parse_args()

    df = pd.read_csv(args.features, sep="\t")
    booster = lgb.Booster(model_file=args.model)
    df["p_true"] = booster.predict(df[ALL_FEATURES])
    edits = load_editing(args.editing_bed)

    def decide(r):
        if (r["chrom"], int(r["pos"])) in edits and (r["ref"], r["alt"]) in (("A", "G"), ("T", "C")):
            return "EDIT"
        return "KEEP" if r["p_true"] >= args.threshold else "REVERT"

    df["decision"] = df.apply(decide, axis=1)
    df.to_csv(f"{args.out_prefix}.rescored.tsv", sep="\t", index=False)
    df[df["decision"] == "EDIT"].to_csv(f"{args.out_prefix}.rna_edits.tsv", sep="\t", index=False)

    keep = df[df["decision"] == "KEEP"]
    with open(f"{args.out_prefix}.pass.vcf", "w") as v:
        v.write("##fileformat=VCFv4.2\n")
        v.write('##INFO=<ID=PT,Number=1,Type=Float,Description="model P(true variant)">\n')
        v.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, r in keep.iterrows():
            qual = int(min(99, round(-10 * np.log10(max(1e-9, 1 - r["p_true"])))))
            v.write(f'{r["chrom"]}\t{int(r["pos"])}\t.\t{r["ref"]}\t{r["alt"]}\t'
                    f'{qual}\tPASS\tPT={r["p_true"]:.3f}\n')

    c = df["decision"].value_counts().to_dict()
    sys.stderr.write(f"[apply_rescore] {len(df)} SNPs: KEEP={c.get('KEEP', 0)} "
                     f"REVERT={c.get('REVERT', 0)} EDIT={c.get('EDIT', 0)}\n")
    sys.stderr.write(
        f"corrected isoform FASTA:\n"
        f"  bcftools sort {args.out_prefix}.pass.vcf -Oz -o {args.out_prefix}.pass.vcf.gz\n"
        f"  tabix -p vcf {args.out_prefix}.pass.vcf.gz\n"
        f"  bcftools consensus -f REF.fa {args.out_prefix}.pass.vcf.gz > corrected_isoforms.fa\n")


if __name__ == "__main__":
    main()
