#!/usr/bin/env python3
"""Step 1 (Claim A): build labeled candidate SNP sites from a BAM.

Within GIAB confident regions, label each site as:
  1 = true variant   -> position is a GIAB SNV
  0 = error          -> confident hom-ref position that still carries alt reads,
                        i.e. that alt support is sequencing or mapping error.

Only SNVs are used. GIAB indel/other positions are excluded from the error class
(ambiguous). Output feeds 02_extract_features.py.

Inputs
------
  --bam            aligned reads. Default input = SE600 (single-end ~600bp,
                   minimap2). Needs .bai. UMI/barcode may be in read name or BX tag
                   (used later by 02, not here).
  --ref            reference FASTA, needs .fai (samtools faidx).
  --truth-vcf      GIAB SNV truth VCF, bgzip + tabix (.tbi). e.g. HG002
                   HG002_GRCh38_1_22_..._benchmark.vcf.gz
  --confident-bed  GIAB high-confidence regions BED (0-based half-open). Labels are
                   only made inside these regions (outside, hom-ref is not trusted).
  --regions        one or more chroms/regions to scan, e.g. `chr1` or
                   `chr20:1-1000000`. Train on one chrom, hold out another for test.
  --out            output candidates TSV.
  --min-alt-reads  (default 2) min alt reads for an error-class candidate.
  --min-base-quality (default 0) SE600 keeps low-qual bases; leave 0.
  --max-depth      (default 8000) pileup depth cap.

Output TSV columns: chrom  pos  ref  alt  label  dp  alt_reads  vaf
(label: 1=true GIAB SNV, 0=error at confident hom-ref)

Usage
-----
  # train chrom (chr1) + held-out test chrom (chr20), HG002 SE600, bam and vcf need index
  python 01_make_candidates.py \
    --bam   HG002.se600.minimap2.bam \
    --ref   GRCh38.fa \
    --truth-vcf HG002_GIAB_SNV.vcf.gz \
    --confident-bed HG002_confident.bed \
    --regions chr1 chr20 \
    --out   out/candidates.tsv \
    --min-alt-reads 2

  # single small region for a quick smoke test
  python 01_make_candidates.py --bam a.bam --ref ref.fa \
    --truth-vcf t.vcf.gz --confident-bed c.bed \
    --regions chr20:1-2000000 --out out/candidates.smoke.tsv

Requires: pysam. Truth VCF must be bgzip+tabix; confident BED is 0-based half-open.
"""
import argparse
import sys

import numpy as np
import pysam


def load_confident(bed_path):
    starts, ends = {}, {}
    with open(bed_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            starts.setdefault(f[0], []).append(int(f[1]))
            ends.setdefault(f[0], []).append(int(f[2]))
    for c in starts:
        order = np.argsort(starts[c])
        starts[c] = np.array(starts[c])[order]
        ends[c] = np.array(ends[c])[order]
    return starts, ends


def in_confident(chrom, pos0, starts, ends):
    if chrom not in starts:
        return False
    i = np.searchsorted(starts[chrom], pos0, side="right") - 1
    return i >= 0 and pos0 < ends[chrom][i]


def load_truth(vcf_path, chrom, start, end):
    snv, any_var = {}, set()
    for rec in pysam.VariantFile(vcf_path).fetch(chrom, start, end):
        pos0 = rec.pos - 1
        any_var.add(pos0)
        if rec.alts:
            for alt in rec.alts:
                if len(rec.ref) == 1 and len(alt) == 1:
                    snv[pos0] = (rec.ref.upper(), alt.upper())
    return snv, any_var


def parse_region(region, fasta):
    if ":" in region:
        c, rng = region.split(":")
        s, e = rng.replace(",", "").split("-")
        return c, int(s) - 1, int(e)
    return region, 0, fasta.get_reference_length(region)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--truth-vcf", required=True, help="GIAB SNV truth VCF (bgzip+tabix)")
    ap.add_argument("--confident-bed", required=True)
    ap.add_argument("--exclude-vcf", default=None,
                    help="TITRATION: variants (union of both samples) whose positions are "
                         "excluded from the ERROR class, so background = true hom-ref in both. "
                         "Positives still come from --truth-vcf (e.g. sample-private SNVs).")
    ap.add_argument("--regions", nargs="+", required=True, help="e.g. chr20 chr20:1-1000000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-alt-reads", type=int, default=2,
                    help="min alt reads to accept an error-class candidate")
    ap.add_argument("--threads", type=int, default=4, help="BAM decompression threads")
    ap.add_argument("--min-base-quality", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=8000)
    args = ap.parse_args()

    starts, ends = load_confident(args.confident_bed)
    bam = pysam.AlignmentFile(args.bam, "rb", threads=max(1, args.threads))
    fasta = pysam.FastaFile(args.ref)

    n_true = n_err = 0
    with open(args.out, "w") as out:
        out.write("chrom\tpos\tref\talt\tlabel\tdp\talt_reads\tvaf\n")
        for region in args.regions:
            chrom, rstart, rend = parse_region(region, fasta)
            if chrom not in starts:
                continue
            snv, any_var = load_truth(args.truth_vcf, chrom, rstart, rend)
            if args.exclude_vcf:
                _, any_var = load_truth(args.exclude_vcf, chrom, rstart, rend)

            # pileup ONLY inside confident intervals (skip non-confident spans);
            # prefetch the interval reference once instead of one fetch per column.
            for s0, e0 in zip(starts[chrom], ends[chrom]):
                s, e = max(int(s0), rstart), min(int(e0), rend)
                if s >= e:
                    continue
                refseq = fasta.fetch(chrom, s, e).upper()
                for col in bam.pileup(chrom, s, e, truncate=True,
                                      min_base_quality=args.min_base_quality,
                                      stepper="samtools", max_depth=args.max_depth):
                    pos0 = col.reference_pos
                    refbase = refseq[pos0 - s]
                    if refbase not in "ACGT":
                        continue

                    counts = {}
                    for pr in col.pileups:
                        if pr.is_del or pr.is_refskip or pr.query_position is None:
                            continue
                        b = pr.alignment.query_sequence[pr.query_position].upper()
                        counts[b] = counts.get(b, 0) + 1
                    dp = sum(counts.values())
                    if dp == 0:
                        continue

                    if pos0 in snv:
                        ref, alt = snv[pos0]
                        alt_reads = counts.get(alt, 0)
                        if alt_reads < 1:
                            continue  # no observed alt -> nothing to featurize
                        label = 1
                        n_true += 1
                    else:
                        if pos0 in any_var:
                            continue  # GIAB indel/other -> ambiguous
                        alt, alt_reads = None, 0
                        for b, c in counts.items():
                            if b != refbase and b in "ACGT" and c > alt_reads:
                                alt, alt_reads = b, c
                        if alt is None or alt_reads < args.min_alt_reads:
                            continue
                        ref, label = refbase, 0
                        n_err += 1

                    out.write(f"{chrom}\t{pos0 + 1}\t{ref}\t{alt}\t{label}\t"
                              f"{dp}\t{alt_reads}\t{alt_reads / dp:.4f}\n")

    sys.stderr.write(f"[make_candidates] true={n_true} error={n_err} -> {args.out}\n")


if __name__ == "__main__":
    main()
