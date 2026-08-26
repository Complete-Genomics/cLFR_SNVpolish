#!/usr/bin/env python3
"""Step 2 (Claim A): molecule-linkage features per candidate site.

Reads are grouped by UMI/barcode (molecule) so we can measure whether alt support
is consistent WITHIN independent molecules (true variant) or scattered across
single reads (sequencing error), plus mapping-error indicators.\n\nPerf: ONE streaming pileup pass per chromosome (candidate positions looked up in a\ndict), NOT one pileup call per site -- ~100x faster on dense candidate sets.

Default input = SE600 (single-end ~600bp, mapped with minimap2). minimap2 CIGAR is
messy near indels/homopolymers/junctions, so we add CIGAR-robustness features that
let the model distrust alignment-artifact positions:
  * alt_indel_near_frac  fraction of alt reads with an I/D within --indel-window bp
                         (N/intron does NOT count -> splice-aware)
  * alt_nm_mean          mean edit distance (NM tag) of alt reads (noisy alignment)
  * homopolymer_run      reference homopolymer run length at the site
Note: for SE600 we keep low-MAPQ reads (min-base-quality/MAPQ not used as a hard
filter); MAPQ enters only as a feature. For RNA, map with `minimap2 -x splice`.

Molecule id source:
  --molecule-source tag            -> read tag (default BX)
  --molecule-source readname_regex -> group(1) of a regex on the read name
  --molecule-source read           -> each read is its own molecule (ablation)

Inputs
------
  --bam              same BAM as 01 (SE600, needs .bai). Reads are re-pileup'd at
                     each candidate to extract molecule/per-read features.
  --ref              reference FASTA (+ .fai) — for homopolymer_run context.
  --candidates       TSV from 01_make_candidates.py (chrom/pos/ref/alt/label...).
  --out              output feature TSV (fed to 03).
  --molecule-source  tag | readname_regex | read (how to get the UMI/barcode).
  --molecule-tag     (default BX) tag name when --molecule-source tag.
  --readname-regex   (default `#([ACGTN]+)`) group(1)=barcode, for readname_regex.
  --indel-window     (default 10) an alt read is "indel-adjacent" if it has an I/D
                     within this many ref bp (N/intron excluded -> splice-aware).
  --regions          (optional) restrict to these chroms, for sharding across jobs;
                     default = every chrom in the candidates file.
  --threads          (default 4) BAM decompression threads (htslib).
  --min-base-quality (default 0) SE600 keeps low-qual bases; leave 0.
  --max-depth        (default 8000) pileup depth cap.

Output TSV: chrom pos ref alt label + 25 feature columns (see FEATURES).

Usage
-----
  # cLFR barcode in read name (SE600), homopolymer needs --ref
  python 02_extract_features.py \
    --bam HG002.se600.minimap2.bam --ref GRCh38.fa \
    --candidates out/candidates.tsv --out out/features.tsv \
    --molecule-source readname_regex --readname-regex '#([ACGTN]+)' \
    --indel-window 10

  # barcode/UMI stored in a BAM tag instead
  python 02_extract_features.py --bam a.bam --ref ref.fa \
    --candidates out/candidates.tsv --out out/features.tsv \
    --molecule-source tag --molecule-tag BX
"""
import argparse
import re
import sys

import numpy as np
import pysam

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


def molecule_id(aln, source, tag, regex):
    if source == "tag":
        try:
            return str(aln.get_tag(tag))
        except KeyError:
            return None
    if source == "readname_regex":
        m = regex.search(aln.query_name)
        return m.group(1) if m else None
    return aln.query_name


def softclip_frac(aln):
    if aln.cigartuples is None:
        return 0.0
    sc = sum(l for op, l in aln.cigartuples if op == 4)
    return sc / (aln.query_length or 1)


def clip_frac(aln):
    """Total clip (soft S + hard H) over full read length. Unlike softclip_frac,
    this catches hard-clipped supplementary/chimeric reads (H is not in the query
    sequence, so query_length misses it)."""
    if aln.cigartuples is None:
        return 0.0
    clip = sum(l for op, l in aln.cigartuples if op in (4, 5))  # S + H
    return clip / (aln.infer_read_length() or 1)


def is_split(aln):
    """Chimeric / split alignment: supplementary or secondary flag, or has SA tag."""
    return aln.is_supplementary or aln.is_secondary or aln.has_tag("SA")


def indel_near(aln, pos0, window):
    """True if the read has an insertion/deletion within `window` ref bp of pos0.
    N (op 3, intron/refskip) is NOT an indel -> spliced alignments are not
    penalized (important for RNA SE600)."""
    if aln.cigartuples is None:
        return False
    ref = aln.reference_start
    for op, length in aln.cigartuples:
        if op in (0, 7, 8):        # M/=/X consume ref
            ref += length
        elif op == 2:              # D deletion = indel
            if abs(ref - pos0) <= window:
                return True
            ref += length
        elif op == 3:              # N intron/refskip = splice, not indel
            ref += length
        elif op == 1:              # I insertion = indel, at current ref
            if abs(ref - pos0) <= window:
                return True
        # S/H/P consume no ref
    return False


def homopolymer_run(fasta, chrom, pos0, flank=20):
    lo = max(0, pos0 - flank)
    seq = fasta.fetch(chrom, lo, pos0 + flank + 1).upper()
    i = pos0 - lo
    if i < 0 or i >= len(seq) or seq[i] not in "ACGT":
        return 0
    base = seq[i]
    l = r = i
    while l - 1 >= 0 and seq[l - 1] == base:
        l -= 1
    while r + 1 < len(seq) and seq[r + 1] == base:
        r += 1
    return r - l + 1


def mean_min(vals):
    if not vals:
        return np.nan, np.nan
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.min())


def featurize(col, chrom, pos0, ref, alt, args, regex, fasta):
    """Compute the 25 feature values from one pileup column at a candidate site."""
    alt_bq, ref_bq, alt_mapq, ref_mapq = [], [], [], []
    alt_sc, alt_readpos, alt_indel, alt_nm = [], [], [], []
    alt_clip, alt_supp = [], []
    alt_fwd = alt_rev = 0
    mol = {}  # mol_id -> [n_reads, n_alt]

    for pr in col.pileups:
        if pr.is_del or pr.is_refskip or pr.query_position is None:
            continue
        aln = pr.alignment
        qpos = pr.query_position
        base = aln.query_sequence[qpos].upper()
        if base not in "ACGT":
            continue
        bq = aln.query_qualities[qpos] if aln.query_qualities is not None else 0
        mid = molecule_id(aln, args.molecule_source, args.molecule_tag, regex) or aln.query_name
        rec = mol.setdefault(mid, [0, 0])
        rec[0] += 1
        if base == alt:
            rec[1] += 1
            alt_bq.append(bq)
            alt_mapq.append(aln.mapping_quality)
            alt_sc.append(softclip_frac(aln))
            ql = aln.query_length or (qpos + 1)
            alt_readpos.append(min(qpos, ql - 1 - qpos))
            alt_indel.append(1.0 if indel_near(aln, pos0, args.indel_window) else 0.0)
            alt_nm.append(float(aln.get_tag("NM")) if aln.has_tag("NM") else np.nan)
            alt_clip.append(clip_frac(aln))
            alt_supp.append(1.0 if is_split(aln) else 0.0)
            if aln.is_reverse:
                alt_rev += 1
            else:
                alt_fwd += 1
        elif base == ref:
            ref_bq.append(bq)
            ref_mapq.append(aln.mapping_quality)

    dp = sum(v[0] for v in mol.values())
    alt_reads = sum(v[1] for v in mol.values())
    ref_reads = len(ref_bq)
    n_mol_total = len(mol)
    n_mol_alt = sum(1 for v in mol.values() if v[1] > 0)
    n_mol_ref = n_mol_total - n_mol_alt
    vaf = alt_reads / dp if dp else np.nan
    mol_alt_fraction = n_mol_alt / n_mol_total if n_mol_total else np.nan
    per_alt_mol = [v[1] for v in mol.values() if v[1] > 0]
    within = [v[1] / v[0] for v in mol.values() if v[1] > 0]

    abq_m, abq_min = mean_min(alt_bq)
    rbq_m, _ = mean_min(ref_bq)
    amq_m, amq_min = mean_min(alt_mapq)
    rmq_m, _ = mean_min(ref_mapq)
    sc_m, _ = mean_min(alt_sc)
    rp_m, rp_min = mean_min(alt_readpos)
    strand_balance = (min(alt_fwd, alt_rev) / (alt_fwd + alt_rev)
                      if (alt_fwd + alt_rev) else np.nan)
    indel_frac = float(np.mean(alt_indel)) if alt_indel else np.nan
    nm_mean = (float(np.nanmean(alt_nm))
               if alt_nm and not np.all(np.isnan(alt_nm)) else np.nan)
    hp = homopolymer_run(fasta, chrom, pos0)
    clip_m, _ = mean_min(alt_clip)
    supp_frac = float(np.mean(alt_supp)) if alt_supp else np.nan

    return [dp, alt_reads, ref_reads, vaf,
            n_mol_total, n_mol_alt, n_mol_ref, mol_alt_fraction,
            float(np.mean(per_alt_mol)) if per_alt_mol else np.nan,
            float(np.mean(within)) if within else np.nan,
            abq_m, abq_min, rbq_m, amq_m, amq_min, rmq_m,
            strand_balance, sc_m, rp_m, rp_min,
            indel_frac, nm_mean, hp, clip_m, supp_frac]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", required=True)
    ap.add_argument("--ref", required=True, help="reference FASTA (+ .fai) for homopolymer context")
    ap.add_argument("--candidates", required=True, help="TSV from 01_make_candidates.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--molecule-source", choices=["tag", "readname_regex", "read"],
                    default="tag")
    ap.add_argument("--molecule-tag", default="BX")
    ap.add_argument("--readname-regex", default=r"#([ACGTN]+)",
                    help="regex, group(1)=barcode (for --molecule-source readname_regex)")
    ap.add_argument("--indel-window", type=int, default=10,
                    help="an alt read counts as indel-adjacent if it has an I/D within this many ref bp")
    ap.add_argument("--regions", nargs="*", default=None,
                    help="restrict to these chroms (for sharding); default = all chroms in candidates")
    ap.add_argument("--threads", type=int, default=4, help="BAM decompression threads")
    ap.add_argument("--min-base-quality", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=8000)
    args = ap.parse_args()

    regex = re.compile(args.readname_regex)
    bam = pysam.AlignmentFile(args.bam, "rb", threads=max(1, args.threads))
    fasta = pysam.FastaFile(args.ref)

    # load candidates grouped by chrom (one pileup pass per chrom, NOT one per site)
    cand = {}
    with open(args.candidates) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            cand.setdefault(f[idx["chrom"]], {})[int(f[idx["pos"]]) - 1] = (
                f[idx["ref"]], f[idx["alt"]], f[idx["label"]])

    regions = set(args.regions) if args.regions else None
    n = 0
    with open(args.out, "w") as out:
        out.write("\t".join(["chrom", "pos", "ref", "alt", "label"] + FEATURES) + "\n")
        for chrom, positions in cand.items():
            if regions and chrom not in regions:
                continue
            if not positions:
                continue
            lo, hi = min(positions), max(positions) + 1
            # ONE streaming pileup over the chrom span; featurize only candidate columns
            for col in bam.pileup(chrom, lo, hi, truncate=True,
                                  min_base_quality=args.min_base_quality,
                                  stepper="samtools", max_depth=args.max_depth):
                entry = positions.get(col.reference_pos)
                if entry is None:
                    continue
                ref, alt, label = entry
                vals = featurize(col, chrom, col.reference_pos, ref, alt, args, regex, fasta)
                out.write("\t".join([chrom, str(col.reference_pos + 1), ref, alt, label]
                                    + [f"{v:.4f}" if isinstance(v, float) else str(v)
                                       for v in vals]) + "\n")
                n += 1
            sys.stderr.write(f"[extract_features] {chrom}: done ({n} total)\n")

    sys.stderr.write(f"[extract_features] {n} sites -> {args.out}\n")


if __name__ == "__main__":
    main()
