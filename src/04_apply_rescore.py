#!/usr/bin/env python3
"""Step 4 (application): PER-ISOFORM SNV polishing of cLFR consensus isoforms.

Each consensus record is one molecule (one UMI). This step decides, for every
position where an isoform disagrees with the reference, whether THAT ISOFORM
should keep its base -- and writes the corrected sequence back into that
isoform's own record.

Why per-isoform (and what the previous locus-level version got wrong)
---------------------------------------------------------------------
Variant calling is a LOCUS-level task ("this sample is 0/1 here"). Isoform
correction is a SEQUENCE-level task ("this molecule's base at this position").
The earlier implementation emitted a VCF of (chrom,pos,ref,alt) with no molecule
identity and ran `bcftools consensus` against the reference -- which collapses
every isoform onto one reference backbone and cannot represent two molecules
disagreeing at the same position (exactly what a het site is). This version
keeps the molecule identity end to end and never touches `bcftools consensus`.

Two-term decision
-----------------
  p_site  = trained LightGBM score at (chrom,pos), from 03_train_eval.py.
            The model was trained on locus-level labels (GIAB SNV = true,
            confident hom-ref with alt reads = error), so it answers
            "is this a real variant site in this sample" -- and nothing more.
  molecule evidence = this molecule's own reads at that position: how many
            support the isoform's base, and how consistently.

  EDIT    A>G / T>C at a known RNA-editing site (--editing-bed) -> kept, annotated
  REVERT  p_site < threshold                      -> not a real site, write ref base
  REVERT  molecule evidence too weak              -> real site, but not on THIS molecule
  KEEP    both terms pass                         -> isoform keeps its base

The molecule term is a RULE, not a trained model. It cannot currently be
trained: at a het site there is no per-molecule truth (each molecule legitimately
carries one allele), so only hom-alt / hom-ref sites could supply labels. Tuning
knobs are --min-mol-reads / --min-mol-agreement; a trained replacement would slot
in at `decide()` without changing anything else here.

Inputs
------
  --consensus-fasta  consensus.fixRC.fasta. Records are `>{umi_id}_{chrom}`
                     (consensus_fasta.py) and survive the fixRC step unchanged.
  --consensus-bam    the SAME consensus FASTA aligned to the reference WITH
                     CIGAR. The production pipeline only emits a PAF
                     (`minimap2 -x asm20`, no -a / no -c), which has no
                     alignment to walk, so build this first:
                        minimap2 -ax asm20 -t 8 REF.fa consensus.fixRC.fasta \
                          | samtools sort -o consensus.bam && samtools index consensus.bam
  --reads-bam        the SE600 reads BAM (same one 01/02 used), for per-molecule
                     evidence. Needs .bai.
  --ref              reference FASTA (+ .fai). Mismatches are called against this
                     directly, so no MD tag is required on --consensus-bam.
  --features         02_extract_features.py output, used only to score p_site per
                     (chrom,pos). Sites absent here fall to --missing-site.
  --model            model.txt from 03_train_eval.py (feature columns = `all`).

Outputs
-------
  <prefix>.corrected.fasta   corrected isoforms, one record per input record
  <prefix>.per_isoform.tsv   one row per (isoform, position) with both terms
  <prefix>.rna_edits.tsv     EDIT-annotated RNA-editing sites
  <prefix>.sites.vcf         locus-level union of KEEP, for comparison only --
                             NOT how the FASTA is produced

Usage
-----
  minimap2 -ax asm20 -t 8 GRCh38.fa consensus.fixRC.fasta \
    | samtools sort -o consensus.bam && samtools index consensus.bam

  python 04_apply_rescore.py \
    --consensus-fasta consensus.fixRC.fasta \
    --consensus-bam   consensus.bam \
    --reads-bam       HG002.se600.minimap2.bam \
    --ref             GRCh38.fa \
    --features        out/consensus_features.tsv \
    --model           out/claimA_all/model.txt \
    --out-prefix      out/corrected \
    --molecule-source readname_regex --readname-regex '#([ACGTN]+)' \
    --threshold 0.5 --editing-bed REDIportal.hg38.bed
"""
import argparse
import re
import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import pysam

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

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
TRAILING_MATE = re.compile(r"/\d+$")


def normalize_mol_id(mid):
    """Reconcile the two molecule-id conventions in this codebase.

    consensus_fasta.py:214 does `read_id.split('#')[-1]`, which on an SE600 read
    name like `..._1752394#ATCGGTTATGTGTCC/2` keeps the `/2` mate suffix, while
    02_extract_features.py's default regex `#([ACGTN]+)` stops before it. Without
    this the consensus records would never join to their own reads.
    """
    if mid is None:
        return None
    return TRAILING_MATE.sub("", str(mid)).upper()


def molecule_id(aln, source, tag, regex):
    """Same molecule-id extraction as 02_extract_features.py, so the two agree."""
    if source == "tag":
        try:
            return str(aln.get_tag(tag))
        except KeyError:
            return None
    if source == "readname_regex":
        m = regex.search(aln.query_name)
        return m.group(1) if m else None
    return aln.query_name


def parse_record_name(name):
    """`{umi_id}_{chrom}` -> (umi_id, chrom). umi_id may itself contain '_', and
    may contain '/', so split on the LAST underscore only."""
    if "_" not in name:
        return None, None
    umi, chrom = name.rsplit("_", 1)
    return umi, chrom


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


def score_sites(features_path, model_path):
    """(chrom,pos1) -> p_true, from the locus-level model. Positions the model
    has no features for are simply absent (handled by --missing-site)."""
    df = pd.read_csv(features_path, sep="\t")
    booster = lgb.Booster(model_file=model_path)
    p = booster.predict(df[ALL_FEATURES])
    return {(c, int(pos)): float(v)
            for c, pos, v in zip(df["chrom"], df["pos"], p)}


def collect_mismatches(consensus_bam, fasta, min_mapq):
    """Walk each consensus alignment and emit every SNV-style disagreement.

    Returns per_chrom: chrom -> list of dicts, and a set of candidate positions.
    Only primary alignments are used: supplementary records carry hard clips, so
    their query positions do not index the FASTA record.
    """
    bam = pysam.AlignmentFile(consensus_bam, "rb")
    per_chrom = defaultdict(list)
    n_aln = n_skip_supp = n_skip_mapq = n_mm = 0

    for aln in bam.fetch(until_eof=True):
        if aln.is_unmapped:
            continue
        if aln.is_supplementary or aln.is_secondary:
            n_skip_supp += 1
            continue
        if aln.mapping_quality < min_mapq:
            n_skip_mapq += 1
            continue
        n_aln += 1
        chrom = aln.reference_name
        qseq = aln.query_sequence
        if qseq is None:
            continue
        umi, rec_chrom = parse_record_name(aln.query_name)
        mol = normalize_mol_id(umi)

        for qpos, refpos in aln.get_aligned_pairs(matches_only=True):
            qbase = qseq[qpos].upper()
            if qbase not in "ACGT":
                continue
            refbase = fasta.fetch(chrom, refpos, refpos + 1).upper()
            if refbase not in "ACGT" or refbase == qbase:
                continue
            per_chrom[chrom].append({
                "record": aln.query_name,
                "mol": mol,
                "chrom": chrom,
                "pos": refpos + 1,          # 1-based
                "ref": refbase,
                "alt": qbase,               # base as aligned (forward orientation)
                "qpos": qpos,               # in ALIGNED orientation
                "is_reverse": aln.is_reverse,
            })
            n_mm += 1

    sys.stderr.write(
        f"[per-isoform] consensus alignments: {n_aln} used, {n_skip_supp} supplementary/secondary "
        f"skipped, {n_skip_mapq} below --min-consensus-mapq; {n_mm} candidate mismatches\n")
    return per_chrom


def molecule_evidence(reads_bam, chrom, positions, args, regex):
    """(pos1) -> {molecule_id: {base: count}} for one chromosome.

    ONE streaming pileup over the chrom span with a dict lookup, NOT one pileup
    per site -- the same fix that took 02_extract_features.py from ~6 days to
    minutes on a dense candidate set.
    """
    want = set(positions)
    if not want:
        return {}
    lo, hi = min(want) - 1, max(want)
    ev = {}
    bam = pysam.AlignmentFile(reads_bam, "rb", threads=max(1, args.threads))
    for col in bam.pileup(chrom, lo, hi, truncate=True,
                          min_base_quality=args.min_base_quality,
                          stepper="samtools", max_depth=args.max_depth):
        pos1 = col.reference_pos + 1
        if pos1 not in want:
            continue
        per_mol = defaultdict(lambda: defaultdict(int))
        for pr in col.pileups:
            if pr.is_del or pr.is_refskip or pr.query_position is None:
                continue
            aln = pr.alignment
            base = aln.query_sequence[pr.query_position].upper()
            if base not in "ACGT":
                continue
            mid = normalize_mol_id(
                molecule_id(aln, args.molecule_source, args.molecule_tag, regex)
                or aln.query_name)
            per_mol[mid][base] += 1
        ev[pos1] = per_mol
    bam.close()
    return ev


def decide(row, p_site, mol_counts, edits, args):
    """-> (decision, mol_reads, mol_total, agreement). See module docstring."""
    if (row["chrom"], row["pos"]) in edits and (row["ref"], row["alt"]) in (("A", "G"), ("T", "C")):
        return "EDIT", np.nan, np.nan, np.nan

    if p_site is None:
        if args.missing_site == "revert":
            return "REVERT_NO_SITE", np.nan, np.nan, np.nan
        if args.missing_site == "keep":
            return "KEEP_NO_SITE", np.nan, np.nan, np.nan
        # molecule-only: fall through and judge on molecule evidence alone
    elif p_site < args.threshold:
        return "REVERT_SITE", np.nan, np.nan, np.nan

    if mol_counts is None:
        # This molecule has no reads at its own consensus position -- normally
        # impossible, so it means the molecule-id join failed. Default is to keep
        # (a broken join then shows up as a large count, not silent mass reversion).
        return ("KEEP_NO_MOL" if args.no_mol_evidence == "keep" else "REVERT_NO_MOL",
                0, 0, np.nan)

    mol_total = sum(mol_counts.values())
    mol_reads = mol_counts.get(row["alt"], 0)
    agreement = mol_reads / mol_total if mol_total else np.nan
    if mol_reads < args.min_mol_reads or (
            mol_total and agreement < args.min_mol_agreement):
        return "REVERT_MOL", mol_reads, mol_total, agreement
    return "KEEP", mol_reads, mol_total, agreement


def write_corrected_fasta(consensus_fasta, out_path, edits_by_record):
    """Stream the FASTA and apply each record's own edits, so the whole set of
    isoforms is never held in memory at once.

    Coordinates: get_aligned_pairs gives qpos in the ALIGNED orientation. When the
    consensus aligned to the minus strand, the aligned query is the reverse
    complement of the FASTA record, so the FASTA offset is (L-1-qpos) and the
    reference base must be complemented before being written.
    """
    n_rec = n_edit = n_oob = 0

    def flush(name, seq, out):
        nonlocal n_rec, n_edit, n_oob
        n_rec += 1
        todo = edits_by_record.get(name)
        if todo:
            chars = list(seq)
            L = len(chars)
            for qpos, is_reverse, refbase in todo:
                idx = (L - 1 - qpos) if is_reverse else qpos
                base = refbase.translate(COMPLEMENT) if is_reverse else refbase
                if 0 <= idx < L:
                    chars[idx] = base
                    n_edit += 1
                else:
                    n_oob += 1
            seq = "".join(chars)
        out.write(f">{name}\n{seq}\n")

    with open(consensus_fasta) as fh, open(out_path, "w") as out:
        name, chunks = None, []
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    flush(name, "".join(chunks), out)
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if name is not None:
            flush(name, "".join(chunks), out)

    if n_oob:
        sys.stderr.write(
            f"[per-isoform] WARNING: {n_oob} edits fell outside their record "
            f"(length mismatch between FASTA and BAM) and were dropped\n")
    sys.stderr.write(f"[per-isoform] wrote {n_rec} isoform records, applied {n_edit} base edits\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus-fasta", required=True,
                    help="consensus.fixRC.fasta, records `>{umi_id}_{chrom}`")
    ap.add_argument("--consensus-bam", required=True,
                    help="consensus FASTA aligned to ref WITH CIGAR (minimap2 -ax asm20 | samtools sort)")
    ap.add_argument("--reads-bam", required=True, help="SE600 reads BAM (+ .bai)")
    ap.add_argument("--ref", required=True, help="reference FASTA (+ .fai)")
    ap.add_argument("--features", required=True,
                    help="02_extract_features.py output, for the locus-level p_site")
    ap.add_argument("--model", required=True, help="model.txt from 03_train_eval.py")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="site term: KEEP requires p_site >= this")
    ap.add_argument("--min-mol-reads", type=int, default=2,
                    help="molecule term: this molecule needs >= N of its own reads carrying the base")
    ap.add_argument("--min-mol-agreement", type=float, default=0.6,
                    help="molecule term: fraction of this molecule's reads carrying the base")
    ap.add_argument("--missing-site", choices=["revert", "keep", "molecule-only"],
                    default="molecule-only",
                    help="position absent from --features (no p_site available)")
    ap.add_argument("--no-mol-evidence", choices=["keep", "revert"], default="keep",
                    help="molecule has no reads at its own position (usually a broken id join)")
    ap.add_argument("--editing-bed", default=None,
                    help="REDIportal-style BED; A>G / T>C hits are EDIT, never reverted")
    ap.add_argument("--min-consensus-mapq", type=int, default=0,
                    help="skip consensus alignments below this MAPQ")
    ap.add_argument("--molecule-source", choices=["tag", "readname_regex", "read"],
                    default="readname_regex")
    ap.add_argument("--molecule-tag", default="BX")
    ap.add_argument("--readname-regex", default=r"#([ACGTN]+)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-base-quality", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=8000)
    args = ap.parse_args()

    regex = re.compile(args.readname_regex)
    fasta = pysam.FastaFile(args.ref)
    edits = load_editing(args.editing_bed)

    sys.stderr.write("[per-isoform] scoring sites with the locus-level model...\n")
    site_p = score_sites(args.features, args.model)
    sys.stderr.write(f"[per-isoform] {len(site_p)} scored sites\n")

    per_chrom = collect_mismatches(args.consensus_bam, fasta, args.min_consensus_mapq)

    edits_by_record = defaultdict(list)
    counts = defaultdict(int)
    keep_sites = {}
    rows = []

    # chrom by chrom: the record name carries its chrom, so evidence for one
    # chromosome can be built and released without holding the genome at once.
    for chrom in sorted(per_chrom):
        cands = per_chrom[chrom]
        ev = molecule_evidence(args.reads_bam, chrom, {c["pos"] for c in cands}, args, regex)
        for c in cands:
            p = site_p.get((c["chrom"], c["pos"]))
            mol_counts = ev.get(c["pos"], {}).get(c["mol"])
            decision, mol_reads, mol_total, agreement = decide(c, p, mol_counts, edits, args)
            counts[decision] += 1
            if decision.startswith("REVERT"):
                edits_by_record[c["record"]].append((c["qpos"], c["is_reverse"], c["ref"]))
            elif p is not None:
                keep_sites[(c["chrom"], c["pos"], c["ref"], c["alt"])] = p
            rows.append({
                "record": c["record"], "mol": c["mol"], "chrom": c["chrom"], "pos": c["pos"],
                "ref": c["ref"], "alt": c["alt"], "is_reverse": int(c["is_reverse"]),
                "p_site": p if p is not None else np.nan,
                "mol_reads": mol_reads, "mol_total": mol_total, "mol_agreement": agreement,
                "decision": decision,
            })
        sys.stderr.write(f"[per-isoform] {chrom}: {len(cands)} candidates scored\n")

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.out_prefix}.per_isoform.tsv", sep="\t", index=False)
    out[out["decision"] == "EDIT"].to_csv(
        f"{args.out_prefix}.rna_edits.tsv", sep="\t", index=False)

    write_corrected_fasta(args.consensus_fasta,
                          f"{args.out_prefix}.corrected.fasta", edits_by_record)

    # Locus-level union of kept calls. Provided for comparison against the old
    # behaviour and against a standard caller -- the corrected FASTA above is NOT
    # produced from this file.
    with open(f"{args.out_prefix}.sites.vcf", "w") as v:
        v.write("##fileformat=VCFv4.2\n")
        v.write('##INFO=<ID=PT,Number=1,Type=Float,Description="model P(true variant site)">\n')
        v.write('##INFO=<ID=NISO,Number=1,Type=Integer,Description="isoforms keeping this call">\n')
        v.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        niso = out[out["decision"] == "KEEP"].groupby(
            ["chrom", "pos", "ref", "alt"]).size().to_dict()
        for (c, pos, ref, alt), p in sorted(keep_sites.items()):
            qual = int(min(99, round(-10 * np.log10(max(1e-9, 1 - p)))))
            v.write(f"{c}\t{pos}\t.\t{ref}\t{alt}\t{qual}\tPASS\t"
                    f"PT={p:.3f};NISO={niso.get((c, pos, ref, alt), 0)}\n")

    total = sum(counts.values())
    sys.stderr.write(f"[per-isoform] {total} (isoform,position) decisions: "
                     + " ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")
    if counts.get("KEEP_NO_MOL", 0) > 0.5 * max(1, total):
        sys.stderr.write(
            "[per-isoform] WARNING: most calls had no molecule evidence -- the consensus "
            "record ids are probably not joining to the reads BAM. Check --molecule-source "
            "/ --readname-regex against the read names in --reads-bam.\n")
    sys.stderr.write(f"[per-isoform] corrected isoforms: {args.out_prefix}.corrected.fasta\n")


if __name__ == "__main__":
    main()
