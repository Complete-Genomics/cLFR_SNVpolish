#!/usr/bin/env bash
# Claim A end-to-end: true-variant vs sequencing/mapping-error confidence model,
# trained/validated on HG002 GIAB truth. Run on the Linux server with your data.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ---- EDIT these paths ----
# DEFAULT INPUT = SE600 (single-end ~600bp, mapped with minimap2).
#   RNA/isoform: map with `minimap2 -x splice` so introns become N (not messy indels).
#   SE600 keeps low-MAPQ reads (MAPQ enters as a feature, not a hard filter).
#   Prefer running on per-molecule CONSENSUS BAM when available (cleaner CIGAR).
BAM=/path/to/HG002.se600.minimap2.bam  # SE600; UMI/barcode in read name (or BX tag)
REF=/path/to/GRCh38.fa                 # + .fai
TRUTH=/path/to/HG002_GIAB_SNV.vcf.gz   # bgzip + tabix (.tbi)
BED=/path/to/HG002_confident.bed       # verify validity for SE600-callable regions
REGIONS="chr20 chr21 chr22"            # include your held-out test chrom(s)
TEST_CHROMS="chr22"
# molecule id: readname_regex for cLFR barcodes in read name, or: --molecule-source tag --molecule-tag BX
MOL_ARGS=(--molecule-source readname_regex --readname-regex '#([ACGTN]+)')
# --------------------------

OUT="$HERE/out"
mkdir -p "$OUT"

python "$HERE/01_make_candidates.py" --bam "$BAM" --ref "$REF" \
  --truth-vcf "$TRUTH" --confident-bed "$BED" --regions $REGIONS \
  --out "$OUT/candidates.tsv" --min-alt-reads 2

python "$HERE/02_extract_features.py" --bam "$BAM" --ref "$REF" \
  --candidates "$OUT/candidates.tsv" --out "$OUT/features.tsv" \
  --indel-window 10 "${MOL_ARGS[@]}"

# three-stage ablation to ISOLATE the UMI/molecule contribution:
#   baseline      = plain pileup (dp/alt_reads/ref_reads/vaf)
#   no_molecule   = baseline + per-read mapping/quality (NO UMI)
#   all           = + UMI/molecule features
# UMI contribution = PR-AUC(all) - PR-AUC(no_molecule)
for FS in baseline no_molecule all; do
  python "$HERE/03_train_eval.py" --features "$OUT/features.tsv" \
    --out-dir "$OUT/claimA_$FS" --test-chroms $TEST_CHROMS --feature-set "$FS"
done

python - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
def pr(fs): return json.load(open(f"{out}/claimA_{fs}/metrics.json"))["pr_auc"]
b, nm, al = pr("baseline"), pr("no_molecule"), pr("all")
print(f"\n=== ablation PR-AUC ===")
print(f"baseline (pileup)        {b:.4f}")
print(f"no_molecule (+per-read)  {nm:.4f}   per-read delta = {nm-b:+.4f}")
print(f"all (+UMI/molecule)      {al:.4f}   *** UMI delta = {al-nm:+.4f} ***")
PY