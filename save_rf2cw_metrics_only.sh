#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$PWD}"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$REPO_ROOT/exports"
TMPDIR="$(mktemp -d)"
mkdir -p "$OUTDIR"

echo "Collecting metrics.jsonl from results/rf2cw_* ..."

find results -type f -name "metrics.jsonl" | grep '/rf2cw_' | while read -r f; do
  rel="${f#results/}"
  dest="$TMPDIR/$rel"
  mkdir -p "$(dirname "$dest")"
  cp "$f" "$dest"
done

ARCHIVE="$OUTDIR/rf2cw_metrics_only_${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$TMPDIR" .
rm -rf "$TMPDIR"

echo
echo "Done."
echo "Archive created:"
echo "  $ARCHIVE"
echo
echo "To download it from your LOCAL machine, run:"
echo "  scp <user>@<cluster>:$ARCHIVE ."
