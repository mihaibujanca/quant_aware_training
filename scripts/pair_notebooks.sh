#!/usr/bin/env bash
# Pair all notebooks under notebooks/ with py:percent scripts and sync.
# Safe to re-run (idempotent).
set -euo pipefail

NOTEBOOK_DIR="${1:-notebooks}"

if [ ! -d "$NOTEBOOK_DIR" ]; then
  echo "Directory $NOTEBOOK_DIR does not exist." >&2
  exit 1
fi

shopt -s nullglob
notebooks=("$NOTEBOOK_DIR"/*.ipynb)
shopt -u nullglob

if [ ${#notebooks[@]} -eq 0 ]; then
  echo "No .ipynb files found in $NOTEBOOK_DIR"
  exit 0
fi

for nb in "${notebooks[@]}"; do
  echo "Pairing: $nb"
  jupytext --set-formats "ipynb,py:percent" "$nb"
done

echo "Syncing all paired notebooks..."
jupytext --sync "$NOTEBOOK_DIR"/*.ipynb

echo "Done. Paired ${#notebooks[@]} notebook(s)."
