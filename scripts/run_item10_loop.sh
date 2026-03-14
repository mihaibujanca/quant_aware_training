#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec env PYTHONUNBUFFERED=1 MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/nanochat_single_block_localization.py \
  --teacher-source hf_pretrained \
  --hf-model gpt2 \
  --dataset-name proxy_shakespeare \
  --text-path data/shakespeare.txt \
  --hidden-fractions 0.125,0.0625 \
  --min-hidden 64 \
  --max-hidden 512 \
  --distill-steps 60 \
  --batch-size 4 \
  --max-tokens-per-layer 1024 \
  --max-repeats 0 \
  --output-dir runs/nanochat_single_block_localization_overnight_main
