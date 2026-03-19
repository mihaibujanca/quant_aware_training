# Nanochat FFN Compression

This is the new transformer-first compression path.

Scope:
- teacher: GPT-style causal LM loaded from Hugging Face or a local nanochat checkout
- data: text file chunks or nanochat-style parquet shards
- compression target: FFN / MLP blocks only
- methods: analytic geometric compression, matched-budget distillation baseline, descriptor-autoencoder compressor

Main code:
- `aleph/lm_geometric_compression.py`
- `experiments/nanochat_ffn_compression.py`

Example, offline smoke path:

```bash
.venv/bin/python experiments/nanochat_ffn_compression.py \
  --teacher-source hf_random \
  --text-path data/shakespeare.txt \
  --sequence-length 64 \
  --probe-train-seqs 32 \
  --probe-holdout-seqs 16 \
  --eval-val-seqs 16 \
  --hidden-fraction 0.5 \
  --min-hidden 16 \
  --max-hidden 64
```

Example, nanochat path:

```bash
.venv/bin/python experiments/nanochat_ffn_compression.py \
  --teacher-source nanochat_local \
  --nanochat-repo /path/to/nanochat \
  --model-tag d24 \
  --parquet-dir /path/to/base_data_climbmix \
  --sequence-length 256 \
  --probe-train-seqs 64 \
  --probe-holdout-seqs 32 \
  --eval-val-seqs 32 \
  --hidden-fraction 0.25 \
  --selection-mode topk
```

Notes:
- parquet ingestion requires `pyarrow`
- `hf_random` is only for unit tests and fast local debugging
- the no-training path uses white-box probing plus closed-form least squares; it does not run SGD on the compressed model
