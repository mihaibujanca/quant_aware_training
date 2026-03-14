# Cycle 004: Benchmark Lock-In and Asset Audit

## Verdict

**Outcome: confirmed**

Item 9 asked for a benchmark ladder that distinguishes:
- an exact open teacher+training-data benchmark
- a proxy benchmark
- clear policy for what counts and what does not count

This audit confirms that the repo has:
- one **exact-open benchmark candidate** with a concrete public acquisition/training path: `karpathy/nanochat` base checkpoints + `climbmix-400b-shuffle`
- one **explicit proxy benchmark** already runnable from local assets: cached Hugging Face `gpt2` + local proxy artifact `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`

The key limitation is local availability:
- **exact-open benchmark is not locally runnable right now**
- **proxy benchmark is runnable right now**

## Local Asset Audit

### Present locally
- Repo code for transformer FFN compression:
  - `aleph/lm_geometric_compression.py`
  - `experiments/nanochat_ffn_compression.py`
  - `docs/nanochat_ffn_compression.md`
- Proxy result artifact:
  - `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`
- Hugging Face cache:
  - `~/.cache/huggingface/hub/models--gpt2`

### Not found locally
- no `nanochat` checkout under `/Users/mihai/Projects` or `/Users/mihai` (searched to depth 4)
- no `climbmix` parquet directory under `/Users/mihai`
- no local `d24` / `d26` checkpoint directories or obvious checkpoint files under `/Users/mihai`

## Exact-Open Benchmark

### Benchmark definition
- **Teacher family:** `karpathy/nanochat` base model checkpoints
- **Preferred tags:** `d24` / `d26`
- **Data family:** `karpathy/climbmix-400b-shuffle`
- **Compression scope:** FFN-only replacement inside the transformer

### Why this counts
- teacher code is open: official public repo `karpathy/nanochat`
- dataset source is open: dataset utility downloads from Hugging Face `karpathy/climbmix-400b-shuffle`
- local loader in this repo is wired to that exact repo layout

### Public acquisition / training path
- Official nanochat README documents:
  - `speedrun.sh` as the quick-start path for the smaller `$100` tier
  - a `d26` path as the next larger model, including `python -m nanochat.dataset -n 450` and `torchrun ... --depth=26 --device_batch_size=16`
- Official `nanochat.dataset` sets:
  - base URL: `https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main`
  - local data directory: `base_data_climbmix`
- Official `nanochat.checkpoint_manager.load_model(...)` resolves checkpoints from:
  - `base_checkpoints/{model_tag}`

### Concrete local path convention
- expected nanochat repo path for this repo:
  - passed as `--nanochat-repo /path/to/nanochat`
- expected dataset directory:
  - `/path/to/nanochat` runtime base dir + `base_data_climbmix`
  - repo-local example in our docs: `--parquet-dir /path/to/base_data_climbmix`
- expected checkpoint directory:
  - nanochat base dir + `base_checkpoints/d24` or `base_checkpoints/d26`

### Local readiness
- **Current status:** not locally runnable
- blocker 1: no local `nanochat` checkout
- blocker 2: no local `base_data_climbmix`
- blocker 3: no local base checkpoint tag such as `d24` or `d26`

## Proxy Benchmark

### Benchmark definition
- **Teacher:** Hugging Face pretrained `gpt2`
- **Probe/eval corpus:** clearly marked proxy corpus
- **Current concrete artifact:** `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`

### Why this counts
- the repo loader explicitly supports `hf_pretrained` with default model `gpt2`
- Hugging Face `gpt2` is cached locally
- there is already a completed FFN-only compression artifact using this path

### What the existing proxy artifact shows
- teacher: `gpt2`
- model family: `gpt2`
- dataset label in artifact: `climbmix-400b-shuffle`
- geometric method: very poor full-model behavior despite `11.98x` FFN compression
- distillation baseline materially outperforms the current analytic method on the same proxy setup

This artifact is useful as a **sanity point** only. It does **not** answer the exact-open benchmark question.

## Smoke / Debug Benchmark

- `hf_random` is a valid smoke path only
- documented in `docs/nanochat_ffn_compression.md`
- used by `tests/test_lm_geometric_compression.py`
- this does **not** count as a serious compression benchmark

## What Counts / What Does Not Count

### Counts
- **Exact-open benchmark:** local nanochat checkout + local nanochat base checkpoints + local `climbmix-400b-shuffle` shards, with FFN-only compression evaluation
- **Proxy benchmark:** `gpt2` loaded via Hugging Face with a clearly marked proxy corpus and explicit proxy labeling in results
- **Smoke/debug benchmark:** `hf_random` for unit tests and pipeline checks only

### Does not count
- `gpt2` proxy results presented as if they answered the trained-model-plus-training-data question
- `hf_random` or toy corpora presented as scientific evidence for transformer FFN compression viability
- runs that compress attention or whole-model components before single-block and multi-block FFN evidence exists

## Immediate Recommendation

The benchmark ladder should now be treated as:
1. **Exact-open target:** nanochat base + `climbmix-400b-shuffle` when assets are available
2. **Current runnable benchmark:** cached `gpt2` proxy
3. **Smoke only:** `hf_random`

With item 9 resolved, the next serious cycle should be item 10: single-block swap localization curves, using `gpt2` as the current teacher unless nanochat assets are added locally first.

## Sources

Local repo sources:
- `docs/nanochat_ffn_compression.md`
- `aleph/lm_geometric_compression.py`
- `experiments/nanochat_ffn_compression.py`
- `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`

Primary public sources:
- official nanochat repo: https://github.com/karpathy/nanochat
- official README (quick start + d26 guidance): https://github.com/karpathy/nanochat
- checkpoint loader: https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/checkpoint_manager.py
- dataset loader: https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/dataset.py
