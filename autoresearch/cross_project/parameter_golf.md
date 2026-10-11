# Cross-Project Reference: parameter-golf

**Purpose**: any qat agenda item whose verdict requires loading a parameter-golf checkpoint, evaluating its activations, or comparing its geometry to GPT-2's must follow the invocation pattern in this document. The qat autoresearch loop has full filesystem access (the runner is launched with `--dangerously-skip-permissions`) so this is **not a permission problem** — it is an environment-isolation problem. parameter-golf uses MLX, qat uses PyTorch; the two cannot share an in-process Python interpreter. Cross-project work is done by **subprocess invocation** of the parameter-golf venv from a qat experiment script.

**Brief stage**: list this file path under `# Key Files`. **Plan stage**: include `cat autoresearch/cross_project/parameter_golf.md` as Step 0 of the plan setup. **Exec stage**: read this file before writing or running any code that touches `/Users/mihai/Projects/parameter-golf/`.

---

## 1. Locations

| What | Path |
|---|---|
| Project root | `/Users/mihai/Projects/parameter-golf/` |
| Python venv (has `mlx 0.31.1`, `brotli 1.2.0`, `numpy 2.4.3`) | `/Users/mihai/Projects/parameter-golf/.venv/bin/python` |
| Model class source | `/Users/mihai/Projects/parameter-golf/train_gpt_mlx.py` (defines `GPT`, `Block`, `MLP`, `Hyperparameters`) |
| Checkpoint loader (reuse this) | `/Users/mihai/Projects/parameter-golf/local_scripts/analyze_checkpoint.py` (`load_model`, `load_val_tokens`, `eval_bpb`) |
| FineWeb val data | `/Users/mihai/Projects/parameter-golf/data/datasets/` (loaded via `load_data_shard` from `train_gpt_mlx`) |
| Standard tokenizer | sentencepiece, vocab=1024 (NOT GPT-2's 50257) |
| Checkpoints (uncompressed `.npz`) | `/Users/mihai/Projects/parameter-golf/logs/*.npz` |
| Checkpoints (`.br` brotli) | `/Users/mihai/Projects/parameter-golf/*.br` (need brotli decompress before `np.load`) |

**Recommended canonical checkpoint** for cross-project items: `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz`. 13 blocks, 640 dim, ReLU² activation (no `mlp.gate.weight`), variable `mlp_mult` per block (early blocks 1920 hidden = 3×, late blocks 1280 hidden = 2×), vocab 1024.

---

## 2. Why subprocess instead of import

| | qat | parameter-golf |
|---|---|---|
| Framework | PyTorch | MLX (Apple) |
| Venv | `quant_aware_training/.venv` | `parameter-golf/.venv` |
| Tensor type | `torch.Tensor` | `mlx.core.array` |
| Module class | `torch.nn.Module` | `mlx.nn.Module` |
| Hooks | `register_forward_hook` available | **no hooks** — must monkey-patch `__call__` or modify forward |
| Tokenizer | wikitext-2 with GPT-2 BPE (50257) | sentencepiece (1024) |

The two stacks cannot share a process. Importing `mlx` from inside the qat venv fails (`ModuleNotFoundError: mlx`). Importing `torch` from inside the pgolf venv may not be available either. **The orchestration layer is qat; the measurement layer is pgolf**, with subprocess as the bridge.

---

## 3. Canonical invocation pattern

Two scripts per cycle:

### qat side: `experiments/pgolf_<measurement>_cycleNNN.py`

```python
import json, subprocess, sys
from pathlib import Path

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_<measurement>.py"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"

def main():
    proc = subprocess.run(
        [PGOLF_PYTHON, PGOLF_SCRIPT, "--checkpoint", CHECKPOINT, "--n-tokens", "25600"],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"pgolf script failed: {proc.returncode}")

    result = json.loads(proc.stdout)  # pgolf script prints JSON to stdout
    out_dir = Path("autoresearch/results")
    cycle = NNN  # set per cycle
    (out_dir / f"cycle_{cycle}_pgolf_<measurement>.json").write_text(json.dumps(result, indent=2))
    # Build summary.md from result fields
    write_summary(result, out_dir / f"cycle_{cycle}_summary.md")

if __name__ == "__main__":
    main()
```

### pgolf side: `parameter-golf/local_scripts/pgolf_<measurement>.py`

```python
#!/usr/bin/env python3
"""Cross-project measurement invoked by quant_aware_training autoresearch.
Prints JSON to stdout. Argument: --checkpoint <path>."""
import argparse, json, sys
from pathlib import Path

# Reuse the existing loader
sys.path.insert(0, str(Path(__file__).parent))
from analyze_checkpoint import load_model, load_val_tokens
from train_gpt_mlx import Hyperparameters, MLP, Block

import mlx.core as mx
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-tokens", type=int, default=25600)
    args = ap.parse_args()

    hp = Hyperparameters()
    model = load_model(args.checkpoint, hp)
    tokens = load_val_tokens(hp, args.n_tokens)

    # ... measurement code ...

    print(json.dumps(result))  # JSON to stdout

if __name__ == "__main__":
    main()
```

The qat script is the cycle artifact (lives in `experiments/`, runs from qat's CWD). The pgolf script is permanent infrastructure (lives in `parameter-golf/local_scripts/`, reusable across cycles).

---

## 4. Capturing intermediate activations in MLX

MLX has no `register_forward_hook`. The cleanest pattern is to **monkey-patch** the `MLP.__call__` (or `Block.__call__`) globally before running the forward pass. After capture, restore.

```python
import mlx.core as mx
from train_gpt_mlx import MLP

# Capture mlp.proj outputs per block (= MLP.__call__ return value)
captured: dict[int, list[mx.array]] = {}

original_mlp_call = MLP.__call__
def hook_mlp_call(self, x, slope=0.5):
    out = original_mlp_call(self, x, slope=slope)
    mx.eval(out)  # force materialization
    captured.setdefault(id(self), []).append(out)
    return out

MLP.__call__ = hook_mlp_call
try:
    # forward pass over the calib slice in batches
    for batch_start in range(0, len(tokens) - SEQ_LEN, BATCH_SIZE * SEQ_LEN):
        batch = tokens[batch_start : batch_start + BATCH_SIZE * SEQ_LEN].reshape(BATCH_SIZE, SEQ_LEN)
        _ = model(mx.array(batch))
finally:
    MLP.__call__ = original_mlp_call

# captured[id(model.blocks[l].mlp)] is a list of (B, T, dim) mlx arrays
# Per block: stack over batches, reshape to (N, dim), compute Y^T Y / N
```

**Identification of which block is which**: iterate `model.blocks` and use `id(block.mlp)` as the key. Map back via `block_index = {id(model.blocks[l].mlp): l for l in range(len(model.blocks))}`.

**Variable mlp_mult**: each block's `MLP.proj.weight` may have a different shape. Don't assume a fixed hidden dim. Loop over blocks individually for SVD/covariance work.

---

## 5. Worked example: output-covariance eff_rank census

This is the parameter-golf analog of qat's cycle 278. The full script lives at `parameter-golf/local_scripts/pgolf_eff_rank_census.py` once written.

```python
# pgolf side
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from analyze_checkpoint import load_model, load_val_tokens
from train_gpt_mlx import Hyperparameters, MLP

import mlx.core as mx
import numpy as np

def eff_rank_from_eigvals(eigvals: np.ndarray) -> float:
    p = eigvals / eigvals.sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-tokens", type=int, default=25600)
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    hp = Hyperparameters()
    model = load_model(args.checkpoint, hp)
    tokens = load_val_tokens(hp, args.n_tokens)

    # Identify blocks
    block_idx = {id(model.blocks[l].mlp): l for l in range(len(model.blocks))}
    captured = {l: [] for l in range(len(model.blocks))}

    original_mlp_call = MLP.__call__
    def hook(self, x, slope=0.5):
        out = original_mlp_call(self, x, slope=slope)
        mx.eval(out)
        captured[block_idx[id(self)]].append(np.array(out))
        return out

    MLP.__call__ = hook
    try:
        seq_len = args.seq_len
        n_seqs = args.n_tokens // seq_len
        for s in range(n_seqs):
            batch = tokens[s * seq_len : (s + 1) * seq_len][None, :]  # (1, T)
            _ = model(mx.array(batch))
    finally:
        MLP.__call__ = original_mlp_call

    # Compute per-block eff_rank
    profile = {}
    for l, chunks in captured.items():
        # chunks is a list of (1, T, dim) arrays
        Y = np.concatenate([c.reshape(-1, c.shape[-1]) for c in chunks], axis=0)  # (N, dim)
        C = (Y.T @ Y) / Y.shape[0]
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.sort(eigvals)[::-1].clip(min=0)
        profile[l] = {
            "eff_rank": eff_rank_from_eigvals(eigvals),
            "rank_95": int(np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), 0.95) + 1),
            "rank_99": int(np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), 0.99) + 1),
            "top5_eigvals": eigvals[:5].tolist(),
            "n_tokens": int(Y.shape[0]),
            "dim": int(Y.shape[1]),
        }

    print(json.dumps({"checkpoint": args.checkpoint, "n_blocks": len(model.blocks), "profile": profile}))

if __name__ == "__main__":
    main()
```

The qat-side wrapper subprocess-invokes this and writes the result to `autoresearch/results/cycle_NNN_pgolf_eff_rank.json`. Verdict thresholds: same as qat cycle 278 (any block with eff_rank ≤ 20 → rank-collapse generalizes; all blocks ≥ 100 → rank-collapse is GPT-2-specific or overparameterization-driven).

---

## 6. Gotchas

1. **Vocab is 1024, not 50257.** Don't try to compare token-level boost lists directly to qat's GPT-2 measurements. The unembedding is `(640, 1024)` here, not `(768, 50257)`.
2. **Variable mlp_mult per block.** Some blocks have hidden=1920, others 1280. SVD/covariance work must be per-block; don't stack across blocks.
3. **No bias on `mlp.fc` or `mlp.proj`.** parameter-golf MLPs are bias-free. Cycle 280-style retrain recipes that add a `bias` parameter need to be adapted (or the bias is just left out).
4. **`mlp_scale` and `resid_mix`.** Each block has learned per-channel scales (`mlp_scale ∈ R^640`) and a learned residual mix (`resid_mix ∈ R^(2,640)`). The MLP output is multiplied by `mlp_scale` before being added to the residual stream. **The "mlp.proj output" you capture is BEFORE this scaling.** Decide per-experiment whether to capture pre- or post-scale; pre-scale is the analog of GPT-2's `c_proj` output.
5. **Encoder/decoder split with skip connections.** The GPT class has `num_encoder_layers + num_decoder_layers = num_layers`. Encoder outputs are stored and added back to decoder inputs via learned `skip_weights`. For an eff_rank census this doesn't matter (you're capturing per-block MLP outputs regardless of skip), but for any analysis that depends on residual stream trajectory, the skip connections change the geometry.
6. **`mx.eval(out)` is required** to force materialization. MLX is lazy by default; without `mx.eval` the captured arrays are unrealized graph nodes that don't survive the function return.
7. **`.br` files require brotli decompression first.** `np.load` does not handle brotli. Pattern: `arr = np.load(io.BytesIO(brotli.decompress(open(path, 'rb').read())), allow_pickle=True)`. Prefer `.npz` files in `logs/` when available.
8. **Float type**: parameter-golf uses bfloat16 for compute (`COMPUTE_DTYPE = mx.bfloat16`). Cast to float32 before passing to numpy: `np.array(arr.astype(mx.float32))`. The `analyze_checkpoint.load_model` already handles bfloat16 casting on load.
9. **Subprocess timeout**: a forward pass over 25,600 tokens at seq_len 1024 is ~25 batches, ~30 seconds wall. Use `timeout=600` in subprocess.run for safety. Longer measurements (curvature residual + SVD + projection): use `timeout=1800`.

---

## 7. Result schema

The qat-side wrapper writes two files:
- `autoresearch/results/cycle_NNN_pgolf_<measurement>.json` — full JSON output from the pgolf script
- `autoresearch/results/cycle_NNN_summary.md` — verdict, key numbers, comparisons to qat baselines (cycle 278 GPT-2 eff_rank profile, cycle 280 LASER result, etc.)

The summary.md MUST cite both qat baselines AND the parameter-golf checkpoint name + git commit, so the comparison is reproducible.

---

## 8. Adding a new cross-project measurement

To add a new pgolf measurement type:
1. Write a new pgolf-side script at `parameter-golf/local_scripts/pgolf_<name>.py` following the section 5 template.
2. Write a qat-side wrapper at `experiments/pgolf_<name>_cycleNNN.py` following the section 3 template.
3. Add an agenda item that:
   - Has `**REQUIRED READING**: autoresearch/cross_project/parameter_golf.md` as the first line of the item body
   - Has `**Setup Step 0**: cat autoresearch/cross_project/parameter_golf.md before running` in the setup
   - Specifies the wrapper script path
   - Has explicit verdict thresholds, baselines from qat cycles
4. The runner picks up the item, the brief lists this doc under Key Files, the planner copies the cat command into Step 0 of the plan, the exec runs Step 0 then runs the wrapper.
