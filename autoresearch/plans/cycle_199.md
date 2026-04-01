# Cycle 199 Implementation Plan: K-sweep for near-rank-1-input blocks

## Overview

Test K={8,16,32,64,128,256,512} for topk+lstsq compression on BLOCKS=[1,4,7,11].
Cycle 198 found blocks 4,7,11 have input_rank_90=1; block 1 has input_rank=207.
Hypothesis: if block input lives on a 1D manifold, K=8-32 may be sufficient.

**Script**: `experiments/k_sweep_rank1_cycle199.py`
**Output**: `autoresearch/results/cycle_199_k_sweep.json`

---

## Config

```python
K_VALUES = [8, 16, 32, 64, 128, 256, 512]
BLOCKS = [1, 4, 7, 11]  # established 4-block set; blocks 4,7,11 have rank-1 input
N_SEQS = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_199_k_sweep.json"
```

---

## Implementation

### Step 1: Collect MLP I/O

```python
def collect_mlp_io(model, calib_ids, block_ids, device):
    """Collect (ln_out, proj_out) for each block via hooks on block.mlp."""
    ln_outs = {b: [] for b in block_ids}
    proj_outs = {b: [] for b in block_ids}
    hooks = []
    for b in block_ids:
        def mk_hook(bb):
            def fn(m, inp, out):
                # inp[0]: LN-2 output = MLP input, (1, T, 768)
                # out: MLP output, (1, T, 768)
                ln_outs[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
                proj_outs[bb].append(out.detach().float().cpu().reshape(-1, 768))
            return fn
        hooks.append(model.transformer.h[b].mlp.register_forward_hook(mk_hook(b)))
    with torch.no_grad():
        for ids in calib_ids:
            model(ids)
    for h in hooks:
        h.remove()
    for b in block_ids:
        ln_outs[b] = torch.cat(ln_outs[b], dim=0)   # (N_tokens, 768)
        proj_outs[b] = torch.cat(proj_outs[b], dim=0) # (N_tokens, 768)
    return ln_outs, proj_outs
```

### Step 2: CompressedMLP

```python
class CompressedMLP(nn.Module):
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight)  # (768, K)
        self.register_buffer("fc_bias", fc_bias)       # (K,)
        self.register_buffer("W_out", W_out)           # (768, K)
        self.register_buffer("b_out", b_out)           # (768,)

    def forward(self, x):  # x: (B, T, 768) — LN-2 output
        act = F.gelu(x @ self.fc_weight + self.fc_bias)  # (B, T, K)
        return act @ self.W_out.T + self.b_out            # (B, T, 768)
```

### Step 3: Fit one block at given K

```python
def fit_block(block, ln_out_cpu, proj_out_cpu, K, device):
    """Fit topk+lstsq compressed MLP for one block at given K."""
    # c_fc.weight shape: (768, 3072) — GPT-2 Conv1D layout, forward = x @ weight + bias
    W_fc = block.mlp.c_fc.weight.data.float().cpu()    # (768, 3072)
    b_fc = block.mlp.c_fc.bias.data.float().cpu()      # (3072,)
    W_proj = block.mlp.c_proj.weight.data.float().cpu() # (3072, 768)
    b_proj = block.mlp.c_proj.bias.data.float().cpu()   # (768,)

    # Score channels by activation std × c_proj output norm
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)  # (N_tokens, 3072)
    scores = act_all.std(0) * W_proj.norm(dim=1)  # (3072,)
    topk_idx = scores.topk(K).indices             # (K,)

    W_fc_sub = W_fc[:, topk_idx]   # (768, K)
    b_fc_sub = b_fc[topk_idx]      # (K,)
    act_sub = act_all[:, topk_idx] # (N_tokens, K)

    # lstsq: act_sub @ W_out.T + b_out ≈ proj_out
    # design = [act_sub | ones] → shape (N, K+1), target = proj_out (N, 768)
    ones = torch.ones(act_sub.shape[0], 1)
    design = torch.cat([act_sub, ones], dim=1)  # (N, K+1)
    sol = torch.linalg.lstsq(design, proj_out_cpu, rcond=None).solution  # (K+1, 768)
    W_out = sol[:K].T   # (768, K)
    b_out = sol[K]      # (768,)

    return CompressedMLP(W_fc_sub, b_fc_sub, W_out, b_out)
```

### Step 4: Replace blocks and eval PPL

```python
def eval_ppl(model, calib_ids, device):
    """Compute perplexity on calibration sequences (same as fitting data for simplicity)."""
    # Use held-out eval tokens instead — load fresh from validation split
    ...
```

Actually, use a **separate** eval batch from the training batch to avoid overfitting measurement.

```python
# Eval: load N_EVAL_SEQS=100 separate sequences from the same wikitext-2 validation split
# Use offset to avoid overlap with calibration sequences.
```

### Step 5: K-sweep loop

```python
results = []
for K in K_VALUES:
    # Load fresh copy of model
    model_k = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_k.eval()

    # Collect I/O from float model on calibration data
    ln_outs, proj_outs = collect_mlp_io(model_k, calib_ids, BLOCKS, DEVICE)

    # Fit and replace each block
    for b in BLOCKS:
        compressed = fit_block(model_k.transformer.h[b], ln_outs[b], proj_outs[b], K, DEVICE)
        compressed = compressed.to(DEVICE)
        model_k.transformer.h[b].mlp = compressed

    ppl = eval_ppl(model_k, eval_ids, DEVICE)
    print(f"K={K:4d}: PPL={ppl:.3f}")
    results.append({"K": K, "ppl": float(ppl)})
```

---

## PPL Evaluation

```python
def eval_ppl(model, eval_ids, device):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)
```

---

## Output JSON

```json
{
  "cycle": 199, "item": 209,
  "verdict": "...",
  "baseline_k512_ppl": 30.089,
  "k_sweep": [
    {"K": 8, "ppl": ...},
    {"K": 16, "ppl": ...},
    {"K": 32, "ppl": ...},
    {"K": 64, "ppl": ...},
    {"K": 128, "ppl": ...},
    {"K": 256, "ppl": ...},
    {"K": 512, "ppl": ...}
  ],
  "min_k_within_5pct": ...,
  "config": {...}
}
```

---

## Verdict Logic

```python
# Threshold: 5% above K=512 PPL = 30.089 * 1.05 = 31.59
THRESHOLD_PPL = baseline_ppl * 1.05
within_5pct = [r for r in k_sweep if r["ppl"] <= THRESHOLD_PPL]
min_k_within = min(r["K"] for r in within_5pct) if within_5pct else None

if min_k_within is not None and min_k_within <= 64:
    verdict = "CONFIRMED"   # rank-1 input enables small K
elif min_k_within is None or min_k_within >= 256:
    verdict = "REFUTED"     # no significant benefit
else:
    verdict = "INCONCLUSIVE"  # benefit but at K>64
```

---

## Notes

- Calibration sequences (N_SEQS=200, SEQ_LEN=128): first 200 matching sequences from wikitext-2 validation
- Eval sequences: next 100 sequences (offset to avoid overlap)
- Float baseline PPL at K=512 should reproduce ~30.089; if it doesn't, report the actual baseline
- Block replacement order: all 4 blocks replaced before eval (not progressive)
