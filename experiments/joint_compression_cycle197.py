"""
Cycle 197 — Item 208: Block-pair joint compression via sequential progressive fitting.
Compare:
  Strategy A: all blocks fitted on FLOAT model I/O (baseline)
  Strategy B: sequential fitting — each block fitted on the COMPRESSED model's I/O
Tests whether fitting each block on actual compressed inputs (rather than float inputs)
reduces cascade PPL.
"""
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [1, 4, 7, 11]
K = 512
N_SEQS_CALIB = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_197_joint_compression.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model and data ───────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [r["text"] for r in train_ds if len(r["text"].strip()) > 50]
calib_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        calib_ids.append(enc["input_ids"][:, :SEQ_LEN].to(DEVICE))
    if len(calib_ids) >= N_SEQS_CALIB:
        break

val_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
val_text = "\n\n".join([r["text"] for r in val_ds])
val_ids = tokenizer(val_text, return_tensors="pt")["input_ids"].to(DEVICE)

print(f"Calibration: {len(calib_ids)} seqs × {SEQ_LEN} = {len(calib_ids) * SEQ_LEN} tokens")


def compute_ppl(m, max_tokens=50000, stride=512, max_length=1024):
    m.eval()
    ids = val_ids[:, :max_tokens]
    T = ids.shape[1]
    nll_sum, n_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, T - 1, stride):
            begin = max(0, i - max_length + stride)
            end = min(T, i + stride)
            chunk = ids[:, begin:end + 1]
            tl = end - i
            out = m(chunk)
            logits = out.logits[:, -tl - 1:-1, :]
            targets = chunk[:, -tl:]
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                  targets.reshape(-1), reduction="sum")
            nll_sum += nll.item()
            n_tokens += tl
    return float(torch.exp(torch.tensor(nll_sum / n_tokens)))


class CompressedMLP(nn.Module):
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        # fc_weight: (768, K), fc_bias: (K,), W_out: (768, K), b_out: (768,)
        self.register_buffer("fc_weight", fc_weight)
        self.register_buffer("fc_bias", fc_bias)
        self.register_buffer("W_out", W_out)
        self.register_buffer("b_out", b_out)

    def forward(self, x):  # x: (B, T, 768)
        act = F.gelu(x @ self.fc_weight + self.fc_bias)  # (B, T, K)
        return act @ self.W_out.T + self.b_out            # (B, T, 768)


def collect_mlp_io(m, calib_ids, block_ids):
    """Collect MLP input and output for specified blocks from model m."""
    mlp_in = {b: [] for b in block_ids}
    mlp_out = {b: [] for b in block_ids}
    hooks = []

    for b in block_ids:
        def mk(bb):
            def fn(module, inp, out):
                mlp_in[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
                mlp_out[bb].append(out.detach().float().cpu().reshape(-1, 768))
            return fn
        hooks.append(m.transformer.h[b].mlp.register_forward_hook(mk(b)))

    with torch.no_grad():
        for ids in calib_ids:
            m(ids)

    for h in hooks:
        h.remove()

    for b in block_ids:
        mlp_in[b] = torch.cat(mlp_in[b], dim=0)
        mlp_out[b] = torch.cat(mlp_out[b], dim=0)

    return mlp_in, mlp_out


def fit_block(block, ln_out_cpu, proj_out_cpu, K, device):
    """Fit topk+lstsq compressed MLP. Returns CompressedMLP."""
    with torch.no_grad():
        ln_out = ln_out_cpu.to(device)
        proj_out = proj_out_cpu.to(device)

        # Channel selection
        fc_full = ln_out @ block.mlp.c_fc.weight + block.mlp.c_fc.bias
        act = F.gelu(fc_full)
        c_proj_w = block.mlp.c_proj.weight  # (3072, 768)
        scores = act.std(0) * c_proj_w.norm(dim=1) * (act.mean(0).abs() + 1e-6)
        idx = scores.topk(K).indices

        # Subsetted input weights
        fc_w = block.mlp.c_fc.weight[:, idx].contiguous().cpu()  # (768, K)
        fc_b = block.mlp.c_fc.bias[idx].contiguous().cpu()        # (K,)

        # lstsq with bias
        act_sub = act[:, idx].float()  # (N, K)
        ones = torch.ones(act_sub.shape[0], 1, device=device)
        design = torch.cat([act_sub, ones], dim=1)  # (N, K+1)
        sol, *_ = torch.linalg.lstsq(design.cpu(), proj_out.float().cpu(), rcond=None)
        W_out = sol[:K].T.contiguous()  # (768, K)
        b_out = sol[K].contiguous()      # (768,)

    return CompressedMLP(fc_w.to(device), fc_b.to(device), W_out.to(device), b_out.to(device))


# ── Float PPL ─────────────────────────────────────────────────────────────────
print("Computing float PPL...")
float_ppl = compute_ppl(model)
print(f"Float PPL: {float_ppl:.3f}")

# ── Strategy A: all-float fitting ─────────────────────────────────────────────
print(f"\nStrategy A: all-float fitting for blocks {BLOCKS}...")
mlp_in_a, mlp_out_a = collect_mlp_io(model, calib_ids, BLOCKS)

comp_a = copy.deepcopy(model)
for b in BLOCKS:
    orig_block = model.transformer.h[b]
    cmlp = fit_block(orig_block, mlp_in_a[b], mlp_out_a[b], K, DEVICE)
    comp_a.transformer.h[b].mlp = cmlp
    print(f"  Block {b}: fitted", flush=True)

print("Computing Strategy A PPL...")
ppl_a = compute_ppl(comp_a)
print(f"Strategy A PPL: {ppl_a:.3f}")
del comp_a

# ── Strategy B: sequential progressive fitting ────────────────────────────────
print(f"\nStrategy B: sequential progressive fitting for blocks {BLOCKS}...")
working_model = copy.deepcopy(model)  # starts as float, blocks replaced sequentially
comp_b = copy.deepcopy(model)         # final compressed model

for b in BLOCKS:
    print(f"  Block {b}: collecting from {'float' if b == BLOCKS[0] else 'partially compressed'} model...", flush=True)
    # Collect MLP I/O from the CURRENT working model (previous blocks may be compressed)
    mlp_in_b, mlp_out_b = collect_mlp_io(working_model, calib_ids, [b])

    orig_block = model.transformer.h[b]  # original float block for weight info
    cmlp = fit_block(orig_block, mlp_in_b[b], mlp_out_b[b], K, DEVICE)

    # Replace in BOTH working_model (for next iteration's calibration) and comp_b (for eval)
    working_model.transformer.h[b].mlp = copy.deepcopy(cmlp)
    comp_b.transformer.h[b].mlp = cmlp
    print(f"  Block {b}: fitted", flush=True)

del working_model

print("Computing Strategy B PPL...")
ppl_b = compute_ppl(comp_b)
print(f"Strategy B PPL: {ppl_b:.3f}")
del comp_b

# ── Verdict ───────────────────────────────────────────────────────────────────
improvement_b_over_a = ppl_a - ppl_b

if improvement_b_over_a > 2:
    verdict = "CONFIRMED"
elif improvement_b_over_a < 0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nFloat PPL:         {float_ppl:.3f}")
print(f"Strategy A PPL:    {ppl_a:.3f}  (all-float fitting)")
print(f"Strategy B PPL:    {ppl_b:.3f}  (sequential progressive)")
print(f"Improvement B→A:   {improvement_b_over_a:.3f}")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 197,
    "item": 208,
    "verdict": verdict,
    "float_ppl": float_ppl,
    "strategy_a_ppl": ppl_a,
    "strategy_b_ppl": ppl_b,
    "improvement_b_over_a": improvement_b_over_a,
    "config": {
        "BLOCKS": BLOCKS,
        "K": K,
        "N_SEQS_CALIB": N_SEQS_CALIB,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
        "DEVICE": DEVICE,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
