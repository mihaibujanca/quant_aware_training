"""
Cycle 208 — Item 219: PCA-conditioned lstsq for block 1.
Block 1 inputs span ~192 dimensions (input_rank_90=192); 576 near-zero-variance
dimensions may add noise to lstsq. Hypothesis: projecting to top-P PCA components
before lstsq (P=192, 256, 384), then projecting back, reduces overfitting.
Compare single-block PPL vs standard lstsq at K=512.
Confirmed if: best PCA variant achieves single-block PPL ≤1 PPL better than standard.
Refuted if: all PCA variants within 0.3 PPL of standard lstsq.
"""
import json
import math
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────
BLOCK = 1
K = 512
PCA_DIMS = [192, 256, 384, 512]   # include 512 (near full-rank) as control
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_208_pca_lstsq.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2 and data...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB + N_EVAL:
        break

calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids)} seqs, Eval: {len(eval_ids)} seqs")

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_block1_io(model, calib_ids):
    ln_outs, proj_outs = [], []
    def fn(m, inp, out):
        ln_outs.append(inp[0].detach().float().cpu().reshape(-1, 768))
        proj_outs.append(out.detach().float().cpu().reshape(-1, 768))
    h = model.transformer.h[BLOCK].mlp.register_forward_hook(fn)
    with torch.no_grad():
        for ids in calib_ids:
            model(ids)
    h.remove()
    return torch.cat(ln_outs, 0), torch.cat(proj_outs, 0)


def fit_standard_lstsq(W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, K):
    """Standard topk+lstsq (no PCA)."""
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)  # (N, 3072)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices
    W_fc_sub = W_fc[:, topk_idx]
    b_fc_sub = b_fc[topk_idx]
    act_sub = act_all[:, topk_idx]
    ones = torch.ones(act_sub.shape[0], 1)
    design = torch.cat([act_sub, ones], dim=1)
    sol = torch.linalg.lstsq(design, proj_out_cpu, rcond=None).solution
    W_out = sol[:K].T
    b_out = sol[K]
    return W_fc_sub, b_fc_sub, W_out, b_out


def fit_pca_lstsq(W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, K, P):
    """PCA-conditioned lstsq: project inputs to top-P PCA components first."""
    # Compute PCA basis from calibration inputs
    mu = ln_out_cpu.mean(0)                          # (768,)
    X_centered = ln_out_cpu - mu                     # (N, 768)
    # SVD of X_centered for PCA
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)  # Vh: (768, 768)
    V_P = Vh[:P].T                                   # (768, P) — top P directions

    # Project inputs to PCA subspace
    ln_pca = (ln_out_cpu - mu) @ V_P                 # (N, P)

    # Transform fc weight/bias into PCA input space
    # Standard: act = gelu(x @ W_fc + b_fc)
    # PCA version: act = gelu(ln_pca @ (V_P.T @ W_fc) + b_fc)
    W_fc_pca = V_P.T @ W_fc                          # (P, 3072)
    # b_fc contribution from mean: gelu((x-mu) @ V_P @ W_fc_pca + mu @ W_fc + b_fc)
    b_fc_pca = mu @ W_fc + b_fc                      # (3072,)

    act_all = F.gelu(ln_pca @ W_fc_pca + b_fc_pca)  # (N, 3072)

    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices
    act_sub = act_all[:, topk_idx]
    W_fc_sub = W_fc_pca[:, topk_idx]                # (P, K)
    b_fc_sub = b_fc_pca[topk_idx]                   # (K,)

    ones = torch.ones(act_sub.shape[0], 1)
    design = torch.cat([act_sub, ones], dim=1)
    sol = torch.linalg.lstsq(design, proj_out_cpu, rcond=None).solution
    W_out = sol[:K].T                                # (768, K)
    b_out = sol[K]                                   # (768,)

    # Return W_fc in PCA-projected form; eval needs V_P, mu
    return W_fc_sub, b_fc_sub, W_out, b_out, V_P, mu


def eval_ppl_block(model_orig, eval_ids, W_fc_sub, b_fc_sub, W_out, b_out,
                   V_P=None, mu=None):
    """Patch block 1 and evaluate PPL. If V_P provided, forward uses PCA projection."""
    import copy

    class StandardMLP(nn.Module):
        def __init__(self, W_fc, b_fc, W_out, b_out):
            super().__init__()
            self.register_buffer("W_fc", W_fc)
            self.register_buffer("b_fc", b_fc)
            self.register_buffer("W_out", W_out)
            self.register_buffer("b_out", b_out)
        def forward(self, x):
            return F.gelu(x @ self.W_fc + self.b_fc) @ self.W_out.T + self.b_out

    class PCAmLP(nn.Module):
        def __init__(self, W_fc_pca, b_fc_pca, W_out, b_out, V_P, mu):
            super().__init__()
            self.register_buffer("W_fc_pca", W_fc_pca)
            self.register_buffer("b_fc_pca", b_fc_pca)
            self.register_buffer("W_out", W_out)
            self.register_buffer("b_out", b_out)
            self.register_buffer("V_P", V_P)    # (768, P)
            self.register_buffer("mu", mu)       # (768,)
        def forward(self, x):
            x_pca = (x - self.mu) @ self.V_P    # (B, T, P)
            act = F.gelu(x_pca @ self.W_fc_pca + self.b_fc_pca)
            return act @ self.W_out.T + self.b_out

    model_copy = copy.deepcopy(model_orig)
    if V_P is None:
        mlp = StandardMLP(W_fc_sub.to(DEVICE), b_fc_sub.to(DEVICE),
                          W_out.to(DEVICE), b_out.to(DEVICE))
    else:
        mlp = PCAmLP(W_fc_sub.to(DEVICE), b_fc_sub.to(DEVICE),
                     W_out.to(DEVICE), b_out.to(DEVICE),
                     V_P.to(DEVICE), mu.to(DEVICE))

    model_copy.transformer.h[BLOCK].mlp = mlp
    model_copy.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model_copy(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    del model_copy
    return math.exp(total_loss / total_tokens)


# ── Float baseline ────────────────────────────────────────────────────────────
print("Float baseline PPL...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
total_loss, total_tokens = 0.0, 0
with torch.no_grad():
    for ids in eval_ids:
        out = model(ids, labels=ids)
        total_loss += out.loss.item() * (ids.shape[1] - 1)
        total_tokens += ids.shape[1] - 1
float_ppl = math.exp(total_loss / total_tokens)
print(f"Float PPL: {float_ppl:.3f}")

# ── Collect I/O ───────────────────────────────────────────────────────────────
print(f"Collecting block {BLOCK} I/O...")
ln_out_cpu, proj_out_cpu = collect_block1_io(model, calib_ids)

W_fc = model.transformer.h[BLOCK].mlp.c_fc.weight.data.float().cpu()
b_fc = model.transformer.h[BLOCK].mlp.c_fc.bias.data.float().cpu()
W_proj = model.transformer.h[BLOCK].mlp.c_proj.weight.data.float().cpu()

# ── Standard lstsq baseline ───────────────────────────────────────────────────
print("Standard lstsq (no PCA)...")
W_fc_sub, b_fc_sub, W_out, b_out = fit_standard_lstsq(
    W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, K)
ppl_standard = eval_ppl_block(model, eval_ids, W_fc_sub, b_fc_sub, W_out, b_out)
print(f"  Standard lstsq PPL: {ppl_standard:.3f}")

# ── PCA-conditioned lstsq variants ────────────────────────────────────────────
ppl_results = {"standard": float(ppl_standard)}
for P in PCA_DIMS:
    print(f"PCA P={P}...")
    W_fc_sub_p, b_fc_sub_p, W_out_p, b_out_p, V_P, mu = fit_pca_lstsq(
        W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, K, P)
    ppl = eval_ppl_block(model, eval_ids, W_fc_sub_p, b_fc_sub_p, W_out_p, b_out_p,
                         V_P=V_P, mu=mu)
    ppl_results[f"pca_p{P}"] = float(ppl)
    print(f"  PCA P={P}: PPL={ppl:.3f}", flush=True)

# ── Verdict ───────────────────────────────────────────────────────────────────
best_pca_ppl = min(v for k, v in ppl_results.items() if k != "standard")
best_pca_name = min((k for k in ppl_results if k != "standard"),
                    key=lambda k: ppl_results[k])
improvement = ppl_results["standard"] - best_pca_ppl

if improvement >= 1.0:
    verdict = "CONFIRMED"
elif improvement < 0.3:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:    {float_ppl:.3f}")
print(f"  Standard:     {ppl_standard:.3f}")
for k, v in ppl_results.items():
    if k != "standard":
        print(f"  {k}: {v:.3f}")
print(f"  Best PCA:     {best_pca_name}={best_pca_ppl:.3f}")
print(f"  Improvement:  {improvement:.3f}")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 208,
    "item": 219,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_results": ppl_results,
    "best_pca_name": best_pca_name,
    "best_pca_ppl": float(best_pca_ppl),
    "improvement_vs_standard": float(improvement),
    "config": {
        "BLOCK": BLOCK,
        "K": K,
        "PCA_DIMS": PCA_DIMS,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
