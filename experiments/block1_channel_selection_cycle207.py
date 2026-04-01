"""
Cycle 207 — Item 218: Block 1 channel selection — magnitude topk vs output-contribution topk.
Standard topk score = act_std × W_proj_row_norm. Test alternatives:
  S1 (current): act_std × W_proj.norm(dim=1)
  S2 (abs mean): act_mean_abs × W_proj.norm(dim=1)
  S3 (pure act):  act_std alone
  S4 (pure weight): W_proj.norm(dim=1) alone
Compute Kendall-tau between all pairs. Measure single-block block 1 PPL for each at K=512.
Confirmed if: best_alternative Kendall-tau < 0.7 AND PPL improvement ≥1.0 vs S1.
Refuted if: all Kendall-tau ≥0.85 OR all PPL differences <0.3.
"""
import json
import math
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BLOCK = 1
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_207_block1_channel_selection.json"

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


def build_compressed_mlp(W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, topk_idx):
    """Fit W_out via lstsq for given channel subset."""
    W_fc_sub = W_fc[:, topk_idx]        # (768, K)
    b_fc_sub = b_fc[topk_idx]           # (K,)
    act_sub = F.gelu(ln_out_cpu @ W_fc_sub + b_fc_sub)  # (N, K)
    ones = torch.ones(act_sub.shape[0], 1)
    design = torch.cat([act_sub, ones], dim=1)
    sol = torch.linalg.lstsq(design, proj_out_cpu, rcond=None).solution
    W_out = sol[:K].T    # (768, K)
    b_out = sol[K]       # (768,)
    return W_fc_sub, b_fc_sub, W_out, b_out


def eval_ppl_single_block(model_orig, eval_ids, W_fc_sub, b_fc_sub, W_out, b_out, block):
    """Patch block and evaluate PPL."""
    import torch.nn as nn
    import copy

    class SimpleMLP(nn.Module):
        def __init__(self, W_fc, b_fc, W_out, b_out):
            super().__init__()
            self.register_buffer("W_fc", W_fc)
            self.register_buffer("b_fc", b_fc)
            self.register_buffer("W_out", W_out)
            self.register_buffer("b_out", b_out)
        def forward(self, x):
            return F.gelu(x @ self.W_fc + self.b_fc) @ self.W_out.T + self.b_out

    model_copy = copy.deepcopy(model_orig)
    mlp = SimpleMLP(W_fc_sub.to(DEVICE), b_fc_sub.to(DEVICE),
                    W_out.to(DEVICE), b_out.to(DEVICE))
    model_copy.transformer.h[block].mlp = mlp

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
print(f"  ln_out shape: {ln_out_cpu.shape}, proj_out shape: {proj_out_cpu.shape}")

# Extract weight tensors (Conv1D: weight.T is the standard linear weight)
W_fc = model.transformer.h[BLOCK].mlp.c_fc.weight.data.float().cpu()   # (768, 3072)
b_fc = model.transformer.h[BLOCK].mlp.c_fc.bias.data.float().cpu()     # (3072,)
W_proj = model.transformer.h[BLOCK].mlp.c_proj.weight.data.float().cpu()  # (3072, 768)

# Compute activations once
act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)  # (N, 3072)

# ── Four scoring criteria ─────────────────────────────────────────────────────
scores = {
    "S1_std_x_wproj":    act_all.std(0) * W_proj.norm(dim=1),
    "S2_mean_abs_x_wproj": act_all.abs().mean(0) * W_proj.norm(dim=1),
    "S3_pure_act_std":   act_all.std(0),
    "S4_pure_weight":    W_proj.norm(dim=1),
}

# Compute Kendall-tau between all pairs
score_names = list(scores.keys())
tau_matrix = {}
for i, n1 in enumerate(score_names):
    for j, n2 in enumerate(score_names):
        if j > i:
            tau, _ = kendalltau(scores[n1].numpy(), scores[n2].numpy())
            tau_matrix[f"{n1}_vs_{n2}"] = float(tau)
            print(f"  Kendall-tau {n1} vs {n2}: {tau:.4f}")

# ── PPL for each scoring criterion ────────────────────────────────────────────
ppl_results = {}
for name, score in scores.items():
    topk_idx = score.topk(K).indices
    W_fc_sub, b_fc_sub, W_out, b_out = build_compressed_mlp(
        W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, topk_idx)
    ppl = eval_ppl_single_block(model, eval_ids, W_fc_sub, b_fc_sub, W_out, b_out, BLOCK)
    ppl_results[name] = float(ppl)
    print(f"  {name}: single-block PPL={ppl:.3f}", flush=True)

# ── Verdict ───────────────────────────────────────────────────────────────────
baseline_ppl = ppl_results["S1_std_x_wproj"]
best_alt = min((ppl_results[k] for k in ppl_results if k != "S1_std_x_wproj"))
best_alt_name = min((k for k in ppl_results if k != "S1_std_x_wproj"),
                    key=lambda k: ppl_results[k])
improvement = baseline_ppl - best_alt

# Find min Kendall-tau (most different ordering)
min_tau = min(tau_matrix.values())

if improvement >= 1.0 and min_tau < 0.7:
    verdict = "CONFIRMED"
elif improvement < 0.3 or min(tau_matrix.values()) >= 0.85:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:       {float_ppl:.3f}")
print(f"  S1 (baseline):   {baseline_ppl:.3f}")
for k, v in ppl_results.items():
    if k != "S1_std_x_wproj":
        print(f"  {k}: {v:.3f}")
print(f"  Best alt:        {best_alt_name}={best_alt:.3f} (improvement={improvement:.3f})")
print(f"  Min Kendall-tau: {min_tau:.4f}")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 207,
    "item": 218,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_by_criterion": ppl_results,
    "tau_matrix": tau_matrix,
    "best_alt_criterion": best_alt_name,
    "best_alt_ppl": float(best_alt),
    "baseline_ppl_s1": float(baseline_ppl),
    "improvement_vs_s1": float(improvement),
    "min_kendall_tau": float(min_tau),
    "config": {
        "BLOCK": BLOCK,
        "K": K,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
