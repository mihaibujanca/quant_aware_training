"""
Cycle 214 — Item 224: Analytical W_out correction — can lstsq replace Adam?
Protocol: Run compressed model on calib data to collect block input activations and
the FLOAT model's MLP outputs. Re-solve lstsq for W_out:
  target = float_mlp_output(x)  [not just proj(x) from init, but the full residual info]
This is a single-pass analytical correction (no gradient descent).
Compare to Adam W_out correction (cycle 206: PPL=48.111 at step 1000).
Confirmed if: analytical PPL ≤50.1 (within 2 PPL of Adam result).
Refuted if: analytical PPL >55 (comparable to topk+lstsq init 61.75).
"""
import json
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

BLOCKS = [1, 4, 7]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_214_analytical_correction.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

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


class CompressedMLP(nn.Module):
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight)
        self.register_buffer("fc_bias", fc_bias)
        self.register_buffer("W_out", W_out)
        self.register_buffer("b_out", b_out)
    def forward(self, x):
        return F.gelu(x @ self.fc_weight + self.fc_bias) @ self.W_out.T + self.b_out


def collect_io_both(float_model, comp_model, calib_ids, block_ids):
    """Collect (compressed_act_sub, float_mlp_out) for each block."""
    comp_acts = {b: [] for b in block_ids}    # activations from compressed model
    float_outs = {b: [] for b in block_ids}   # float model MLP outputs
    float_hooks, comp_hooks = [], []

    # Float model: collect MLP outputs
    for b in block_ids:
        def mk_float_hook(bb):
            def fn(m, inp, out):
                float_outs[bb].append(out.detach().float().cpu().reshape(-1, 768))
            return fn
        float_hooks.append(float_model.transformer.h[b].mlp.register_forward_hook(mk_float_hook(b)))

    with torch.no_grad():
        for ids in calib_ids:
            float_model(ids)
    for h in float_hooks:
        h.remove()

    # Compressed model: collect fc activations (intermediate, post-GELU)
    for b in block_ids:
        def mk_comp_hook(bb):
            def fn(m, inp, out):
                # Compute the intermediate activation (post-GELU, before W_out)
                act = F.gelu(inp[0].detach().float().cpu().reshape(-1, 768) @
                             m.fc_weight.float().cpu() + m.fc_bias.float().cpu())
                comp_acts[bb].append(act)
            return fn
        comp_hooks.append(comp_model.transformer.h[b].mlp.register_forward_hook(mk_comp_hook(b)))

    with torch.no_grad():
        for ids in calib_ids:
            comp_model(ids)
    for h in comp_hooks:
        h.remove()

    for b in block_ids:
        comp_acts[b] = torch.cat(comp_acts[b])
        float_outs[b] = torch.cat(float_outs[b])
    return comp_acts, float_outs


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# Float baseline
print("Loading float model...")
float_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
float_model.eval()
float_ppl = eval_ppl(float_model, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")

# Build topk+lstsq compressed model (standard init)
print("Building topk+lstsq init...")

def fit_block_lstsq_init(block, ln_out_cpu, proj_out_cpu, K):
    W_fc = block.mlp.c_fc.weight.data.float().cpu()
    b_fc = block.mlp.c_fc.bias.data.float().cpu()
    W_proj = block.mlp.c_proj.weight.data.float().cpu()
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices
    act_sub = act_all[:, topk_idx]
    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    return CompressedMLP(W_fc[:, topk_idx].to(DEVICE), b_fc[topk_idx].to(DEVICE),
                         sol[:K].T.to(DEVICE), sol[K].to(DEVICE)), topk_idx


# Need to collect initial io using float model (same as standard init)
def collect_float_io(float_model, calib_ids, block_ids):
    ln_outs = {b: [] for b in block_ids}
    proj_outs = {b: [] for b in block_ids}
    hooks = []
    for b in block_ids:
        def mk_hook(bb):
            def fn(m, inp, out):
                ln_outs[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
                proj_outs[bb].append(out.detach().float().cpu().reshape(-1, 768))
            return fn
        hooks.append(float_model.transformer.h[b].mlp.register_forward_hook(mk_hook(b)))
    with torch.no_grad():
        for ids in calib_ids:
            float_model(ids)
    for h in hooks:
        h.remove()
    return {b: torch.cat(v) for b, v in ln_outs.items()}, {b: torch.cat(v) for b, v in proj_outs.items()}

ln_outs, proj_outs = collect_float_io(float_model, calib_ids, BLOCKS)

# Build compressed model
comp_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
comp_model.eval()
topk_indices = {}
for b in BLOCKS:
    mlp, topk_idx = fit_block_lstsq_init(comp_model.transformer.h[b],
                                          ln_outs[b], proj_outs[b], K)
    comp_model.transformer.h[b].mlp = mlp
    topk_indices[b] = topk_idx

ppl_init = eval_ppl(comp_model, eval_ids)
print(f"Init PPL (topk+lstsq): {ppl_init:.3f}")

# ── Analytical correction ─────────────────────────────────────────────────────
# Collect (compressed_act_sub, float_mlp_out) from compressed model running forward pass
# BUT with float model's MLP outputs as targets
print("Collecting analytical correction data...")
comp_acts, float_outs_collected = collect_io_both(float_model, comp_model, calib_ids, BLOCKS)

print("Solving analytical correction (lstsq on float targets)...")
analytically_corrected_W_outs = {}
for b in BLOCKS:
    act_sub = comp_acts[b]          # (N, K) — compressed model's GELU activations
    target = float_outs_collected[b]  # (N, 768) — float model's MLP outputs
    ones = torch.ones(act_sub.shape[0], 1)
    design = torch.cat([act_sub, ones], 1)  # (N, K+1)
    sol = torch.linalg.lstsq(design, target, rcond=None).solution
    analytically_corrected_W_outs[b] = (sol[:K].T, sol[K])   # (768,K), (768,)
    print(f"  Block {b}: lstsq solved, fit residual = {(design @ sol - target).norm():.3f}")

# Patch compressed model with analytically corrected W_out
for b in BLOCKS:
    W_out_corr, b_out_corr = analytically_corrected_W_outs[b]
    comp_model.transformer.h[b].mlp.W_out = W_out_corr.to(DEVICE)
    comp_model.transformer.h[b].mlp.b_out = b_out_corr.to(DEVICE)

ppl_analytical = eval_ppl(comp_model, eval_ids)
print(f"\nPPL after analytical correction: {ppl_analytical:.3f}")

# Reference: Adam correction result (cycle 206)
adam_1000_steps = 48.111
adam_best = 46.556

improvement_over_init = ppl_init - ppl_analytical
gap_vs_adam = ppl_analytical - adam_1000_steps

if ppl_analytical <= 50.1:
    verdict = "CONFIRMED"
elif ppl_analytical > 55.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:              {float_ppl:.3f}")
print(f"  Init PPL:               {ppl_init:.3f}")
print(f"  Analytical correction:  {ppl_analytical:.3f}")
print(f"  Adam @1000 steps (ref): {adam_1000_steps:.3f}")
print(f"  Adam best (ref):        {adam_best:.3f}")
print(f"  Improvement over init:  {improvement_over_init:+.3f}")
print(f"  Gap vs Adam:            {gap_vs_adam:+.3f}")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 214,
    "item": 224,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_init": float(ppl_init),
    "ppl_analytical": float(ppl_analytical),
    "improvement_over_init": float(improvement_over_init),
    "gap_vs_adam_1000": float(gap_vs_adam),
    "adam_1000_ref": adam_1000_steps,
    "adam_best_ref": adam_best,
    "config": {"BLOCKS": BLOCKS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
               "SEQ_LEN": SEQ_LEN, "SEED": SEED},
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")

del float_model, comp_model
