"""
Cycle 300 — Item 310: semantic decomposition of cycle 280's frozen A factor.

Reproduce cycle 280 seed-42 block-1 rank-2 retrain, then for each of the 2
rank-2 directions:
  (1) Input side: top-20 FFN-intermediate channels by |A[:, i]|, with σ_c
      (cycle 261 sign frequency), ||W_fc[:, c]||, pre-activation mean/std.
  (2) Output side: wte.weight @ B_final[i, :] → top-20 boost/suppress tokens.
  (3) Block-2 Q/K/V top-5 principal cosines of B_final[i, :] with left SV
      bases (cycle 218 infrastructure).
  (4) Compare trained B_final to SVD init Vh_init[:2, :] via output-side
      overlap.

Verdict per item 310:
  CONFIRMED structured iff ≥1 direction has top-20 boost tokens with ≥10/20
                       non-punct/non-stopword alphabetic subwords.
  CONFIRMED unigram    iff both directions' top-20 boost lists are ≥14/20
                       stopwords/punct.
  REFUTED interpretable iff neither cluster is coherent AND all Q/K/V cos
                       values for both directions are below random baseline
                       mean + 1σ.
  INCONCLUSIVE otherwise.
"""
import json
import math
import random
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_BLOCK = 1
RETRAIN_RANK = 2
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_LINALG = "cpu"
N_STEPS = 4000
LR = 1e-4
EVAL_EVERY = 100
SEED = 42
TOPK = 20
N_RANDOM_SEEDS = 100

FLOAT_BASELINE_REF = 53.895
CYCLE_280_INIT_REF = 57.664
CYCLE_280_BEST_REF = 49.032

OUTPUT_PATH = "autoresearch/results/cycle_300_semantic.json"
SUMMARY_PATH = "autoresearch/results/cycle_300_summary.md"

# Stopword/punct set for verdict classification (GPT-2 BPE forms, with leading space).
STOPWORD_BASE = {
    ",", ".", ":", ";", "(", ")", "?", "!", "'", '"', "-", "--", "'s", "'t",
    "the", "a", "an", "of", "in", "to", "is", "was", "and", "it", "that",
    "for", "on", "with", "as", "by", "at", "from", "or", "but", "not", "be",
    "are", "were", "has", "have", "had", "this", "these", "those", "he",
    "she", "they", "we", "you", "his", "her", "their", "its",
}
# Also accept leading-space variants (GPT-2 tokenizer standard)
STOPWORD_SET = set(STOPWORD_BASE) | {" " + w for w in STOPWORD_BASE}


torch.manual_seed(SEED)
random.seed(SEED)
print(f"Device: {DEVICE}", flush=True)


# ── Load model + data ────────────────────────────────────────────────────────
print("Loading GPT-2 and data...", flush=True)
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

calib_ids_cpu = [t for t in all_ids[:N_CALIB]]
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids)} seqs", flush=True)


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


class LowRankCProj(nn.Module):
    def __init__(self, A_frozen, B_init, b_init, nf=768):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())
        self.bias = nn.Parameter(b_init.clone().contiguous())
        self.nf = nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W = self.A @ self.B
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        x = x.view(size_out)
        return x


# ── Float baseline + SVD ─────────────────────────────────────────────────────
t0_total = time.time()
print("Float baseline PPL...", flush=True)
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
t0 = time.time()
float_ppl = eval_ppl(model, eval_ids)
print(f"Float PPL: {float_ppl:.3f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from cycle reference {FLOAT_BASELINE_REF} by > 0.5"
)

print(f"\nSVD of block {TARGET_BLOCK} c_proj.weight...", flush=True)
c_proj_orig_weight = model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.detach().clone()
c_proj_orig_bias = model.transformer.h[TARGET_BLOCK].mlp.c_proj.bias.data.detach().clone()
assert c_proj_orig_weight.shape == (3072, 768)
W_orig = c_proj_orig_weight.float().cpu()
t0 = time.time()
U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
print(f"  SVD done in {time.time()-t0:.1f}s; S[:5]={S[:5].tolist()}", flush=True)

A2 = (U[:, :RETRAIN_RANK] * S[:RETRAIN_RANK]).contiguous()  # (3072, 2)
B2_init = Vh[:RETRAIN_RANK, :].contiguous()  # (2, 768)
Vh_init = Vh[:RETRAIN_RANK, :].clone()  # (2, 768), pre-retrain reference


# ── Cache sigma_c, W_fc col norms, pre-activation stats on block 1 ───────────
# Block input layernorm output for block 1: the input to c_fc is ln_2(x)
# where x is the attention-summed residual. We compute via a forward hook.
print("\nCollecting block-1 pre-activation z = ln_2(x) @ W_fc + b_fc over calib...", flush=True)
W_fc = model.transformer.h[TARGET_BLOCK].mlp.c_fc.weight.detach().float().cpu()  # (768, 3072)
b_fc = model.transformer.h[TARGET_BLOCK].mlp.c_fc.bias.detach().float().cpu()  # (3072,)
assert W_fc.shape == (768, 3072)
W_fc_col_norms = torch.linalg.norm(W_fc, dim=0)  # (3072,)

ln2_out_chunks = []
def _hook(_mod, _inp, out):
    # out is ln_2(x); shape (B, T, 768)
    ln2_out_chunks.append(out.detach().float().cpu())

h = model.transformer.h[TARGET_BLOCK].ln_2.register_forward_hook(_hook)
model.eval()
with torch.no_grad():
    for ids in calib_ids_cpu:
        _ = model(ids.to(DEVICE))
h.remove()
ln2_out = torch.cat([c.reshape(-1, 768) for c in ln2_out_chunks], dim=0)  # (N_TOKENS, 768)
N_TOKENS = ln2_out.shape[0]
assert N_TOKENS == N_CALIB * SEQ_LEN, f"{N_TOKENS} != {N_CALIB * SEQ_LEN}"
print(f"  ln2_out: {tuple(ln2_out.shape)}", flush=True)

preact = ln2_out @ W_fc + b_fc  # (N_TOKENS, 3072)
sigma_c = (preact > 0).float().mean(dim=0)  # (3072,)
preact_mean = preact.mean(dim=0)  # (3072,)
preact_std = preact.std(dim=0)  # (3072,)
assert sigma_c.shape == (3072,)
print(f"  σ_c stats: min={float(sigma_c.min()):.3f} max={float(sigma_c.max()):.3f} "
      f"mean={float(sigma_c.mean()):.3f}", flush=True)

# Free memory
del preact, ln2_out_chunks, ln2_out


# ── Build rank-2 replacement and retrain ─────────────────────────────────────
print(f"\n=== Retrain at r={RETRAIN_RANK} (seed {SEED}, {N_STEPS} steps) ===", flush=True)
torch.manual_seed(SEED)
random.seed(SEED)

model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_s.eval()

lowrank = LowRankCProj(
    A_frozen=A2.to(device=DEVICE, dtype=torch.float32),
    B_init=B2_init.to(device=DEVICE, dtype=torch.float32),
    b_init=c_proj_orig_bias.to(device=DEVICE, dtype=torch.float32),
    nf=768,
).to(DEVICE)
model_s.transformer.h[TARGET_BLOCK].mlp.c_proj = lowrank

init_ppl = eval_ppl(model_s, eval_ids)
print(f"  init PPL (rank-{RETRAIN_RANK} no retrain): {init_ppl:.4f} (cycle 280 ref {CYCLE_280_INIT_REF})", flush=True)
assert abs(init_ppl - CYCLE_280_INIT_REF) < 0.05, (
    f"init PPL {init_ppl} != cycle 280 ref {CYCLE_280_INIT_REF}"
)

for p in model_s.parameters():
    p.requires_grad_(False)
lowrank.B.requires_grad_(True)
lowrank.bias.requires_grad_(True)
trainable = [lowrank.B, lowrank.bias]
n_trainable = sum(p.numel() for p in trainable)
expected_trainable = RETRAIN_RANK * 768 + 768
print(f"  Trainable params: {n_trainable:,} (expect {expected_trainable:,})", flush=True)
assert n_trainable == expected_trainable

optimizer = torch.optim.Adam(trainable, lr=LR)

calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
random.shuffle(calib_ids)

best_ppl = init_ppl
best_step = 0
best_state = {
    "B": lowrank.B.detach().clone(),
    "bias": lowrank.bias.detach().clone(),
}

step = 0
t_train = time.time()
while step < N_STEPS:
    for ids in calib_ids:
        if step >= N_STEPS:
            break
        model_s.train()
        optimizer.zero_grad()
        out = model_s(ids, labels=ids)
        loss = out.loss
        loss.backward()
        optimizer.step()
        step += 1

        if step % EVAL_EVERY == 0:
            ppl_e = eval_ppl(model_s, eval_ids)
            if ppl_e < best_ppl:
                best_ppl = ppl_e
                best_step = step
                best_state = {
                    "B": lowrank.B.detach().clone(),
                    "bias": lowrank.bias.detach().clone(),
                }
            if step % 500 == 0:
                print(f"    step {step:4d}: eval_ppl={ppl_e:.4f}  best={best_ppl:.4f}@{best_step}", flush=True)

final_ppl = eval_ppl(model_s, eval_ids)
# Restore best
with torch.no_grad():
    lowrank.B.data.copy_(best_state["B"])
    lowrank.bias.data.copy_(best_state["bias"])
restored_ppl = eval_ppl(model_s, eval_ids)
assert abs(restored_ppl - best_ppl) < 1e-3
assert best_ppl < init_ppl
dt_train = time.time() - t_train
print(
    f"\nFinal: init={init_ppl:.4f}  best={best_ppl:.4f}@{best_step}  "
    f"final@{N_STEPS}={final_ppl:.4f}  restored={restored_ppl:.4f}  [{dt_train/60:.1f} min]",
    flush=True,
)

B_final = lowrank.B.detach().float().cpu().clone()  # (2, 768)
bias_final = lowrank.bias.detach().float().cpu().clone()  # (768,)
assert B_final.shape == (RETRAIN_RANK, 768)
assert bias_final.shape == (768,)


# ── Block-2 Q/K/V left SV bases (cycle 218 infra) ────────────────────────────
print("\nBuilding block-2 Q/K/V left SV bases...", flush=True)
W_attn_b2 = model.transformer.h[TARGET_BLOCK + 1].attn.c_attn.weight.detach().float().cpu()
assert W_attn_b2.shape == (768, 2304)
Q_b2 = W_attn_b2[:, 0:768]
K_b2 = W_attn_b2[:, 768:1536]
V_b2 = W_attn_b2[:, 1536:2304]

def top_left_svs(W, k=20):
    U_, _, _ = torch.linalg.svd(W, full_matrices=False)
    return U_[:, :k].contiguous()  # (768, k)

U_q = top_left_svs(Q_b2, 20)
U_k = top_left_svs(K_b2, 20)
U_v = top_left_svs(V_b2, 20)
print(f"  U_q {tuple(U_q.shape)}, U_k {tuple(U_k.shape)}, U_v {tuple(U_v.shape)}", flush=True)


# ── Random null for Q/K/V cos ────────────────────────────────────────────────
print(f"Random null (N={N_RANDOM_SEEDS})...", flush=True)
g_null = torch.Generator().manual_seed(20260406)
null_cos_sums = {"Q": [], "K": [], "V": []}
null_cos_max = {"Q": [], "K": [], "V": []}
for _ in range(N_RANDOM_SEEDS):
    r = torch.randn(768, generator=g_null)
    r = r / torch.linalg.norm(r)
    for name, Ub in (("Q", U_q), ("K", U_k), ("V", U_v)):
        cos_vec = (Ub.T @ r).abs()  # (20,)
        cos_sorted, _ = torch.sort(cos_vec, descending=True)
        null_cos_sums[name].append(float(cos_sorted[:5].sum().item()))
        null_cos_max[name].append(float(cos_sorted[0].item()))

null_stats = {}
for name in ("Q", "K", "V"):
    t5 = torch.tensor(null_cos_sums[name])
    tm = torch.tensor(null_cos_max[name])
    null_stats[name] = {
        "mean_top5_sum": float(t5.mean().item()),
        "std_top5_sum": float(t5.std().item()),
        "mean_top1": float(tm.mean().item()),
        "std_top1": float(tm.std().item()),
    }
    print(f"  {name}: top-5 sum {t5.mean():.4f}±{t5.std():.4f}  top-1 {tm.mean():.4f}±{tm.std():.4f}", flush=True)


# ── Helper: analyse one direction on (input, output, Q/K/V) ──────────────────
wte = model.transformer.wte.weight.detach().float().cpu()  # (50257, 768)
print(f"\nwte: {tuple(wte.shape)}", flush=True)

def decode_tokens(ids_list):
    return [tokenizer.decode([int(i)], clean_up_tokenization_spaces=False) for i in ids_list]


def classify_token_list(decoded):
    """Return (n_stopword_or_punct, n_alpha_subword)."""
    n_stop = 0
    n_alpha = 0
    for t in decoded:
        # strip nothing; a stopword in the set is exact-match including leading space.
        if t in STOPWORD_SET:
            n_stop += 1
            continue
        # Alpha subword heuristic: at least one alpha character, and the non-space
        # content is alphabetic (allows leading space + word).
        body = t.lstrip(" ")
        if len(body) > 0 and body.isalpha():
            n_alpha += 1
        else:
            # Non-alpha non-stopword (digits, punct, hybrid): count as stopword-ish
            n_stop += 1
    return n_stop, n_alpha


def analyse_direction(a_i, b_i):
    """a_i (3072,), b_i (768,). Returns dict."""
    # ── Input side ──
    abs_a = a_i.abs()
    top_idx = torch.argsort(abs_a, descending=True)[:TOPK]
    input_channels = []
    for rank, c in enumerate(top_idx.tolist()):
        input_channels.append({
            "rank": rank,
            "channel": int(c),
            "a_value": float(a_i[c].item()),
            "sigma_c": float(sigma_c[c].item()),
            "W_fc_col_norm": float(W_fc_col_norms[c].item()),
            "preact_mean": float(preact_mean[c].item()),
            "preact_std": float(preact_std[c].item()),
        })
    sig_vals = torch.tensor([ic["sigma_c"] for ic in input_channels])
    n_neg = int((sig_vals <= 0.1).sum().item())
    n_pos = int((sig_vals >= 0.9).sum().item())
    n_bal = int(((sig_vals > 0.4) & (sig_vals < 0.6)).sum().item())
    n_bump = int(((sig_vals > 0.1) & (sig_vals < 0.9)).sum().item())

    # ── Output side: token projection ──
    logits = wte @ b_i  # (50257,)
    # Normalise for cosine
    b_norm = float(torch.linalg.norm(b_i).item())
    wte_norms = torch.linalg.norm(wte, dim=1)  # (50257,)
    cos = logits / (b_norm * wte_norms + 1e-12)

    top_boost_idx = torch.argsort(logits, descending=True)[:TOPK]
    top_supp_idx = torch.argsort(logits, descending=False)[:TOPK]

    boost_tokens = decode_tokens(top_boost_idx.tolist())
    supp_tokens = decode_tokens(top_supp_idx.tolist())
    boost_logits = [float(logits[i].item()) for i in top_boost_idx.tolist()]
    supp_logits = [float(logits[i].item()) for i in top_supp_idx.tolist()]
    boost_cos = [float(cos[i].item()) for i in top_boost_idx.tolist()]
    supp_cos = [float(cos[i].item()) for i in top_supp_idx.tolist()]

    n_stop_b, n_alpha_b = classify_token_list(boost_tokens)
    n_stop_s, n_alpha_s = classify_token_list(supp_tokens)

    max_abs_cos_any_token = float(cos.abs().max().item())

    # ── Q/K/V alignment ──
    b_unit = b_i / (b_norm + 1e-12)
    qkv_cos = {}
    for name, Ub in (("Q", U_q), ("K", U_k), ("V", U_v)):
        c_vec = (Ub.T @ b_unit).abs()  # (20,)
        sorted_c, _ = torch.sort(c_vec, descending=True)
        qkv_cos[name] = {
            "top5_sum": float(sorted_c[:5].sum().item()),
            "top1": float(sorted_c[0].item()),
            "top5": [float(x) for x in sorted_c[:5].tolist()],
        }

    return {
        "b_norm": b_norm,
        "input_channels": input_channels,
        "sign_group_counts": {
            "mostly_neg_le_0.1": n_neg,
            "mostly_pos_ge_0.9": n_pos,
            "balanced_0.4_to_0.6": n_bal,
            "bump_between_0.1_and_0.9": n_bump,
        },
        "output": {
            "boost_tokens": boost_tokens,
            "boost_logits": boost_logits,
            "boost_cos": boost_cos,
            "supp_tokens": supp_tokens,
            "supp_logits": supp_logits,
            "supp_cos": supp_cos,
            "max_abs_cos_any_token": max_abs_cos_any_token,
            "n_stopword_or_punct_in_boost_top20": n_stop_b,
            "n_alpha_subword_in_boost_top20": n_alpha_b,
            "n_stopword_or_punct_in_supp_top20": n_stop_s,
            "n_alpha_subword_in_supp_top20": n_alpha_s,
        },
        "block2_qkv_cos": qkv_cos,
    }


# ── Analyse both directions on TRAINED B_final ───────────────────────────────
print("\n=== Directional analysis on TRAINED B_final ===", flush=True)
A_cpu = A2.float().cpu()  # (3072, 2)
directions_trained = {}
for i in range(RETRAIN_RANK):
    a_i = A_cpu[:, i]
    b_i = B_final[i, :]
    d = analyse_direction(a_i, b_i)
    directions_trained[f"dir_{i}"] = d
    print(f"\n-- Direction {i} (trained) --", flush=True)
    print(f"  ||b|| = {d['b_norm']:.4f}", flush=True)
    print(f"  sign groups (top-20 channels): neg≤0.1={d['sign_group_counts']['mostly_neg_le_0.1']}, "
          f"bump(0.1,0.9)={d['sign_group_counts']['bump_between_0.1_and_0.9']}, "
          f"pos≥0.9={d['sign_group_counts']['mostly_pos_ge_0.9']}", flush=True)
    print(f"  top-5 channels: " + ", ".join(f"c={ic['channel']}(a={ic['a_value']:+.3f},σ={ic['sigma_c']:.2f})"
                                              for ic in d['input_channels'][:5]), flush=True)
    print(f"  max |cos(b, wte)| = {d['output']['max_abs_cos_any_token']:.4f}", flush=True)
    print(f"  boost top-5: {d['output']['boost_tokens'][:5]}", flush=True)
    print(f"  supp  top-5: {d['output']['supp_tokens'][:5]}", flush=True)
    print(f"  n_stop/n_alpha in boost top-20: {d['output']['n_stopword_or_punct_in_boost_top20']}/{d['output']['n_alpha_subword_in_boost_top20']}", flush=True)
    print(f"  Q top5_sum={d['block2_qkv_cos']['Q']['top5_sum']:.4f} top1={d['block2_qkv_cos']['Q']['top1']:.4f}", flush=True)
    print(f"  K top5_sum={d['block2_qkv_cos']['K']['top5_sum']:.4f} top1={d['block2_qkv_cos']['K']['top1']:.4f}", flush=True)
    print(f"  V top5_sum={d['block2_qkv_cos']['V']['top5_sum']:.4f} top1={d['block2_qkv_cos']['V']['top1']:.4f}", flush=True)


# ── Analyse SVD-init directions (no retrain) on output side only ────────────
print("\n=== Directional analysis on SVD INIT B_init=Vh[:2] ===", flush=True)
directions_init = {}
for i in range(RETRAIN_RANK):
    a_i = A_cpu[:, i]  # same A (frozen)
    b_i = Vh_init[i, :]
    d = analyse_direction(a_i, b_i)
    directions_init[f"dir_{i}"] = d
    print(f"\n-- Direction {i} (SVD init) --", flush=True)
    print(f"  ||b|| = {d['b_norm']:.4f}  (Vh rows are unit → expect 1.0)", flush=True)
    print(f"  max |cos(b, wte)| = {d['output']['max_abs_cos_any_token']:.4f}", flush=True)
    print(f"  boost top-5: {d['output']['boost_tokens'][:5]}", flush=True)
    print(f"  supp  top-5: {d['output']['supp_tokens'][:5]}", flush=True)
    print(f"  n_stop/n_alpha in boost top-20: {d['output']['n_stopword_or_punct_in_boost_top20']}/{d['output']['n_alpha_subword_in_boost_top20']}", flush=True)


# ── Trained vs init overlap ──────────────────────────────────────────────────
overlap = {}
for i in range(RETRAIN_RANK):
    tr = set(directions_trained[f"dir_{i}"]["output"]["boost_tokens"])
    iv = set(directions_init[f"dir_{i}"]["output"]["boost_tokens"])
    overlap[f"dir_{i}"] = {
        "trained_boost": sorted(tr),
        "init_boost": sorted(iv),
        "intersection": sorted(tr & iv),
        "only_trained": sorted(tr - iv),
        "only_init": sorted(iv - tr),
        "n_intersect": len(tr & iv),
        "n_only_trained": len(tr - iv),
        "n_only_init": len(iv - tr),
    }
    print(f"\nDirection {i}: trained∩init = {overlap[f'dir_{i}']['n_intersect']}/20, "
          f"only_trained={overlap[f'dir_{i}']['n_only_trained']}, "
          f"only_init={overlap[f'dir_{i}']['n_only_init']}", flush=True)


# ── Verdict per item 310 ─────────────────────────────────────────────────────
def direction_is_structured(d):
    n_alpha = d["output"]["n_alpha_subword_in_boost_top20"]
    return n_alpha >= 10

def direction_is_unigram(d):
    n_stop = d["output"]["n_stopword_or_punct_in_boost_top20"]
    return n_stop >= 14

def qkv_below_null(d):
    for name in ("Q", "K", "V"):
        ns = null_stats[name]
        thr = ns["mean_top5_sum"] + ns["std_top5_sum"]
        if d["block2_qkv_cos"][name]["top5_sum"] >= thr:
            return False
    return True


any_structured = any(
    direction_is_structured(directions_trained[f"dir_{i}"]) for i in range(RETRAIN_RANK)
)
both_unigram = all(
    direction_is_unigram(directions_trained[f"dir_{i}"]) for i in range(RETRAIN_RANK)
)
all_below_null = all(
    qkv_below_null(directions_trained[f"dir_{i}"]) for i in range(RETRAIN_RANK)
)
neither_coherent = (not any_structured) and (not both_unigram)

if any_structured:
    verdict = "CONFIRMED_structured"
elif both_unigram:
    verdict = "CONFIRMED_unigram"
elif neither_coherent and all_below_null:
    verdict = "REFUTED_interpretable"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== Verdict ===", flush=True)
print(f"  any direction structured (≥10/20 alpha)?  {any_structured}", flush=True)
print(f"  both directions unigram (≥14/20 stop)?    {both_unigram}", flush=True)
print(f"  all Q/K/V below null mean+1σ?             {all_below_null}", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 300,
    "item": 310,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "init_ppl": float(init_ppl),
    "best_ppl": float(best_ppl),
    "best_step": int(best_step),
    "final_ppl_at_n_steps": float(final_ppl),
    "restored_best_ppl": float(restored_ppl),
    "adam_recovery": float(init_ppl - best_ppl),
    "cycle_280_init_ref": CYCLE_280_INIT_REF,
    "cycle_280_best_ref": CYCLE_280_BEST_REF,
    "svd_top_5": S[:5].tolist(),
    "S_at_0": float(S[0]),
    "S_at_1": float(S[1]),
    "sigma_c_stats": {
        "min": float(sigma_c.min().item()),
        "max": float(sigma_c.max().item()),
        "mean": float(sigma_c.mean().item()),
        "median": float(sigma_c.median().item()),
        "n_mostly_neg_le_0.1": int((sigma_c <= 0.1).sum().item()),
        "n_mostly_pos_ge_0.9": int((sigma_c >= 0.9).sum().item()),
        "n_between_0.1_0.9": int(((sigma_c > 0.1) & (sigma_c < 0.9)).sum().item()),
    },
    "null_stats_qkv": null_stats,
    "directions_trained": directions_trained,
    "directions_init_svd": directions_init,
    "trained_vs_init_overlap": overlap,
    "verdict_logic": {
        "any_structured": bool(any_structured),
        "both_unigram": bool(both_unigram),
        "neither_coherent": bool(neither_coherent),
        "all_qkv_below_null_mean_plus_1sigma": bool(all_below_null),
        "verdict": verdict,
    },
    "config": {
        "TARGET_BLOCK": TARGET_BLOCK,
        "RETRAIN_RANK": RETRAIN_RANK,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS": N_STEPS,
        "LR": LR,
        "EVAL_EVERY": EVAL_EVERY,
        "SEED": SEED,
        "TOPK": TOPK,
        "N_RANDOM_SEEDS": N_RANDOM_SEEDS,
        "trainable_params": RETRAIN_RANK * 768 + 768,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ─────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"

lines = []
lines.append(f"# Cycle 300 — Item 310: semantic decomposition of cycle 280's A factor")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Block {TARGET_BLOCK}, rank {RETRAIN_RANK}, seed {SEED}")
lines.append(f"- Init PPL: {fmt(init_ppl)} (cycle 280 ref {CYCLE_280_INIT_REF})")
lines.append(f"- Best PPL: {fmt(best_ppl)} @ step {best_step} (cycle 280 ref {CYCLE_280_BEST_REF})")
lines.append(f"- Adam recovery: {init_ppl - best_ppl:+.3f} PPL")
lines.append("")
lines.append(f"## Random null (Q/K/V, 100 unit R^768 vectors, top-5 |cos| sum)")
for name in ("Q", "K", "V"):
    ns = null_stats[name]
    lines.append(f"- {name}: mean_top5_sum={fmt(ns['mean_top5_sum'])} ± {fmt(ns['std_top5_sum'])}, "
                 f"mean_top1={fmt(ns['mean_top1'])} ± {fmt(ns['std_top1'])}")
lines.append("")

for i in range(RETRAIN_RANK):
    d = directions_trained[f"dir_{i}"]
    lines.append(f"## Direction {i} (trained B_final)")
    lines.append("")
    lines.append(f"- ||B_final[{i}, :]|| = {fmt(d['b_norm'])}")
    lines.append(f"- Sign-group split on top-20 input channels: "
                 f"mostly_neg(≤0.1)={d['sign_group_counts']['mostly_neg_le_0.1']}, "
                 f"bump(0.1,0.9)={d['sign_group_counts']['bump_between_0.1_and_0.9']}, "
                 f"mostly_pos(≥0.9)={d['sign_group_counts']['mostly_pos_ge_0.9']}")
    lines.append("")
    lines.append("### Input side — top-20 FFN-intermediate channels by |A[c, i]|")
    lines.append("")
    lines.append("| rank | channel | A[c,i] | σ_c | ‖W_fc[:,c]‖ | preact_mean | preact_std |")
    lines.append("|-----:|--------:|-------:|----:|------------:|------------:|-----------:|")
    for ic in d["input_channels"]:
        lines.append(
            f"| {ic['rank']} | {ic['channel']} | {ic['a_value']:+.4f} | "
            f"{ic['sigma_c']:.3f} | {ic['W_fc_col_norm']:.3f} | "
            f"{ic['preact_mean']:+.3f} | {ic['preact_std']:.3f} |"
        )
    lines.append("")
    lines.append(f"### Output side — top-20 boost tokens (B_final[{i}, :] @ wte.T)")
    lines.append("")
    lines.append("| rank | token | logit | cos |")
    lines.append("|-----:|:------|------:|----:|")
    for r in range(TOPK):
        tok = d["output"]["boost_tokens"][r]
        lg = d["output"]["boost_logits"][r]
        cs = d["output"]["boost_cos"][r]
        tok_esc = tok.replace("|", "\\|").replace("\n", "\\n")
        lines.append(f"| {r} | `{tok_esc}` | {lg:+.3f} | {cs:+.4f} |")
    lines.append("")
    lines.append(f"### Output side — top-20 suppress tokens")
    lines.append("")
    lines.append("| rank | token | logit | cos |")
    lines.append("|-----:|:------|------:|----:|")
    for r in range(TOPK):
        tok = d["output"]["supp_tokens"][r]
        lg = d["output"]["supp_logits"][r]
        cs = d["output"]["supp_cos"][r]
        tok_esc = tok.replace("|", "\\|").replace("\n", "\\n")
        lines.append(f"| {r} | `{tok_esc}` | {lg:+.3f} | {cs:+.4f} |")
    lines.append("")
    lines.append(f"- max |cos(b, wte)| = {fmt(d['output']['max_abs_cos_any_token'])}")
    lines.append(f"- n_stopword_or_punct / n_alpha_subword in boost top-20 = "
                 f"{d['output']['n_stopword_or_punct_in_boost_top20']} / {d['output']['n_alpha_subword_in_boost_top20']}")
    lines.append("")
    lines.append("### Block-2 Q/K/V top-5 principal cosines (|cos|)")
    lines.append("")
    for name in ("Q", "K", "V"):
        q = d["block2_qkv_cos"][name]
        ns = null_stats[name]
        thr = ns["mean_top5_sum"] + ns["std_top5_sum"]
        marker = "above null+1σ" if q["top5_sum"] > thr else "below null+1σ"
        lines.append(f"- {name}: top5_sum={fmt(q['top5_sum'])} top1={fmt(q['top1'])} "
                     f"(null {fmt(ns['mean_top5_sum'])} ± {fmt(ns['std_top5_sum'])}) — {marker}")
    lines.append("")

for i in range(RETRAIN_RANK):
    d = directions_init[f"dir_{i}"]
    lines.append(f"## Direction {i} — SVD init Vh_init[{i}, :] boost top-10 (no retrain)")
    lines.append("")
    lines.append("| rank | token | logit |")
    lines.append("|-----:|:------|------:|")
    for r in range(10):
        tok = d["output"]["boost_tokens"][r]
        lg = d["output"]["boost_logits"][r]
        tok_esc = tok.replace("|", "\\|").replace("\n", "\\n")
        lines.append(f"| {r} | `{tok_esc}` | {lg:+.3f} |")
    lines.append("")

lines.append("## Trained vs SVD-init boost-token overlap")
lines.append("")
for i in range(RETRAIN_RANK):
    o = overlap[f"dir_{i}"]
    lines.append(f"- Direction {i}: intersection = {o['n_intersect']}/20; only_trained = {o['n_only_trained']}; only_init = {o['n_only_init']}")
lines.append("")

lines.append("## Verdict logic (item 310 verbatim)")
lines.append("")
lines.append(f"- CONFIRMED structured iff ≥1 direction has ≥ 10/20 alpha-subword boost tokens")
lines.append(f"- CONFIRMED unigram    iff both directions have ≥ 14/20 stopword/punct boost tokens")
lines.append(f"- REFUTED interpretable iff neither cluster is coherent AND all Q/K/V below null mean+1σ")
lines.append(f"- any_structured = {any_structured}")
lines.append(f"- both_unigram = {both_unigram}")
lines.append(f"- neither_coherent = {neither_coherent}")
lines.append(f"- all_qkv_below_null_mean_plus_1sigma = {all_below_null}")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)

print(f"\nTotal wall: {time.time() - t0_total:.1f}s", flush=True)
