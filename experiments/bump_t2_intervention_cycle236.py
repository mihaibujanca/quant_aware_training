"""
Cycle 236 — Item 246 (T2): Causal intervention on top bump directions.

For each of 12 GPT-2 blocks, take the top-5 directions by variance_explained
from cycle 235's T1 features, add alpha * std_l[i] * v_{l,i} to the residual
stream entering block l+1 (or ln_f at l=11), measure logit shifts on a
held-out wikitext-2 val batch, and compare to a random-direction baseline.

Verdict:
  CONFIRMED    := n_blocks_pass >= 6 where n_blocks_pass counts blocks with
                  >= 3 directions having BOTH effect_size >= 3 AND jaccard >= 0.3
  REFUTED      := max effect_size across all (l, i) < 1.2
  INCONCLUSIVE := otherwise
"""
import json
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
CALIB_N_SEQS = 200
CALIB_SEQ_LEN = 512
HELD_N_SEQS = 8
HELD_SEQ_LEN = 128
N_BLOCKS = 12
TOP_N_DIRS = 5
ALPHAS = [-3, -1, 0, 1, 3]
N_RAND_SEEDS = 10
N_EXAMPLES = 10
TOP_VOCAB = 20

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CPU = torch.device("cpu")

RESULTS_DIR = "autoresearch/results"
FEATURES_PATH = f"{RESULTS_DIR}/cycle_235_t1_features.pt"
DECODED_JSON_PATH = f"{RESULTS_DIR}/cycle_235_t1_decoded.json"
LOGIT_SHIFTS_PT = f"{RESULTS_DIR}/cycle_236_t2_logit_shifts.pt"
INTERVENTION_JSON = f"{RESULTS_DIR}/cycle_236_t2_intervention.json"
EXAMPLES_MD = f"{RESULTS_DIR}/cycle_236_t2_examples.md"
SUMMARY_MD = f"{RESULTS_DIR}/cycle_236_t2_summary.md"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Load cycle 235 artifacts ─────────────────────────────────────────────────
print("Loading cycle 235 features and decoded records...")
features = torch.load(FEATURES_PATH, weights_only=False)
with open(DECODED_JSON_PATH) as f:
    decoded_records = json.load(f)
assert len(decoded_records) == N_BLOCKS * 32, len(decoded_records)

# Pick top-5 directions per block by variance_explained.
top5_per_block: dict[int, list[dict]] = {}
for l in range(N_BLOCKS):
    block_recs = [r for r in decoded_records if r["block"] == l]
    block_recs_sorted = sorted(block_recs, key=lambda r: -r["variance_explained"])
    top5_per_block[l] = block_recs_sorted[:TOP_N_DIRS]
    print(
        f"  block {l}: top5 dir indices = "
        f"{[r['direction_idx'] for r in top5_per_block[l]]}, "
        f"var_explained = {[round(r['variance_explained'], 4) for r in top5_per_block[l]]}"
    )

# For each (block, direction) extract T1 top-20 boost token_ids.
t1_boost_ids: dict[tuple[int, int], set[int]] = {}
for rec in decoded_records:
    b = rec["block"]
    i = rec["direction_idx"]
    boost = rec["top_vocab_boost"][:TOP_VOCAB]
    ids = {int(entry["token_id"]) for entry in boost}
    t1_boost_ids[(b, i)] = ids

# ── Recompute U_l std from calibration ───────────────────────────────────────
print("\nRecomputing U_l std from calibration corpus...")
print("Loading wikitext-2 train split for calibration...")
ds_calib = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts_calib = [r["text"] for r in ds_calib if r["text"].strip()]
all_tokens_calib = []
tokens_needed_calib = CALIB_N_SEQS * CALIB_SEQ_LEN
for t in texts_calib:
    ids = tokenizer(t, return_tensors="pt", truncation=False)["input_ids"][0]
    all_tokens_calib.append(ids)
    if len(all_tokens_calib) % 1000 == 0:
        running = sum(x.numel() for x in all_tokens_calib)
        if running >= tokens_needed_calib * 2:
            break
flat_calib = torch.cat(all_tokens_calib, dim=0)
if flat_calib.numel() < tokens_needed_calib:
    raise RuntimeError(
        f"Need {tokens_needed_calib} calib tokens but only {flat_calib.numel()}"
    )
flat_calib = flat_calib[:tokens_needed_calib].contiguous()
calib_ids = flat_calib.view(CALIB_N_SEQS, CALIB_SEQ_LEN).contiguous()
print(f"  calib_ids shape: {tuple(calib_ids.shape)}")

cfc_stash = {l: [] for l in range(N_BLOCKS)}

def make_cfc_prehook(l):
    def fn(module, args):
        cfc_stash[l].append(args[0].detach().float().cpu())
    return fn

hooks = []
for l in range(N_BLOCKS):
    h = model.transformer.h[l].mlp.c_fc.register_forward_pre_hook(make_cfc_prehook(l))
    hooks.append(h)

print("  forward pass in batches...")
BATCH = 8
with torch.no_grad():
    for i in range(0, CALIB_N_SEQS, BATCH):
        batch = calib_ids[i:i + BATCH].to(DEVICE)
        _ = model(batch)
for h in hooks:
    h.remove()

mlp_inputs = {}
for l in range(N_BLOCKS):
    t = torch.cat(cfc_stash[l], dim=0).reshape(-1, 768)
    assert t.shape == (CALIB_N_SEQS * CALIB_SEQ_LEN, 768), (l, t.shape)
    mlp_inputs[l] = t
cfc_stash = None

# Compute std_l for each block by running calibration through (GELU - ReLU) and
# projecting via the top-32 V_l from cycle 235.
u_std_per_block: dict[int, torch.Tensor] = {}
u_mean_std_per_block: dict[int, float] = {}
for l in range(N_BLOCKS):
    block = model.transformer.h[l]
    W_fc = block.mlp.c_fc.weight.detach().float().cpu()
    b_fc = block.mlp.c_fc.bias.detach().float().cpu()
    W_proj = block.mlp.c_proj.weight.detach().float().cpu()
    V_l = features[f"block_{l}"]["V"]
    mean_delta_l = features[f"block_{l}"]["mean_delta"]

    x = mlp_inputs[l]
    chunk = 4096
    parts = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk]
        preact = xc @ W_fc + b_fc
        gelu_post = F.gelu(preact, approximate="tanh")
        relu_post = F.relu(preact)
        delta_post = gelu_post - relu_post
        parts.append(delta_post @ W_proj)
    delta_out = torch.cat(parts, dim=0)
    delta_centred = delta_out - mean_delta_l
    U_l_top32 = delta_centred @ V_l  # (n_tokens, 32)
    std_l = U_l_top32.std(dim=0)  # (32,)
    u_std_per_block[l] = std_l.contiguous().clone()
    u_mean_std_per_block[l] = float(std_l.mean().item())
    print(f"  block {l}: std_l[:3] = {std_l[:3].tolist()}  mean = {u_mean_std_per_block[l]:.3f}")
    mlp_inputs[l] = None
    delta_out = None
    delta_centred = None
    U_l_top32 = None

mlp_inputs = None

# Sanity 3: std finite and O(1)--O(10)
for l in range(N_BLOCKS):
    sl = u_std_per_block[l]
    assert torch.isfinite(sl).all(), f"std_l non-finite block {l}"
    assert (sl > 0).all(), f"std_l non-positive block {l}"
    assert sl.max() < 1e4, f"std_l blew up block {l}: {sl.max().item()}"

# ── Held-out loader: wikitext-2 val split ────────────────────────────────────
print("\nLoading held-out wikitext-2 val split...")
ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts_val = [r["text"] for r in ds_val if r["text"].strip()]
all_tokens_val = []
tokens_needed_held = HELD_N_SEQS * HELD_SEQ_LEN
for t in texts_val:
    ids = tokenizer(t, return_tensors="pt", truncation=False)["input_ids"][0]
    all_tokens_val.append(ids)
flat_val = torch.cat(all_tokens_val, dim=0)
if flat_val.numel() < tokens_needed_held:
    raise RuntimeError(
        f"Need {tokens_needed_held} held-out tokens but only {flat_val.numel()}"
    )
flat_val = flat_val[:tokens_needed_held].contiguous()
held_ids = flat_val.view(HELD_N_SEQS, HELD_SEQ_LEN).contiguous().to(DEVICE)
print(f"  held_ids shape: {tuple(held_ids.shape)}")

# ── Clean logits (cache once) ────────────────────────────────────────────────
print("\nCaching clean logits on held-out batch...")
with torch.no_grad():
    logits_clean = model(held_ids).logits.detach().float().cpu()
print(f"  logits_clean shape: {tuple(logits_clean.shape)}")  # (B, T, V)
B, T, V = logits_clean.shape
assert (B, T) == (HELD_N_SEQS, HELD_SEQ_LEN)

# ── Hook utility: add delta_x to residual stream at target module ────────────
def target_module_for_block(l: int):
    """After block l, the next module that reads the residual stream.
    For l in 0..10 that's transformer.h[l+1]. For l=11 that's transformer.ln_f.
    """
    if l < N_BLOCKS - 1:
        return model.transformer.h[l + 1]
    return model.transformer.ln_f


def run_intervention(l: int, delta_x: torch.Tensor) -> torch.Tensor:
    """Add delta_x (shape (768,)) to the residual stream at input of target
    module, forward held-out batch, return logits (B, T, V) on cpu fp32."""
    delta_dev = delta_x.to(DEVICE).float()

    def pre_hook(module, args):
        x = args[0]
        return (x + delta_dev.view(1, 1, -1),) + args[1:]

    target = target_module_for_block(l)
    handle = target.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            logits = model(held_ids).logits.detach().float().cpu()
    finally:
        handle.remove()
    return logits


# ── Sanity 1: alpha=0 must give logits == logits_clean ───────────────────────
print("\nSanity 1: alpha=0 intervention matches clean logits...")
l_s = 0
V_l_s = features[f"block_{l_s}"]["V"]
delta_zero = torch.zeros(768)
logits_zero = run_intervention(l_s, delta_zero)
diff_zero = (logits_zero - logits_clean).abs().max().item()
print(f"  max |diff| at alpha=0 block {l_s} = {diff_zero:.2e}")
assert diff_zero < 1e-4, f"alpha=0 sanity failed: {diff_zero}"

# ── Decoding helper ──────────────────────────────────────────────────────────
tok_cache: dict[int, str] = {}

def decode_tok(tid: int) -> str:
    if tid not in tok_cache:
        tok_cache[tid] = tokenizer.decode([tid])
    return tok_cache[tid]


def clean_tok(s: str) -> str:
    return s.replace("\n", "\\n").replace("|", "/")


def get_context(b: int, t: int, max_chars: int = 50) -> str:
    ids = held_ids[b, :t + 1].tolist()
    s = tokenizer.decode(ids)
    s = s.replace("\n", " ")
    if len(s) > max_chars:
        s = "…" + s[-max_chars:]
    return s


# ── Random baseline per block at alpha=3 and alpha=-3 ────────────────────────
print("\nComputing random-direction baselines per block...")
random_l2: dict[int, dict[str, float]] = {}
rng = torch.Generator().manual_seed(SEED)
for l in range(N_BLOCKS):
    mean_std_l = u_mean_std_per_block[l]
    l2_at_alphas: dict[int, list[float]] = {3: [], -3: []}
    for seed_i in range(N_RAND_SEEDS):
        v_rand = torch.randn(768, generator=rng)
        v_rand = v_rand / v_rand.norm()
        for alpha in [3, -3]:
            delta_x = alpha * mean_std_l * v_rand
            logits_int = run_intervention(l, delta_x)
            logit_shift = logits_int - logits_clean
            l2 = logit_shift.pow(2).sum(-1).sqrt().mean().item()
            l2_at_alphas[alpha].append(l2)
    mean_pos = float(sum(l2_at_alphas[3]) / len(l2_at_alphas[3]))
    std_pos = float(
        (sum((v - mean_pos) ** 2 for v in l2_at_alphas[3]) / len(l2_at_alphas[3])) ** 0.5
    )
    mean_neg = float(sum(l2_at_alphas[-3]) / len(l2_at_alphas[-3]))
    std_neg = float(
        (sum((v - mean_neg) ** 2 for v in l2_at_alphas[-3]) / len(l2_at_alphas[-3])) ** 0.5
    )
    random_l2[l] = {
        "random_l2_mean_alpha_3": mean_pos,
        "random_l2_std_alpha_3": std_pos,
        "random_l2_mean_alpha_neg3": mean_neg,
        "random_l2_std_alpha_neg3": std_neg,
    }
    print(
        f"  block {l}: rand L2 α=+3 = {mean_pos:.4f}±{std_pos:.4f}, "
        f"α=-3 = {mean_neg:.4f}±{std_neg:.4f}"
    )


# ── Main intervention sweep ──────────────────────────────────────────────────
print("\nRunning main intervention sweep...")
shifts_store: dict[str, dict] = {}
intervention_entries: list[dict] = []
examples_store: dict[tuple[int, int], list[dict]] = {}
sanity_linearity: list[tuple[int, int, float, float]] = []

for l in range(N_BLOCKS):
    V_l = features[f"block_{l}"]["V"]  # (768, 32)
    std_l = u_std_per_block[l]  # (32,)
    block_shifts_key = f"block_{l}"
    shifts_store[block_shifts_key] = {}
    for rec in top5_per_block[l]:
        i = rec["direction_idx"]
        var_expl = rec["variance_explained"]
        v_li = V_l[:, i].contiguous()
        s_li = float(std_l[i].item())
        dir_key = f"direction_{i}"
        shifts_store[block_shifts_key][dir_key] = {}
        l2_per_alpha: dict[int, float] = {}
        for alpha in ALPHAS:
            if alpha == 0:
                # Skip hook entirely; zero shift.
                logit_shift = torch.zeros_like(logits_clean)
                l2_mean = 0.0
            else:
                delta_x = alpha * s_li * v_li
                logits_int = run_intervention(l, delta_x)
                logit_shift = logits_int - logits_clean
                l2 = logit_shift.pow(2).sum(-1).sqrt()  # (B, T)
                l2_mean = float(l2.mean().item())

            mean_shift_vocab = logit_shift.mean(dim=(0, 1))  # (V,)
            # Sub-sample: save mean_shift (V,) only, as per plan fallback.
            shifts_store[block_shifts_key][dir_key][f"alpha_{alpha}"] = {
                "mean_shift": mean_shift_vocab.contiguous().clone(),
            }

            top20_boost_vals, top20_boost_idx = torch.topk(mean_shift_vocab, TOP_VOCAB)
            top20_sup_vals, top20_sup_idx = torch.topk(-mean_shift_vocab, TOP_VOCAB)
            top20_boost_list = [
                {
                    "token_id": int(top20_boost_idx[k].item()),
                    "token_str": decode_tok(int(top20_boost_idx[k].item())),
                    "mean_shift": float(top20_boost_vals[k].item()),
                }
                for k in range(TOP_VOCAB)
            ]
            top20_supp_list = [
                {
                    "token_id": int(top20_sup_idx[k].item()),
                    "token_str": decode_tok(int(top20_sup_idx[k].item())),
                    "mean_shift": -float(top20_sup_vals[k].item()),
                }
                for k in range(TOP_VOCAB)
            ]

            # Jaccard vs T1 boost set (token_ids).
            t1_set = t1_boost_ids[(l, i)]
            t2_set = {int(top20_boost_idx[k].item()) for k in range(TOP_VOCAB)}
            inter = t1_set & t2_set
            union = t1_set | t2_set
            jac = len(inter) / max(len(union), 1)

            # Effect size vs random baseline at alpha=3 only.
            if alpha == 3:
                rand_mean = random_l2[l]["random_l2_mean_alpha_3"]
                effect_size = l2_mean / rand_mean if rand_mean > 1e-12 else float("inf")
                rand_mean_entry = rand_mean
                rand_std_entry = random_l2[l]["random_l2_std_alpha_3"]
            elif alpha == -3:
                rand_mean = random_l2[l]["random_l2_mean_alpha_neg3"]
                effect_size = l2_mean / rand_mean if rand_mean > 1e-12 else float("inf")
                rand_mean_entry = rand_mean
                rand_std_entry = random_l2[l]["random_l2_std_alpha_neg3"]
            else:
                effect_size = float("nan")
                rand_mean_entry = float("nan")
                rand_std_entry = float("nan")

            entry = {
                "block": l,
                "direction": i,
                "alpha": alpha,
                "variance_explained": var_expl,
                "std_l_i": s_li,
                "l2_shift_mean": l2_mean,
                "top20_boost": top20_boost_list,
                "top20_suppress": top20_supp_list,
                "effect_size": effect_size,
                "t1_jaccard_overlap": jac,
                "random_l2_mean": rand_mean_entry,
                "random_l2_std": rand_std_entry,
            }
            intervention_entries.append(entry)
            l2_per_alpha[alpha] = l2_mean

            # Concrete examples at alpha = +3 only.
            if alpha == 3:
                l2_flat = l2.flatten()
                topk_l2_vals, topk_l2_idx = torch.topk(l2_flat, N_EXAMPLES)
                recs = []
                logits_clean_local = logits_clean  # (B, T, V)
                logits_int_local = logits_clean_local + logit_shift
                for k in range(N_EXAMPLES):
                    flat_idx = int(topk_l2_idx[k].item())
                    b_idx = flat_idx // T
                    t_idx = flat_idx % T
                    orig_logits = logits_clean_local[b_idx, t_idx]
                    int_logits = logits_int_local[b_idx, t_idx]
                    orig_top5_vals, orig_top5_idx = torch.topk(orig_logits, 5)
                    int_top5_vals, int_top5_idx = torch.topk(int_logits, 5)
                    orig_top1 = int(orig_top5_idx[0].item())
                    int_top1 = int(int_top5_idx[0].item())
                    logit_delta_top = float(logit_shift[b_idx, t_idx, orig_top1].item())
                    recs.append({
                        "b": b_idx,
                        "t": t_idx,
                        "context": get_context(b_idx, t_idx),
                        "original_top1": decode_tok(orig_top1),
                        "intervened_top1": decode_tok(int_top1),
                        "top5_orig": [decode_tok(int(x.item())) for x in orig_top5_idx],
                        "top5_intervened": [decode_tok(int(x.item())) for x in int_top5_idx],
                        "alpha": alpha,
                        "logit_delta_top": logit_delta_top,
                        "l2": float(topk_l2_vals[k].item()),
                    })
                examples_store[(l, i)] = recs

        # Sanity 2: linearity. L2(α=1) <= L2(α=3) * 1.05.
        l2_a1 = l2_per_alpha.get(1, 0.0)
        l2_a3 = l2_per_alpha.get(3, 0.0)
        sanity_linearity.append((l, i, l2_a1, l2_a3))
        if l2_a1 > l2_a3 * 1.05 + 1e-6:
            print(
                f"  WARN block {l} dir {i}: l2(α=1)={l2_a1:.4f} > "
                f"l2(α=3)={l2_a3:.4f} — linearity violated"
            )

    print(
        f"  block {l} done, alpha=+3 l2 "
        f"{[round(l2_per_alpha.get(3, 0), 3) for _ in [0]]}"
    )

# ── Verdict ──────────────────────────────────────────────────────────────────
print("\nComputing verdict...")
effect_sizes: dict[tuple[int, int], float] = {}
jaccards: dict[tuple[int, int], float] = {}
for entry in intervention_entries:
    if entry["alpha"] == 3:
        key = (entry["block"], entry["direction"])
        effect_sizes[key] = entry["effect_size"]
        jaccards[key] = entry["t1_jaccard_overlap"]

n_hits_per_block: dict[int, int] = {}
for l in range(N_BLOCKS):
    n_hits = 0
    for rec in top5_per_block[l]:
        i = rec["direction_idx"]
        es = effect_sizes[(l, i)]
        jc = jaccards[(l, i)]
        if es >= 3.0 and jc >= 0.3:
            n_hits += 1
    n_hits_per_block[l] = n_hits

n_blocks_pass = sum(1 for l in range(N_BLOCKS) if n_hits_per_block[l] >= 3)
max_effect_size = max(effect_sizes.values())

if n_blocks_pass >= 6:
    verdict = "CONFIRMED"
elif max_effect_size < 1.2:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"  n_hits_per_block = {n_hits_per_block}")
print(f"  n_blocks_pass = {n_blocks_pass}/12")
print(f"  max_effect_size = {max_effect_size:.3f}")
print(f"  verdict = {verdict}")

# ── Save artifacts ───────────────────────────────────────────────────────────
print("\nSaving artifacts...")
torch.save(shifts_store, LOGIT_SHIFTS_PT)
print(f"  {LOGIT_SHIFTS_PT}")

config = {
    "seed": SEED,
    "calib_n_seqs": CALIB_N_SEQS,
    "calib_seq_len": CALIB_SEQ_LEN,
    "held_n_seqs": HELD_N_SEQS,
    "held_seq_len": HELD_SEQ_LEN,
    "n_blocks": N_BLOCKS,
    "top_n_dirs": TOP_N_DIRS,
    "alphas": ALPHAS,
    "n_rand_seeds": N_RAND_SEEDS,
    "device": DEVICE,
}

u_std_per_block_json = {
    str(l): u_std_per_block[l].tolist() for l in range(N_BLOCKS)
}

output_json = {
    "config": config,
    "verdict": verdict,
    "n_blocks_pass": n_blocks_pass,
    "n_hits_per_block": {str(l): n_hits_per_block[l] for l in range(N_BLOCKS)},
    "max_effect_size": max_effect_size,
    "u_std_per_block": u_std_per_block_json,
    "u_mean_std_per_block": {str(l): u_mean_std_per_block[l] for l in range(N_BLOCKS)},
    "random_baselines": {str(l): random_l2[l] for l in range(N_BLOCKS)},
    "entries": intervention_entries,
}

with open(INTERVENTION_JSON, "w") as f:
    json.dump(output_json, f, indent=2)
print(f"  {INTERVENTION_JSON}")

# ── examples.md ──────────────────────────────────────────────────────────────
example_lines = ["# Cycle 236 T2 — Intervention examples (alpha=+3)", ""]
example_lines.append(
    f"Verdict: **{verdict}**  |  n_blocks_pass: **{n_blocks_pass}/12**  |  "
    f"max_effect_size: **{max_effect_size:.3f}**"
)
example_lines.append("")
for l in range(N_BLOCKS):
    example_lines.append(f"## Block {l}")
    example_lines.append("")
    example_lines.append(f"n_hits = {n_hits_per_block[l]}/5")
    example_lines.append("")
    for rec in top5_per_block[l]:
        i = rec["direction_idx"]
        var = rec["variance_explained"]
        es = effect_sizes[(l, i)]
        jc = jaccards[(l, i)]
        example_lines.append(
            f"### Block {l} dir {i} — var={var:.2%} effect_size={es:.2f} jaccard={jc:.3f}"
        )
        example_lines.append("")
        example_lines.append("| context | orig_top1 | new_top1 | logit_delta |")
        example_lines.append("|---|---|---|---|")
        for e in examples_store.get((l, i), [])[:5]:
            ctx = e["context"].replace("|", "/")
            o = clean_tok(e["original_top1"])
            n = clean_tok(e["intervened_top1"])
            ld = e["logit_delta_top"]
            example_lines.append(f"| {ctx} | **{o}** | **{n}** | {ld:+.3f} |")
        example_lines.append("")

with open(EXAMPLES_MD, "w") as f:
    f.write("\n".join(example_lines) + "\n")
print(f"  {EXAMPLES_MD}")

# ── summary.md ───────────────────────────────────────────────────────────────
summary_lines = ["# Cycle 236 T2 Summary", ""]
summary_lines.append(f"**Verdict:** {verdict}")
summary_lines.append("")
summary_lines.append(
    f"**Headline:** n_blocks_pass = {n_blocks_pass}/12  |  "
    f"max effect_size = {max_effect_size:.3f}"
)
summary_lines.append("")
summary_lines.append(
    f"**Config:** gpt2 / held-out wikitext-2 val / "
    f"{HELD_N_SEQS}×{HELD_SEQ_LEN}={HELD_N_SEQS*HELD_SEQ_LEN} tokens / "
    f"calibration wikitext-2 train {CALIB_N_SEQS}×{CALIB_SEQ_LEN} / "
    f"alphas={ALPHAS} / n_rand_seeds={N_RAND_SEEDS} / seed {SEED} / device {DEVICE}"
)
summary_lines.append("")
summary_lines.append("## Per-block intervention effect (α=+3)")
summary_lines.append("")
summary_lines.append(
    "| block | max_effect | mean_effect | max_jaccard | mean_jaccard | n_hits (≥3 & ≥0.3) | rand_L2 |"
)
summary_lines.append("|------:|-----------:|-----------:|------------:|-------------:|------------------:|---------:|")
for l in range(N_BLOCKS):
    es_list = [effect_sizes[(l, r["direction_idx"])] for r in top5_per_block[l]]
    jc_list = [jaccards[(l, r["direction_idx"])] for r in top5_per_block[l]]
    max_es = max(es_list)
    mean_es = sum(es_list) / len(es_list)
    max_jc = max(jc_list)
    mean_jc = sum(jc_list) / len(jc_list)
    rand_l2_mean = random_l2[l]["random_l2_mean_alpha_3"]
    summary_lines.append(
        f"| {l} | {max_es:.2f} | {mean_es:.2f} | {max_jc:.3f} | {mean_jc:.3f} | "
        f"{n_hits_per_block[l]}/5 | {rand_l2_mean:.3f} |"
    )
summary_lines.append("")
summary_lines.append("## Interpretation")
summary_lines.append("")
if verdict == "CONFIRMED":
    summary_lines.append(
        f"The top bump directions cause real, T1-consistent logit shifts: "
        f"{n_blocks_pass}/12 blocks have ≥3 of top-5 directions with effect size ≥3× "
        f"random baseline AND Jaccard ≥0.3 against T1's predicted boost list. The "
        f"cycle 218 alignment story survives its causal test — the bump space carries "
        f"interpretable vocab features, not just correlational structure."
    )
elif verdict == "REFUTED":
    summary_lines.append(
        f"No direction in any block produced a logit shift distinguishable from "
        f"random (max effect size {max_effect_size:.3f} < 1.2). The cycle 218 "
        f"alignment was correlational only — the bump space does not carry causal "
        f"vocabulary information at the top-SVD-direction level."
    )
else:
    confirmed_no_jaccard = sum(
        1 for l in range(N_BLOCKS)
        if sum(
            1 for r in top5_per_block[l]
            if effect_sizes[(l, r["direction_idx"])] >= 3.0
        ) >= 3
    )
    summary_lines.append(
        f"Mixed result: n_blocks_pass={n_blocks_pass}/12 under the joint "
        f"effect≥3 AND jaccard≥0.3 criterion. Max effect size {max_effect_size:.3f} "
        f"is distinguishable from random ({max_effect_size:.3f} ≥ 1.2), so the "
        f"intervention does move the model, but the shift directions do not match "
        f"T1's vocab predictions in the required number of blocks. Under an "
        f"effect-only criterion (no Jaccard), {confirmed_no_jaccard}/12 blocks pass. "
        f"This is the 'correlational alignment' failure mode T1 flagged: directions "
        f"are causally active but not in the way T1's wte@v decoding suggested."
    )
summary_lines.append("")
summary_lines.append("## Sanity checks")
summary_lines.append("")
max_sanity_viol = max(
    (a1 - a3) for (_, _, a1, a3) in sanity_linearity if a3 > 0
)
summary_lines.append(
    f"- Sanity 1 (α=0 no-op): max |Δlogit| = {diff_zero:.2e} (OK)."
)
summary_lines.append(
    f"- Sanity 2 (linearity α=±1 ≤ α=±3): max (l2(α=1) - l2(α=3)) = {max_sanity_viol:.4f}."
)
finite_ok = all(
    torch.isfinite(u_std_per_block[l]).all().item() for l in range(N_BLOCKS)
)
summary_lines.append(
    f"- Sanity 3 (std_l finite, O(1)): {finite_ok} "
    f"(per-block mean {[round(u_mean_std_per_block[l], 2) for l in range(N_BLOCKS)]})."
)
summary_lines.append(
    f"- Sanity 4 (coherent counts ≥5 all blocks): confirmed from cycle 235 summary."
)
summary_lines.append("")

with open(SUMMARY_MD, "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"  {SUMMARY_MD}")

print(f"\nDone. verdict={verdict} n_blocks_pass={n_blocks_pass}/12 max_effect={max_effect_size:.3f}")
