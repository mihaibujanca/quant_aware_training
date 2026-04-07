"""
Cycle 235 — Item 245 (T1): Centred Δ̃ feature SVD with concrete example extraction.

For each of 12 GPT-2 blocks, on a wikitext-2 train split calibration corpus
(N_SEQS=200, SEQ_LEN=512), compute Δ̃_l = (GELU - ReLU)(post) @ W_proj - mean_l
per token, SVD → top 32 right singular vectors V_l. Decode each direction to
vocab via W_U and to corpus via |U_l[:, i]|, label heuristic coherence, and save
artifacts loadable by notebooks/bump_features_explore.py.

Verdict:
  CONFIRMED   := ≥6 of 12 blocks have ≥5 directions with coherence_label ∉ {mixed, incoherent}
  REFUTED     := all 12 blocks have <2 coherent directions
  INCONCLUSIVE := otherwise
"""
import json
import string
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
N_SEQS = 200
SEQ_LEN = 512
N_BLOCKS = 12
TOP_K_DIRS = 32
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CPU = torch.device("cpu")

RESULTS_DIR = "autoresearch/results"
FEATURES_PATH = f"{RESULTS_DIR}/cycle_235_t1_features.pt"
META_PATH = f"{RESULTS_DIR}/cycle_235_t1_calibration_meta.pt"
DECODED_JSON = f"{RESULTS_DIR}/cycle_235_t1_decoded.json"
EXAMPLES_MD = f"{RESULTS_DIR}/cycle_235_t1_examples.md"
SUMMARY_MD = f"{RESULTS_DIR}/cycle_235_t1_summary.md"

STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "or", "but", "is", "are",
    "was", "were", "for", "on", "at", "with", "by", "as", "that", "this",
    "it", "be", "from", "which", "he", "she", "they", "we", "you", "i",
    "not", "have", "has", "had", "do", "does", "did",
}
PUNCT = set(string.punctuation)

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Calibration corpus: wikitext-2 train split, concat+chunk ─────────────────
split_used = "train"
fallback = False
try:
    print("Loading wikitext-2 train split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
except Exception as e:
    print(f"WARNING: train split load failed ({e}); using val split as fallback")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    split_used = "validation"
    fallback = True

texts = [r["text"] for r in ds if r["text"].strip()]
print(f"Lines (non-empty): {len(texts)}")

# Tokenize each non-empty line, then concatenate all token ids into one long stream.
all_tokens = []
tokens_needed = N_SEQS * SEQ_LEN
for t in texts:
    ids = tokenizer(t, return_tensors="pt", truncation=False)["input_ids"][0]
    all_tokens.append(ids)
    # Rough early-exit: check running length every few thousand lines
    if len(all_tokens) % 1000 == 0:
        running = sum(x.numel() for x in all_tokens)
        if running >= tokens_needed * 2:
            # Already more than enough, stop tokenizing
            break

flat = torch.cat(all_tokens, dim=0)
print(f"Total tokens tokenised: {flat.numel()}")
if flat.numel() < tokens_needed:
    if not fallback:
        print(f"WARNING: train split yielded only {flat.numel()} tokens (need {tokens_needed}); "
              "falling back to val split.")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [r["text"] for r in ds if r["text"].strip()]
        all_tokens = []
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=False)["input_ids"][0]
            all_tokens.append(ids)
        flat = torch.cat(all_tokens, dim=0)
        split_used = "validation"
        fallback = True
        print(f"Fallback total tokens: {flat.numel()}")
    if flat.numel() < tokens_needed:
        raise RuntimeError(f"Need {tokens_needed} tokens but only have {flat.numel()}")

flat = flat[:tokens_needed].contiguous()
calib_ids = flat.view(N_SEQS, SEQ_LEN).contiguous()
print(f"calib_ids shape: {tuple(calib_ids.shape)}  split={split_used}  fallback={fallback}")

token_ids_all = calib_ids.flatten().clone()  # (102400,)
seq_index_all = torch.arange(N_SEQS).unsqueeze(1).expand(-1, SEQ_LEN).flatten().clone()
pos_in_seq_all = torch.arange(SEQ_LEN).unsqueeze(0).expand(N_SEQS, -1).flatten().clone()
n_tokens = token_ids_all.numel()
assert n_tokens == N_SEQS * SEQ_LEN == 102400, n_tokens

# ── Hooks: capture MLP c_fc INPUT (post-LN2 residual) per block ──────────────
cfc_stash = {l: [] for l in range(N_BLOCKS)}

def make_cfc_prehook(l):
    def fn(module, args):
        cfc_stash[l].append(args[0].detach().float().cpu())
    return fn

hooks = []
for l in range(N_BLOCKS):
    h = model.transformer.h[l].mlp.c_fc.register_forward_pre_hook(make_cfc_prehook(l))
    hooks.append(h)

print("Running forward pass in batches...")
BATCH = 8
with torch.no_grad():
    for i in range(0, N_SEQS, BATCH):
        batch = calib_ids[i:i+BATCH].to(DEVICE)
        _ = model(batch)
for h in hooks:
    h.remove()

mlp_inputs = {}
for l in range(N_BLOCKS):
    t = torch.cat(cfc_stash[l], dim=0).reshape(-1, 768)
    assert t.shape == (n_tokens, 768), (l, t.shape)
    mlp_inputs[l] = t
cfc_stash = None
print(f"Captured mlp_inputs[0] shape: {tuple(mlp_inputs[0].shape)}")

# ── Per-block Δ̃ + SVD ───────────────────────────────────────────────────────
features_dict = {}
U_top_per_block = {}
S_full_per_block = {}

for l in range(N_BLOCKS):
    print(f"\n── Block {l} ──")
    block = model.transformer.h[l]
    W_fc = block.mlp.c_fc.weight.detach().float().cpu()      # (768, 3072)
    b_fc = block.mlp.c_fc.bias.detach().float().cpu()        # (3072,)
    W_proj = block.mlp.c_proj.weight.detach().float().cpu()  # (3072, 768)

    x = mlp_inputs[l]  # (n_tokens, 768)
    chunk = 4096
    parts = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i+chunk]
        preact = xc @ W_fc + b_fc
        gelu_post = F.gelu(preact, approximate="tanh")
        relu_post = F.relu(preact)
        delta_post = gelu_post - relu_post
        parts.append(delta_post @ W_proj)
    delta_out = torch.cat(parts, dim=0)  # (n_tokens, 768)
    parts = None

    assert not torch.isnan(delta_out).any(), f"NaN in delta_out block {l}"
    mean_delta = delta_out.mean(0).contiguous()
    delta_centred = delta_out - mean_delta
    delta_out = None

    print(f"  SVD on delta_centred {tuple(delta_centred.shape)}...")
    U, S, Vh = torch.linalg.svd(delta_centred, full_matrices=False)
    delta_centred = None

    V_top32 = Vh[:TOP_K_DIRS].T.contiguous()     # (768, 32)
    S_top32 = S[:TOP_K_DIRS].contiguous().clone()  # (32,)
    U_top32 = U[:, :TOP_K_DIRS].contiguous().clone()  # (n_tokens, 32)

    assert V_top32.shape == (768, TOP_K_DIRS)
    assert S_top32.shape == (TOP_K_DIRS,)
    assert mean_delta.shape == (768,)

    sigma0 = float(S_top32[0].item())
    sigma1 = float(S_top32[1].item())
    print(f"  σ₀={sigma0:.2f}, σ₁={sigma1:.2f}, ||mean_Δ||={mean_delta.norm().item():.3f}")

    features_dict[f"block_{l}"] = {
        "V": V_top32,
        "S": S_top32,
        "mean_delta": mean_delta,
    }
    U_top_per_block[l] = U_top32
    S_full_per_block[l] = S.clone()

    U = None
    Vh = None
    S = None

    # Free the captured mlp_inputs[l] after use to save memory
    mlp_inputs[l] = None

mlp_inputs = None

# ── Save features + calibration meta ─────────────────────────────────────────
print("\nSaving features + calibration meta...")
torch.save(features_dict, FEATURES_PATH)
torch.save(
    {"token_ids": token_ids_all, "seq_index": seq_index_all, "pos_in_seq": pos_in_seq_all},
    META_PATH,
)

# Reload sanity check
feats_reloaded = torch.load(FEATURES_PATH, weights_only=False)
for l in range(N_BLOCKS):
    bd = feats_reloaded[f"block_{l}"]
    assert bd["V"].shape == (768, TOP_K_DIRS)
    assert bd["S"].shape == (TOP_K_DIRS,)
    assert bd["mean_delta"].shape == (768,)
print(f"  {FEATURES_PATH} shape OK")
meta_reloaded = torch.load(META_PATH, weights_only=False)
for k, v in meta_reloaded.items():
    assert v.shape == (n_tokens,), (k, v.shape)
print(f"  {META_PATH} shape OK")

# ── Cycle 218 sanity cross-check ─────────────────────────────────────────────
c218_path = "autoresearch/results/cycle_218_delta_centred_alignment.json"
try:
    import os
    if os.path.exists(c218_path):
        with open(c218_path) as f:
            c218 = json.load(f)
        # c218 doesn't save vh_top20 to JSON; just log sigmas for comparison
        print("\n── Cycle 218 sanity (σ₀, σ₁ comparison) ──")
        for l in range(N_BLOCKS):
            b218 = c218["blocks"][l]
            cs0 = float(features_dict[f"block_{l}"]["S"][0])
            cs1 = float(features_dict[f"block_{l}"]["S"][1])
            print(f"  block {l}: c218 σ₀={b218['sigma0']:.1f} σ₁={b218['sigma1']:.1f} | "
                  f"c235 σ₀={cs0:.1f} σ₁={cs1:.1f}")
    else:
        print("(cycle 218 JSON not found — skipping sanity cross-check)")
except Exception as e:
    print(f"(cycle 218 sanity skipped: {e})")

# ── Vocab decoding setup ─────────────────────────────────────────────────────
print("\nVocab decoding setup...")
wte = model.transformer.wte.weight.detach().float().cpu()  # (50257, 768)
tok_cache: dict = {}

def decode_tok(tid: int) -> str:
    if tid not in tok_cache:
        tok_cache[tid] = tokenizer.decode([tid])
    return tok_cache[tid]

# ── Coherence classifier ─────────────────────────────────────────────────────
def classify_coherence(top_corpus_tokens):
    token_strs = [t["token_str"] for t in top_corpus_tokens]
    token_ids_list = [t["token_id"] for t in top_corpus_tokens]
    stripped = [s.strip() for s in token_strs]

    n_punct = sum(
        1 for s in stripped
        if len(s) > 0 and all(ch in PUNCT for ch in s)
    )
    n_stop = sum(1 for s in stripped if s.lower() in STOPWORDS)
    n_cap = sum(1 for s in stripped if len(s) >= 2 and s[0].isupper())

    def is_number(s):
        if not s:
            return False
        if s.isdigit():
            return True
        rep = s.replace(".", "", 1)
        return rep.isdigit()

    n_num = sum(1 for s in stripped if is_number(s))
    mode_count = Counter(token_ids_list).most_common(1)[0][1] if token_ids_list else 0

    if mode_count >= 30:
        return "repeated"
    if n_punct >= 40:
        return "punctuation"
    if n_stop >= 40:
        return "stopword"
    if n_cap >= 40:
        return "capitalized"
    if n_num >= 40:
        return "number"
    max_class = max(n_punct, n_stop, n_cap, n_num, mode_count)
    if max_class >= 20:
        return "mixed"
    return "incoherent"

# ── Per-direction decoding loop ──────────────────────────────────────────────
print("\nDecoding 12 blocks × 32 directions = 384 records...")
decoded_records = []
for l in range(N_BLOCKS):
    V_l = features_dict[f"block_{l}"]["V"]  # (768, 32)
    S_l = features_dict[f"block_{l}"]["S"]  # (32,)
    S_full_l = S_full_per_block[l]  # (768,)
    total_energy = float((S_full_l ** 2).sum().item())
    U_l = U_top_per_block[l]  # (n_tokens, 32)

    for i in range(TOP_K_DIRS):
        u_col = U_l[:, i]  # (n_tokens,)
        score_abs = u_col.abs()
        top_vals, top_idx = torch.topk(score_abs, 100)

        top_corpus_tokens = []
        for k in range(100):
            idx = int(top_idx[k].item())
            seq_idx = int(seq_index_all[idx].item())
            pos = int(pos_in_seq_all[idx].item())
            tid = int(token_ids_all[idx].item())
            tok_str = decode_tok(tid)
            left_ids = calib_ids[seq_idx, max(0, pos - 5):pos].tolist()
            right_ids = calib_ids[seq_idx, pos + 1:min(SEQ_LEN, pos + 6)].tolist()
            context_left = tokenizer.decode(left_ids)
            context_right = tokenizer.decode(right_ids)
            signed_score = float(u_col[idx].item())
            top_corpus_tokens.append({
                "token_id": tid,
                "token_str": tok_str,
                "context_left": context_left,
                "context_right": context_right,
                "position": pos,
                "score": signed_score,
            })

        v = V_l[:, i]  # (768,)
        logit_impact = wte @ v  # (50257,)
        boost_vals, boost_idx = torch.topk(logit_impact, 50)
        sup_vals, sup_idx = torch.topk(-logit_impact, 50)
        top_vocab_boost = [
            {
                "token_id": int(boost_idx[k].item()),
                "token_str": decode_tok(int(boost_idx[k].item())),
                "score": float(boost_vals[k].item()),
            }
            for k in range(50)
        ]
        top_vocab_suppress = [
            {
                "token_id": int(sup_idx[k].item()),
                "token_str": decode_tok(int(sup_idx[k].item())),
                "score": -float(sup_vals[k].item()),
            }
            for k in range(50)
        ]

        variance_explained = float((S_l[i] ** 2).item() / total_energy)
        singular_value = float(S_l[i].item())
        coherence_label = classify_coherence(top_corpus_tokens)

        decoded_records.append({
            "block": l,
            "direction_idx": i,
            "singular_value": singular_value,
            "variance_explained": variance_explained,
            "top_corpus_tokens": top_corpus_tokens,
            "top_vocab_boost": top_vocab_boost,
            "top_vocab_suppress": top_vocab_suppress,
            "coherence_label": coherence_label,
        })
    print(f"  block {l} done")

assert len(decoded_records) == N_BLOCKS * TOP_K_DIRS == 384

with open(DECODED_JSON, "w") as f:
    json.dump(decoded_records, f, indent=2)
print(f"\nSaved decoded JSON → {DECODED_JSON}")

# ── Per-block coherent counts + verdict ──────────────────────────────────────
INCOH = {"mixed", "incoherent"}
coherent_per_block = [0] * N_BLOCKS
label_counts_per_block = [Counter() for _ in range(N_BLOCKS)]
for rec in decoded_records:
    b = rec["block"]
    lab = rec["coherence_label"]
    label_counts_per_block[b][lab] += 1
    if lab not in INCOH:
        coherent_per_block[b] += 1

n_blocks_5plus = sum(1 for c in coherent_per_block if c >= 5)
all_below_2 = all(c < 2 for c in coherent_per_block)
if n_blocks_5plus >= 6:
    verdict = "CONFIRMED"
elif all_below_2:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"\nVerdict: {verdict}")
print(f"coherent_per_block: {coherent_per_block}")
print(f"n_blocks_5plus: {n_blocks_5plus}/12")

# ── Build examples.md (top-5 directions per block by variance_explained) ────
def clean_ctx(s: str, max_len: int = 60) -> str:
    s = s.replace("\n", " ").replace("|", "/").strip()
    if len(s) > max_len:
        s = s[:max_len] + "…"
    return s

def clean_tok(s: str) -> str:
    return s.replace("\n", "\\n").replace("|", "/")

example_lines = ["# Cycle 235 T1 — Per-block top-5 direction examples", ""]
example_lines.append(f"Verdict: **{verdict}**  |  "
                     f"Coherent≥5 blocks: **{n_blocks_5plus}/12**  |  "
                     f"split={split_used}  tokens={n_tokens}")
example_lines.append("")
for l in range(N_BLOCKS):
    block_recs = [r for r in decoded_records if r["block"] == l]
    block_recs_sorted = sorted(block_recs, key=lambda r: -r["variance_explained"])
    top5 = block_recs_sorted[:5]
    example_lines.append(f"## Block {l}")
    example_lines.append("")
    example_lines.append(f"Coherent directions: {coherent_per_block[l]}/32  |  "
                         f"labels: {dict(label_counts_per_block[l])}")
    example_lines.append("")
    for rec in top5:
        i = rec["direction_idx"]
        var = rec["variance_explained"]
        sigma = rec["singular_value"]
        lab = rec["coherence_label"]
        example_lines.append(
            f"### Block {l} dir {i} — var={var:.2%} σ={sigma:.2f} coherence={lab}"
        )
        example_lines.append("")
        example_lines.append("| context_left | TOKEN | context_right |")
        example_lines.append("|---|---|---|")
        for t in rec["top_corpus_tokens"][:3]:
            cl = clean_ctx(t["context_left"])
            cr = clean_ctx(t["context_right"])
            tok = clean_tok(t["token_str"])
            example_lines.append(f"| {cl} | **{tok}** | {cr} |")
        example_lines.append("")
        boost10 = ", ".join(clean_tok(v["token_str"]) for v in rec["top_vocab_boost"][:10])
        sup10 = ", ".join(clean_tok(v["token_str"]) for v in rec["top_vocab_suppress"][:10])
        example_lines.append(f"- boost: [{boost10}]")
        example_lines.append(f"- suppress: [{sup10}]")
        example_lines.append("")

with open(EXAMPLES_MD, "w") as f:
    f.write("\n".join(example_lines) + "\n")
print(f"Saved examples → {EXAMPLES_MD}")

# ── Build summary.md ─────────────────────────────────────────────────────────
summary_lines = ["# Cycle 235 T1 Summary", ""]
summary_lines.append(f"**Verdict:** {verdict}")
summary_lines.append("")
summary_lines.append(
    f"**Headline:** {n_blocks_5plus}/12 blocks have ≥5 coherent directions "
    f"(coherence_label ∉ {{mixed, incoherent}})."
)
summary_lines.append("")
summary_lines.append(f"**Config:** gpt2 / wikitext-2 {split_used} split / "
                     f"N_SEQS={N_SEQS} × SEQ_LEN={SEQ_LEN} = {n_tokens} tokens / "
                     f"seed {SEED} / device {DEVICE}" +
                     (" (FALLBACK to val)" if fallback else ""))
summary_lines.append("")
summary_lines.append("## Per-block coherent direction counts")
summary_lines.append("")
summary_lines.append("| block | coherent | total | labels |")
summary_lines.append("|------:|---------:|------:|:-------|")
for l in range(N_BLOCKS):
    labs = dict(label_counts_per_block[l])
    labs_str = ", ".join(f"{k}:{v}" for k, v in sorted(labs.items(), key=lambda kv: -kv[1]))
    summary_lines.append(f"| {l} | {coherent_per_block[l]} | {TOP_K_DIRS} | {labs_str} |")
summary_lines.append("")
summary_lines.append("## Interpretation")
summary_lines.append("")
if verdict == "CONFIRMED":
    summary_lines.append(
        f"The centred per-token bump space contains interpretable latent features: "
        f"{n_blocks_5plus}/12 blocks have ≥5 of 32 top directions whose top-100 "
        f"corpus tokens form a coherent heuristic class (punctuation / stopword / "
        f"capitalized / number / repeated). Coherence here is a cheap surface heuristic "
        f"— a direction labelled 'punctuation' means ≥40 of its top-100 corpus tokens "
        f"are single-character punctuation, not that the direction causally boosts "
        f"punctuation logits. T2 (item 246) is the causal test: it adds α·v_{{l,i}} to "
        f"the residual stream and checks whether the actual logit shift matches the T1 "
        f"vocab prediction. Until then, these features are a correlational feature "
        f"inventory — the story is alive but unproven."
    )
elif verdict == "REFUTED":
    summary_lines.append(
        f"All 12 blocks have <2 directions whose top-100 corpus tokens form a coherent "
        f"heuristic class. The top-32 SVD basis of the centred per-token bump space "
        f"looks like noise under this surface-level labelling, so the cycle 218 "
        f"alignment story (Δ̃ writes into downstream reading subspaces) has no per-token "
        f"interpretable feature backing at this level — T2/T3/T4 are unlikely to find a "
        f"clean causal signal unless they search outside the top-32 SVD basis or abandon "
        f"the per-token-feature framing."
    )
else:
    summary_lines.append(
        f"Mixed result: {n_blocks_5plus}/12 blocks pass the ≥5-coherent threshold and "
        f"not all 12 blocks fall below 2 coherent directions. Some blocks may carry "
        f"clean punctuation/stopword/number features while others look like noise. "
        f"T2 (item 246) should focus its intervention budget on the highest-coherence "
        f"blocks first; coherence labels here are a cheap surface heuristic and the "
        f"causal test is required to separate real features from artefacts."
    )
summary_lines.append("")

with open(SUMMARY_MD, "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"Saved summary → {SUMMARY_MD}")

print("\nDone.")
