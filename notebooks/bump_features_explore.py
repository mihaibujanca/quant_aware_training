# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bump Features Exploration
#
# Loads artifacts from the T1-T4 GELU bump interpretability tests (agenda items
# 245-248) and provides interactive exploration cells.
#
# **In the morning**:
# 1. Set the four `CYCLE_*` variables below to the actual cycle numbers the
#    runner produced overnight (check `git log` or
#    `ls autoresearch/results/cycle_*_t1_*`).
# 2. Run all cells from the top.
# 3. Each section first dumps the runner's verdict + headline, then loads the
#    full artifact and shows interactive exploration.
# 4. Use the helper functions at the bottom to do custom interventions on
#    your own prompts.

# %%
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# %%
# CONFIG — set these to match the actual cycle numbers from the overnight run
CYCLE_T1 = None  # e.g. 235
CYCLE_T2 = None  # e.g. 236
CYCLE_T3 = None  # e.g. 237
CYCLE_T4 = None  # e.g. 238

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "autoresearch" / "results"

# Auto-discover most recent cycles for each test if not set explicitly
def _latest_cycle(suffix: str) -> int | None:
    matches = sorted(RESULTS.glob(f"cycle_*_{suffix}"))
    if not matches:
        return None
    return int(matches[-1].name.split("_")[1])

if CYCLE_T1 is None:
    CYCLE_T1 = _latest_cycle("t1_features.pt")
if CYCLE_T2 is None:
    CYCLE_T2 = _latest_cycle("t2_logit_shifts.pt")
if CYCLE_T3 is None:
    CYCLE_T3 = _latest_cycle("t3_ablation_ppl.json")
if CYCLE_T4 is None:
    CYCLE_T4 = _latest_cycle("t4_patching.json")

print(f"T1 cycle: {CYCLE_T1}")
print(f"T2 cycle: {CYCLE_T2}")
print(f"T3 cycle: {CYCLE_T3}")
print(f"T4 cycle: {CYCLE_T4}")

# %% [markdown]
# ## Tokenizer (used by every section to decode token IDs)

# %%
TOK = GPT2TokenizerFast.from_pretrained("gpt2")

def decode_ids(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [TOK.decode([i]) for i in ids]

# %% [markdown]
# ## T1 — Centred bump direction features
#
# For each block, we extracted the top 32 right singular vectors of Δ̃ (the
# centred per-token bump vector). Each direction has a decoded vocab profile
# (what tokens it boosts/suppresses) and a list of corpus tokens that score
# highly on it.
#
# Read this section first. The "headline" tells you whether T1 found
# coherent latent features at all. If T1 says "incoherent" you can stop —
# the story is dead.

# %%
def load_t1(cycle: int):
    if cycle is None:
        raise RuntimeError("CYCLE_T1 not set and no t1_features.pt found")
    features = torch.load(RESULTS / f"cycle_{cycle:03d}_t1_features.pt", weights_only=False)
    with open(RESULTS / f"cycle_{cycle:03d}_t1_decoded.json") as f:
        decoded = json.load(f)
    summary = (RESULTS / f"cycle_{cycle:03d}_t1_summary.md").read_text()
    examples = (RESULTS / f"cycle_{cycle:03d}_t1_examples.md").read_text()
    return {"features": features, "decoded": decoded, "summary": summary, "examples": examples}

t1 = load_t1(CYCLE_T1)
print(t1["summary"])

# %% [markdown]
# ### Per-block top-5 directions (full text from runner's `_examples.md`)

# %%
print(t1["examples"])

# %% [markdown]
# ### Coherence label distribution per block

# %%
from collections import Counter
for block in range(12):
    labels = [d["coherence_label"] for d in t1["decoded"] if d["block"] == block]
    print(f"block {block:2d}: {Counter(labels).most_common()}")

# %% [markdown]
# ### Pick a (block, direction) and inspect it interactively

# %%
def inspect_direction(block: int, direction_idx: int):
    rec = next(
        d for d in t1["decoded"]
        if d["block"] == block and d["direction_idx"] == direction_idx
    )
    print(f"=== block {block} direction {direction_idx} ===")
    print(f"singular value: {rec['singular_value']:.2f}")
    print(f"variance explained: {rec['variance_explained']:.3%}")
    print(f"coherence label: {rec['coherence_label']}")
    print()
    print("BOOST top 20:")
    for v in rec["top_vocab_boost"][:20]:
        print(f"  {v['token_str']:20s} {v['score']:+.3f}")
    print()
    print("SUPPRESS top 20:")
    for v in rec["top_vocab_suppress"][:20]:
        print(f"  {v['token_str']:20s} {v['score']:+.3f}")
    print()
    print("Top corpus tokens (with context):")
    for t in rec["top_corpus_tokens"][:5]:
        print(f"  ...{t['context_left']}[{t['token_str']}]{t['context_right']}...  score={t['score']:+.3f}")

# Example: inspect block 5, direction 0
# inspect_direction(5, 0)

# %% [markdown]
# ## T2 — Causal intervention results
#
# We added α · v_{l,i} to the residual stream after block l for each top
# direction and measured the logit shift. T2 confirms or kills the story:
# if T1 said "this direction boosts commas" and T2 doesn't actually shift
# the comma logit when you add α·v, the alignment was correlational only.

# %%
def load_t2(cycle: int):
    if cycle is None:
        return None
    shifts = torch.load(RESULTS / f"cycle_{cycle:03d}_t2_logit_shifts.pt", weights_only=False)
    with open(RESULTS / f"cycle_{cycle:03d}_t2_intervention.json") as f:
        intervention = json.load(f)
    summary = (RESULTS / f"cycle_{cycle:03d}_t2_summary.md").read_text()
    examples = (RESULTS / f"cycle_{cycle:03d}_t2_examples.md").read_text()
    return {"shifts": shifts, "intervention": intervention, "summary": summary, "examples": examples}

t2 = load_t2(CYCLE_T2)
if t2 is not None:
    print(t2["summary"])

# %% [markdown]
# ### Concrete intervention examples (the part you actually want to read)

# %%
if t2 is not None:
    print(t2["examples"])

# %% [markdown]
# ### T1 ↔ T2 cross-check
# For each direction, did the actual logit shift match what T1 predicted?
# Jaccard overlap between T1's top-20 boost list and T2's top-20 boosted
# vocab at α=+3.

# %%
if t2 is not None:
    from collections import defaultdict

    overlaps = defaultdict(list)
    for entry in t2["intervention"]:
        if entry.get("alpha") != 3:
            continue
        block = entry["block"]
        overlaps[block].append(entry.get("t1_jaccard_overlap", float("nan")))

    print("Block | mean Jaccard | max Jaccard | n directions")
    for b in sorted(overlaps):
        ov = overlaps[b]
        mean = sum(ov) / max(len(ov), 1)
        print(f"  {b:3d} |  {mean:.3f}      |  {max(ov):.3f}     | {len(ov)}")

# %% [markdown]
# ## T3 — Subspace ablation results
#
# We projected out the top-10 bump directions from the residual stream after
# each block and measured PPL. Compared against random k=10 subspaces and
# the top-10 directions of the FULL FFN output.

# %%
def load_t3(cycle: int):
    if cycle is None:
        return None
    with open(RESULTS / f"cycle_{cycle:03d}_t3_ablation_ppl.json") as f:
        data = json.load(f)
    summary = (RESULTS / f"cycle_{cycle:03d}_t3_summary.md").read_text()
    return {"data": data, "summary": summary}

t3 = load_t3(CYCLE_T3)
if t3 is not None:
    print(t3["summary"])

# %% [markdown]
# ### Per-block ablation table (bump vs random vs full-FFN)

# %%
if t3 is not None:
    data = t3["data"]
    print(f"baseline PPL: {data['baseline']:.3f}")
    print(f"joint bump (12 blocks): {data['joint_bump']:.3f}")
    print(f"joint random baseline: {data['random_baseline']['joint_mean']:.3f} ± {data['random_baseline']['joint_std']:.3f}")
    print()
    print("Block | bump  | random (mean±std) | full-FFN top10")
    for b in range(12):
        pb = data["per_block_bump"][b]
        pr = data["random_baseline"]["per_block_mean"][b]
        prs = data["random_baseline"]["per_block_std"][b]
        pf = data["full_ffn_top10"][b]
        print(f"  {b:3d} | {pb:5.2f} | {pr:5.2f} ± {prs:.2f}      | {pf:5.2f}")

# %% [markdown]
# ## T4 — Activation patching across token pairs
#
# For pairs (A, B) of tokens with different next-token predictions, we
# patched only the bump component from B into A and measured how much of
# the prediction difference was carried by that patch.

# %%
def load_t4(cycle: int):
    if cycle is None:
        return None
    with open(RESULTS / f"cycle_{cycle:03d}_t4_patching.json") as f:
        data = json.load(f)
    summary = (RESULTS / f"cycle_{cycle:03d}_t4_summary.md").read_text()
    examples = (RESULTS / f"cycle_{cycle:03d}_t4_examples.md").read_text()
    return {"data": data, "summary": summary, "examples": examples}

t4 = load_t4(CYCLE_T4)
if t4 is not None:
    print(t4["summary"])

# %% [markdown]
# ### Concrete patching examples

# %%
if t4 is not None:
    print(t4["examples"])

# %% [markdown]
# ## Custom interventions
#
# Below are the building blocks for running your own intervention on a
# custom prompt. Loads GPT-2, sets up hooks, and applies α · v_{l,i} from
# T1's saved feature tensors.

# %%
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    return model

# Lazy: only load if user actually wants to do interventions
_model = None
def model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def get_direction(block: int, direction_idx: int) -> torch.Tensor:
    """Return v_{block, direction_idx} ∈ R^768."""
    V = t1["features"][f"block_{block}"]["V"]  # (768, 32)
    return V[:, direction_idx].to(DEVICE)

def intervene_on_prompt(prompt: str, block: int, direction_idx: int, alpha: float, top_k: int = 10):
    """Apply α · v_{block, direction_idx} to the residual stream after block,
    return top-k predictions for the next token before vs after."""
    m = model()
    tok_ids = TOK.encode(prompt, return_tensors="pt").to(DEVICE)

    v = get_direction(block, direction_idx)

    captured = {}
    def add_v(module, inp, out):
        # GPT-2 block forward returns (hidden_states, ...). Modify hidden states.
        if isinstance(out, tuple):
            h = out[0]
            captured["original"] = h.clone()
            h_new = h + alpha * v
            return (h_new,) + out[1:]
        else:
            captured["original"] = out.clone()
            return out + alpha * v

    handle = m.transformer.h[block].register_forward_hook(add_v)
    try:
        with torch.no_grad():
            out_intervened = m(tok_ids)
    finally:
        handle.remove()

    with torch.no_grad():
        out_orig = m(tok_ids)

    last_pos = tok_ids.shape[1] - 1
    logits_orig = out_orig.logits[0, last_pos]
    logits_int = out_intervened.logits[0, last_pos]

    top_orig = torch.topk(logits_orig, top_k)
    top_int = torch.topk(logits_int, top_k)

    print(f"prompt: {prompt!r}")
    print(f"intervention: block={block} direction={direction_idx} alpha={alpha}")
    print()
    print(f"{'rank':<6}{'original':<25}{'intervened':<25}")
    for i in range(top_k):
        ot = TOK.decode([top_orig.indices[i].item()])
        it = TOK.decode([top_int.indices[i].item()])
        print(f"{i:<6}{ot!r:<25}{it!r:<25}")

# Example usage (uncomment in the morning):
# intervene_on_prompt("The cat sat on the", block=5, direction_idx=0, alpha=3.0)
# intervene_on_prompt("The cat sat on the", block=5, direction_idx=0, alpha=-3.0)
