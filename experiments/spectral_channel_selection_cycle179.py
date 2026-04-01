"""Cycle 179 / Item 189: PruneNet spectral channel selection — GPT-2 FFN k=4.

Hypothesis: selecting FFN channels that preserve the singular value distribution
of W_down (output projection) outperforms activation-magnitude topk. The spectral
criterion is calibration-free (weight-only); lstsq fitting still uses data.

Two conditions (4 blocks each: [1,4,7,11]):
  A) Standard topk+lstsq+polish (control)
  B) Spectral greedy selection + lstsq + polish

Verdict:
  CONFIRMED    if spectral PPL <= topk PPL - 2
  REFUTED      if spectral PPL >= topk PPL + 2
  INCONCLUSIVE if within +/-2 PPL
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressedMLP,
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    FfnLayerGeometry,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _apply_activation,
    _evaluate_model,
    _fit_output_projection,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
POLISH_STEPS = 500
POLISH_LR = 1e-4

GPT2_FFN_HIDDEN = 3072
GPT2_MODEL_DIM = 768
BLOCKS = [1, 4, 7, 11]
TOPK_HIDDEN = 512  # k=4 means 512 channels (3072/6)

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Spectral greedy channel selection
# ---------------------------------------------------------------------------
def _ks_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """KS-distance between two 1D distributions (empirical CDFs)."""
    a_sorted = torch.sort(a)[0]
    b_sorted = torch.sort(b)[0]
    # Normalize to [0,1] CDFs
    n_a = a_sorted.shape[0]
    n_b = b_sorted.shape[0]
    # Merge and compute CDFs
    combined = torch.cat([a_sorted, b_sorted])
    combined_sorted = torch.sort(combined)[0]
    cdf_a = torch.searchsorted(a_sorted, combined_sorted).float() / n_a
    cdf_b = torch.searchsorted(b_sorted, combined_sorted).float() / n_b
    return float((cdf_a - cdf_b).abs().max().item())


def spectral_greedy_select(W: torch.Tensor, k: int, dead_mask: torch.Tensor,
                           verbose: bool = True) -> tuple[torch.Tensor, dict]:
    """Greedily select k columns of W (768, 3072) to preserve SVD spectrum.

    Returns (selected_indices, stats_dict).
    """
    t0 = time.time()
    n_out, n_hidden = W.shape  # 768, 3072

    # Reference spectrum
    sigma_ref = torch.linalg.svdvals(W.float())  # (768,)

    # Candidate pool: exclude dead channels
    live_mask = ~dead_mask
    candidates = torch.nonzero(live_mask, as_tuple=False).squeeze(1)
    n_live = candidates.shape[0]

    if n_live <= k:
        P(f"    Warning: only {n_live} live channels, returning all")
        return candidates.sort().values, {"ks_distance": 0.0, "time_s": 0.0}

    # Pre-compute column norms for candidate filtering
    col_norms = W[:, candidates].norm(dim=0)  # (n_live,)

    # Seed: channel most aligned with top singular vector
    _, _, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    top_v = Vh[0]  # (3072,) top right singular vector
    alignments = (W[:, candidates].float().t() @ W.float() @ top_v.unsqueeze(1)).squeeze(1).abs()
    seed_idx = int(alignments.argmax().item())
    selected_set = {int(candidates[seed_idx].item())}
    remaining = set(range(n_live)) - {seed_idx}

    if verbose:
        P(f"    Seed channel: {candidates[seed_idx].item()} "
          f"(alignment={alignments[seed_idx].item():.4f})")

    # Greedy loop
    # For speed: at each step, restrict candidates to top-256 by column norm
    # among remaining, then compute exact KS for those
    CANDIDATE_BATCH = 256

    for step in range(1, k):
        selected_list = sorted(selected_set)
        W_sel = W[:, selected_list].float()

        if len(remaining) <= CANDIDATE_BATCH:
            trial_pool = list(remaining)
        else:
            # Restrict to top candidates by column norm
            rem_list = list(remaining)
            rem_norms = col_norms[rem_list]
            top_idx = torch.topk(rem_norms, k=min(CANDIDATE_BATCH, len(rem_list)),
                                 largest=True).indices
            trial_pool = [rem_list[int(i)] for i in top_idx]

        best_ks = float('inf')
        best_c = trial_pool[0]

        # Current spectrum for reference
        sigma_sel = torch.linalg.svdvals(W_sel)

        for c_idx in trial_pool:
            c_global = int(candidates[c_idx].item())
            trial_cols = selected_list + [c_global]
            W_trial = W[:, sorted(trial_cols)].float()
            sigma_trial = torch.linalg.svdvals(W_trial)
            # Pad to match ref length for KS
            if sigma_trial.shape[0] < sigma_ref.shape[0]:
                sigma_trial = F.pad(sigma_trial,
                                    (0, sigma_ref.shape[0] - sigma_trial.shape[0]))
            ks = _ks_distance(sigma_trial, sigma_ref)
            if ks < best_ks:
                best_ks = ks
                best_c = c_idx

        selected_set.add(int(candidates[best_c].item()))
        remaining.discard(best_c)

        if verbose and (step % 100 == 0 or step == k - 1):
            elapsed = time.time() - t0
            P(f"    Step {step}/{k}: KS={best_ks:.6f}, "
              f"elapsed={elapsed:.1f}s")

    selected = torch.tensor(sorted(selected_set), dtype=torch.long)
    # Final KS
    sigma_final = torch.linalg.svdvals(W[:, selected].float())
    if sigma_final.shape[0] < sigma_ref.shape[0]:
        sigma_final = F.pad(sigma_final,
                            (0, sigma_ref.shape[0] - sigma_final.shape[0]))
    final_ks = _ks_distance(sigma_final, sigma_ref)
    elapsed = time.time() - t0

    stats = {
        "ks_distance": round(final_ks, 6),
        "time_s": round(elapsed, 1),
        "n_candidates": n_live,
    }
    if verbose:
        P(f"    Final: {k} channels selected, KS={final_ks:.6f}, "
          f"time={elapsed:.1f}s")
    return selected, stats


# ---------------------------------------------------------------------------
# Helper: find MLP in GPT-2 block
# ---------------------------------------------------------------------------
def _find_mlp(model, block_id):
    return model.transformer.h[block_id].mlp


# ---------------------------------------------------------------------------
# Joint polish (same as cycle 178)
# ---------------------------------------------------------------------------
def _joint_polish(student_model, block_ids, polish_tokens, n_steps, lr, device, seed):
    """Fine-tune c_proj weights of compressed blocks on LM loss."""
    if n_steps == 0:
        return []

    student_model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = _find_mlp(student_model, bid)
        if hasattr(mlp, "c_proj"):
            w = mlp.c_proj.weight
            w.requires_grad_(True)
            params.append(w)
            b = getattr(mlp.c_proj, "bias", None)
            if b is not None:
                b.requires_grad_(True)
                params.append(b)

    if not params:
        return []

    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()
    losses = []
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = raw.logits if hasattr(raw, "logits") else raw[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


# ---------------------------------------------------------------------------
# Build compressed models
# ---------------------------------------------------------------------------
def _build_compressed_model(teacher, atlas, block_ids, selected_per_block, device):
    """Build a compressed model using given channel selections per block."""
    model = copy.deepcopy(teacher.model)
    model.eval()
    stats = {}

    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        bid = layer.layer_id
        selected = selected_per_block[bid]
        k = selected.shape[0]

        source_w_in = _standard_weight(layer.handle.c_fc).float()
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        hidden = layer.hidden_sample[:, selected].float()
        outputs = layer.output_sample.float()
        w_out, b_out = _fit_output_projection(hidden, outputs, use_bias=use_output_bias)

        new_mlp = CompressedMLP(
            input_dim=source_w_in.shape[1],
            hidden_dim=k,
            output_dim=w_out.shape[0],
            activation_kind=layer.activation_kind,
            c_fc_bias=source_b_in is not None,
            c_proj_bias=b_out is not None,
        )
        with torch.no_grad():
            new_mlp.c_fc.weight.copy_(source_w_in[selected])
            if source_b_in is not None:
                new_mlp.c_fc.bias.copy_(source_b_in[selected])
            new_mlp.c_proj.weight.copy_(w_out)
            if b_out is not None:
                new_mlp.c_proj.bias.copy_(b_out)

        new_mlp.to(device)
        model.transformer.h[bid].mlp = new_mlp

        # lstsq residual
        with torch.no_grad():
            pred = hidden @ w_out.t()
            if b_out is not None:
                pred = pred + b_out
            mse = F.mse_loss(pred, outputs).item()

        n_params = k * GPT2_MODEL_DIM + k + GPT2_MODEL_DIM * k + GPT2_MODEL_DIM
        stats[bid] = {
            "n_channels": k,
            "lstsq_mse": round(mse, 6),
            "params": n_params,
        }
        P(f"    Block {bid}: {k} channels, lstsq_mse={mse:.6f}, params={n_params:,}")

    return model, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
P("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
P(f"  Teacher on device={device}")

from aleph.lm_geometric_compression import _set_seed, _extract_ffn_handles

# ---------------------------------------------------------------------------
# Spectral selection is WEIGHT-ONLY — compute once before the seed loop
# ---------------------------------------------------------------------------
P("\n--- Computing spectral channel selection (weight-only, once) ---")
spectral_selected = {}
spectral_stats = {}
handles = {h.layer_id: h for h in _extract_ffn_handles(teacher.model, teacher.activation_kind)}

for bid in BLOCKS:
    handle = handles[bid]
    W_down = _standard_weight(handle.c_proj).float()  # (768, 3072)
    # Dead mask: channels where W_down column norm is negligible
    col_norms = W_down.norm(dim=0)
    dead_mask = col_norms < 1e-6
    P(f"  Block {bid}: W_down shape={tuple(W_down.shape)}, "
      f"dead={dead_mask.sum().item()}")
    selected, sel_stats = spectral_greedy_select(
        W_down, TOPK_HIDDEN, dead_mask, verbose=True
    )
    spectral_selected[bid] = selected
    spectral_stats[bid] = sel_stats

P("\nSpectral selection complete for all blocks.")

# ---------------------------------------------------------------------------
# Per-seed loop (only atlas probing, topk selection, lstsq, polish vary)
# ---------------------------------------------------------------------------
all_seed_results = {}

for seed in SEEDS:
    P(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    _set_seed(seed)

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl["perplexity"])
    P(f"  Baseline PPL: {baseline_ppl:.3f}")

    if not (15.0 <= baseline_ppl <= 65.0):
        P(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [15, 65]")
        sys.exit(1)

    # Probe FFN geometry (needed for hidden activations + topk scores)
    P(f"  Probing FFN geometry for blocks {BLOCKS}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    # --- Topk channel selection (data-dependent) ---
    P("\n  --- Topk channel selection ---")
    budget = CompressionBudgetSpec(
        hidden_fraction=TOPK_HIDDEN / GPT2_FFN_HIDDEN,
        min_hidden=32,
        selection_mode="topk",
        per_layer_hidden={bid: TOPK_HIDDEN for bid in BLOCKS},
    )
    topk_selected = {}
    for layer in atlas.layer_geometries:
        if layer.layer_id not in BLOCKS:
            continue
        bid = layer.layer_id
        selected = _select_hidden_units(layer, budget, TOPK_HIDDEN)
        topk_selected[bid] = selected

    # --- Channel overlap ---
    overlap_stats = {}
    for bid in BLOCKS:
        s_spec = set(spectral_selected[bid].tolist())
        s_topk = set(topk_selected[bid].tolist())
        overlap = len(s_spec & s_topk)
        overlap_stats[bid] = {
            "overlap": overlap,
            "overlap_frac": round(overlap / TOPK_HIDDEN, 4),
        }
        P(f"  Block {bid}: spectral-topk overlap = {overlap}/{TOPK_HIDDEN} "
          f"({100*overlap/TOPK_HIDDEN:.1f}%)")

    # --- Condition A: Topk + lstsq + polish ---
    P("\n  --- Condition A: Topk + lstsq ---")
    model_A, stats_A = _build_compressed_model(
        teacher, atlas, BLOCKS, topk_selected, device
    )
    res_A_pre = _evaluate_model(teacher, model_A, corpus, batch_size=BATCH_SIZE)
    ppl_A_pre = float(res_A_pre["perplexity"])
    P(f"  Topk PPL (pre-polish): {ppl_A_pre:.3f}")

    P(f"  Polishing topk ({POLISH_STEPS} steps)...")
    polish_losses_A = _joint_polish(
        model_A, BLOCKS, corpus.eval_val, POLISH_STEPS, POLISH_LR, device, seed
    )
    res_A = _evaluate_model(teacher, model_A, corpus, batch_size=BATCH_SIZE)
    ppl_A = float(res_A["perplexity"])
    P(f"  Topk PPL (post-polish): {ppl_A:.3f}")
    del model_A

    # --- Condition B: Spectral + lstsq + polish ---
    P("\n  --- Condition B: Spectral + lstsq ---")
    model_B, stats_B = _build_compressed_model(
        teacher, atlas, BLOCKS, spectral_selected, device
    )
    res_B_pre = _evaluate_model(teacher, model_B, corpus, batch_size=BATCH_SIZE)
    ppl_B_pre = float(res_B_pre["perplexity"])
    P(f"  Spectral PPL (pre-polish): {ppl_B_pre:.3f}")

    P(f"  Polishing spectral ({POLISH_STEPS} steps)...")
    polish_losses_B = _joint_polish(
        model_B, BLOCKS, corpus.eval_val, POLISH_STEPS, POLISH_LR, device, seed
    )
    res_B = _evaluate_model(teacher, model_B, corpus, batch_size=BATCH_SIZE)
    ppl_B = float(res_B["perplexity"])
    P(f"  Spectral PPL (post-polish): {ppl_B:.3f}")
    del model_B

    all_seed_results[seed] = {
        "baseline_ppl": baseline_ppl,
        "ppl_topk_pre": ppl_A_pre,
        "ppl_topk_post": ppl_A,
        "ppl_spectral_pre": ppl_B_pre,
        "ppl_spectral_post": ppl_B,
        "spectral_stats": {str(k): v for k, v in spectral_stats.items()},
        "topk_stats": {str(k): v for k, v in stats_A.items()},
        "spectral_block_stats": {str(k): v for k, v in stats_B.items()},
        "overlap": {str(k): v for k, v in overlap_stats.items()},
        "delta_topk_pct": round(100 * (ppl_A - baseline_ppl) / baseline_ppl, 2),
        "delta_spectral_pct": round(100 * (ppl_B - baseline_ppl) / baseline_ppl, 2),
    }

    P(f"\n  Summary seed {seed}:")
    P(f"    Baseline:           {baseline_ppl:.3f}")
    P(f"    Topk (post-polish): {ppl_A:.3f} ({all_seed_results[seed]['delta_topk_pct']:+.1f}%)")
    P(f"    Spectral (post):    {ppl_B:.3f} ({all_seed_results[seed]['delta_spectral_pct']:+.1f}%)")

# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------
P(f"\n{'='*60}\nAGGREGATE ACROSS {len(SEEDS)} SEEDS\n{'='*60}")

mean_base = sum(r["baseline_ppl"] for r in all_seed_results.values()) / len(SEEDS)
mean_topk = sum(r["ppl_topk_post"] for r in all_seed_results.values()) / len(SEEDS)
mean_spec = sum(r["ppl_spectral_post"] for r in all_seed_results.values()) / len(SEEDS)
mean_topk_pre = sum(r["ppl_topk_pre"] for r in all_seed_results.values()) / len(SEEDS)
mean_spec_pre = sum(r["ppl_spectral_pre"] for r in all_seed_results.values()) / len(SEEDS)

P(f"  Mean baseline:           {mean_base:.3f}")
P(f"  Mean topk (pre-polish):  {mean_topk_pre:.3f} (delta: {100*(mean_topk_pre-mean_base)/mean_base:+.1f}%)")
P(f"  Mean topk (post-polish): {mean_topk:.3f} (delta: {100*(mean_topk-mean_base)/mean_base:+.1f}%)")
P(f"  Mean spectral (pre):     {mean_spec_pre:.3f} (delta: {100*(mean_spec_pre-mean_base)/mean_base:+.1f}%)")
P(f"  Mean spectral (post):    {mean_spec:.3f} (delta: {100*(mean_spec-mean_base)/mean_base:+.1f}%)")

# Verdict
diff = mean_spec - mean_topk
if diff <= -2:
    verdict = "CONFIRMED"
elif diff >= 2:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

P(f"\n  VERDICT: {verdict}")
P(f"    spectral - topk = {diff:+.2f} PPL (threshold: +/-2)")
P(f"    spectral={mean_spec:.1f}, topk={mean_topk:.1f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out = {
    "cycle": 179,
    "item": 189,
    "seeds": SEEDS,
    "blocks": BLOCKS,
    "topk_hidden": TOPK_HIDDEN,
    "polish_steps": POLISH_STEPS,
    "verdict": verdict,
    "mean_baseline_ppl": round(mean_base, 3),
    "mean_ppl_topk_pre": round(mean_topk_pre, 3),
    "mean_ppl_topk_post": round(mean_topk, 3),
    "mean_ppl_spectral_pre": round(mean_spec_pre, 3),
    "mean_ppl_spectral_post": round(mean_spec, 3),
    "spectral_minus_topk": round(diff, 3),
    "per_seed": {str(k): v for k, v in all_seed_results.items()},
}

out_json = RESULTS_DIR / "cycle_179_spectral_selection.json"
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
P(f"\nResults saved to {out_json}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: PPL by condition across seeds
ax = axes[0]
seeds_list = list(all_seed_results.keys())
x = range(len(seeds_list))
width = 0.2
ppls_base = [all_seed_results[s]["baseline_ppl"] for s in seeds_list]
ppls_topk_pre = [all_seed_results[s]["ppl_topk_pre"] for s in seeds_list]
ppls_topk_post = [all_seed_results[s]["ppl_topk_post"] for s in seeds_list]
ppls_spec_pre = [all_seed_results[s]["ppl_spectral_pre"] for s in seeds_list]
ppls_spec_post = [all_seed_results[s]["ppl_spectral_post"] for s in seeds_list]

ax.bar([i - 1.5*width for i in x], ppls_base, width, label="Float baseline", color="grey")
ax.bar([i - 0.5*width for i in x], ppls_topk_post, width, label="Topk+lstsq+polish", color="salmon")
ax.bar([i + 0.5*width for i in x], ppls_spec_post, width, label="Spectral+lstsq+polish", color="steelblue")

ax.set_xticks(list(x))
ax.set_xticklabels([f"seed {s}" for s in seeds_list])
ax.set_ylabel("PPL")
ax.set_title(f"PPL by condition (verdict: {verdict})")
ax.legend(fontsize=8)

# Panel 2: Pre vs post-polish comparison
ax = axes[1]
ax.bar([i - 0.3 for i in x], ppls_topk_pre, 0.25, label="Topk pre-polish", color="lightsalmon")
ax.bar([i - 0.05 for i in x], ppls_topk_post, 0.25, label="Topk post-polish", color="salmon")
ax.bar([i + 0.2 for i in x], ppls_spec_pre, 0.25, label="Spectral pre-polish", color="lightsteelblue")
ax.bar([i + 0.45 for i in x], ppls_spec_post, 0.25, label="Spectral post-polish", color="steelblue")

ax.set_xticks(list(x))
ax.set_xticklabels([f"seed {s}" for s in seeds_list])
ax.set_ylabel("PPL")
ax.set_title("Pre vs post-polish")
ax.legend(fontsize=7)

# Panel 3: Channel overlap per block
ax = axes[2]
first_seed = SEEDS[0]
overlaps = all_seed_results[first_seed]["overlap"]
block_labels = [f"Block {b}" for b in BLOCKS]
overlap_vals = [overlaps[str(b)]["overlap"] for b in BLOCKS]
overlap_fracs = [overlaps[str(b)]["overlap_frac"] for b in BLOCKS]

bars = ax.bar(range(len(BLOCKS)), overlap_vals, color="mediumpurple")
ax.axhline(y=TOPK_HIDDEN, color="red", linestyle="--", alpha=0.5, label=f"Total k={TOPK_HIDDEN}")
ax.set_xticks(range(len(BLOCKS)))
ax.set_xticklabels(block_labels)
ax.set_ylabel("Overlapping channels")
ax.set_title("Spectral-Topk channel overlap")
ax.legend(fontsize=8)
for i, (val, frac) in enumerate(zip(overlap_vals, overlap_fracs)):
    ax.text(i, val + 5, f"{100*frac:.0f}%", ha="center", fontsize=9)

plt.tight_layout()
out_png = RESULTS_DIR / "cycle_179_spectral_selection.png"
fig.savefig(out_png, dpi=150)
P(f"Plot saved to {out_png}")
plt.close(fig)

P("\n=== DONE ===")
