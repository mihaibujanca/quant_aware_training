"""Cycle 087: Mixed-corpus stacking evaluation — GPT-2 FFN (item 96).

Two conditions, stacking blocks [7,5,4,1] at 6x compression (k=64):
  A: OWT-only 2048 tokens for refit
  B: Mixed-corpus 2048 tokens (512 each from Shakespeare, WikiText-2, OWT, C4)

Channel selection: OWT atlas for BOTH conditions.
Polish: +500 Adam steps, lr=1e-4, W_down only, joint across all 4 blocks.
Seeds: [42, 123, 7].
Eval: OWT (200 seqs) AND WikiText-2 (200 seqs).

Criteria (item 96):
  CONFIRMED if:  mixed reduces stacked PPL by >=10% relative vs OWT-only on WikiText-2
                 AND does not degrade OWT by >3%
  REFUTED if:    mixed degrades stacked OWT PPL by >5%
  INCONCLUSIVE:  helps WikiText-2 but degrades OWT by 3-5%
"""
from __future__ import annotations

import copy
import json
import random
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressionBudgetSpec,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _iter_lm_batches,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_bias,
    _standard_weight,
    _apply_activation,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR  = Path(__file__).parent.parent / "data"
OWT_PATH  = str(DATA_DIR / "openwebtext_sample.txt")
WIKI_PATH = str(DATA_DIR / "wikitext2_valid.txt")
SHAKES_PATH = str(DATA_DIR / "shakespeare.txt")
C4_PATH   = str(DATA_DIR / "c4_sample.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BLOCKS  = [7, 5, 4, 1]
COMPRESSION    = 6
SEEDS          = [42, 123, 7]
BATCH_SIZE     = 4
SEQ_LEN        = 128
EVAL_SEQUENCES = 200

PROBE_TRAIN_SEQS   = 200
PROBE_HOLDOUT_SEQS = 200   # OWT holdout for polish

N_TOKENS_REFIT      = 2048
N_TOKENS_PER_CORPUS = 512   # 512 × 4 = 2048 total for condition B

POLISH_STEPS = 500
POLISH_LR    = 1e-4

budget = CompressionBudgetSpec(
    hidden_fraction=1.0 / COMPRESSION,
    min_hidden=32,
    selection_mode="topk",
)

MIXED_CORPUS_PATHS = {
    "shakespeare": SHAKES_PATH,
    "wikitext2":   WIKI_PATH,
    "openwebtext": OWT_PATH,
    "c4":          C4_PATH,
}


# ---------------------------------------------------------------------------
# Helpers (adapted from cycle 078 + cycle 061)
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def collect_hidden_activations(
    teacher: TeacherBundle,
    token_tensor: torch.Tensor,
    block_id: int,
    n_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect hidden-layer activations (post-activation) and MLP outputs for one block."""
    from aleph.lm_geometric_compression import _extract_ffn_handles, _sample_tensor

    all_handles = _extract_ffn_handles(teacher.model, teacher.activation_kind)
    handle = next(h for h in all_handles if h.layer_id == block_id)

    hidden_samples: list[torch.Tensor] = []
    output_samples: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        x = inputs[0].detach()
        z = module.c_fc(x).detach()
        h = _apply_activation(z, handle.activation_kind).detach()
        remaining = n_tokens - sum(item.shape[0] for item in hidden_samples)
        taken = _sample_tensor(hidden_samples, h, remaining)
        if taken > 0:
            _sample_tensor(output_samples, output, taken)

    hook = handle.mlp.register_forward_hook(hook_fn)
    try:
        teacher.model.eval()
        with torch.no_grad():
            for input_ids, _targets in _iter_lm_batches(token_tensor, BATCH_SIZE, teacher.device):
                teacher.model(input_ids)
                if sum(item.shape[0] for item in hidden_samples) >= n_tokens:
                    break
    finally:
        hook.remove()

    h_cat = torch.cat(hidden_samples, dim=0).float()[:n_tokens]
    y_cat = torch.cat(output_samples, dim=0).float()[:n_tokens]
    return h_cat, y_cat


def build_pack_from_refits(
    teacher: TeacherBundle,
    atlas,   # FfnGeometryAtlas (channel selection source = OWT)
    refit_data: dict[int, tuple[torch.Tensor, torch.Tensor]],
    dataset_label: str,
) -> CompressedFfnPack:
    """Build CompressedFfnPack using OWT channel selection + provided refit activations."""
    compressed_layers = []
    for layer in atlas.layer_geometries:
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)

        source_w_in  = _standard_weight(layer.handle.c_fc).float()
        source_b_in  = _standard_bias(layer.handle.c_fc)
        use_out_bias = _standard_bias(layer.handle.c_proj) is not None

        student_w_in = source_w_in[selected].contiguous()
        student_b_in = (None if source_b_in is None
                        else source_b_in[selected].contiguous())

        hidden_full, output = refit_data[layer.layer_id]
        hidden_sel = hidden_full[:, selected].float()
        student_w_out, student_b_out = _fit_output_projection(
            hidden_sel, output, use_bias=use_out_bias
        )

        source_param_count = (
            source_w_in.numel()
            + (source_b_in.numel() if source_b_in is not None else 0)
            + _standard_weight(layer.handle.c_proj).numel()
            + (use_out_bias and _standard_bias(layer.handle.c_proj).numel() or 0)
        )
        compressed_param_count = (
            student_w_in.numel()
            + (student_b_in.numel() if student_b_in is not None else 0)
            + student_w_out.numel()
            + (student_b_out.numel() if student_b_out is not None else 0)
        )

        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id,
            activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim,
            compressed_hidden_dim=int(k),
            c_fc_weight=student_w_in,
            c_fc_bias=student_b_in,
            c_proj_weight=student_w_out,
            c_proj_bias=student_b_out,
            selected_indices=selected,
            source_param_count=source_param_count,
            compressed_param_count=compressed_param_count,
            notes={"selection_corpus": "openwebtext", "refit_corpus": dataset_label},
        ))

    return CompressedFfnPack(
        method="geometric_mixed_refit",
        model_family=atlas.model_family,
        dataset_name=dataset_label,
        layers=tuple(compressed_layers),
        metadata={"sequence_length": SEQ_LEN},
    )


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _collect_polish_params(student_model: nn.Module, block_ids: list[int]) -> list[torch.Tensor]:
    params = []
    for block_id in block_ids:
        mlp = _find_compressed_mlp(student_model, block_id)
        if mlp is None or not hasattr(mlp, "c_proj"):
            print(f"  WARNING: no c_proj found for block {block_id}, skipping")
            continue
        w = mlp.c_proj.weight
        w.requires_grad_(True)
        params.append(w)
        b = getattr(mlp.c_proj, "bias", None)
        if b is not None:
            b.requires_grad_(True)
            params.append(b)
    return params


def _polish_blocks(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Run n_steps of Adam on W_down jointly across all block_ids."""
    if n_steps == 0:
        return []

    student_model.requires_grad_(False)
    params = _collect_polish_params(student_model, block_ids)
    if not params:
        return []

    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()
    losses = []
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs  = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = (raw[0] if isinstance(raw, tuple)
                  else (raw.logits if hasattr(raw, "logits") else raw))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

# Build OWT eval corpus (seed=999, held-out)
print("Building OWT eval corpus (seed=999)...")
owt_eval_spec = ProbeCorpusSpec(
    teacher=teacher, text_path=OWT_PATH,
    sequence_length=SEQ_LEN,
    probe_train_sequences=32, probe_holdout_sequences=32,
    eval_val_sequences=EVAL_SEQUENCES,
    dataset_name="openwebtext", shuffle_seed=999,
)
owt_eval_corpus = build_probe_corpus(owt_eval_spec)

# Build WikiText-2 eval corpus (seed=999, held-out)
print("Building WikiText-2 eval corpus (seed=999)...")
wiki_eval_spec = ProbeCorpusSpec(
    teacher=teacher, text_path=WIKI_PATH,
    sequence_length=SEQ_LEN,
    probe_train_sequences=32, probe_holdout_sequences=32,
    eval_val_sequences=EVAL_SEQUENCES,
    dataset_name="wikitext2", shuffle_seed=999,
)
wiki_eval_corpus = build_probe_corpus(wiki_eval_spec)

# Baseline PPL on both eval sets
print("Computing baseline PPL (uncompressed teacher)...")
baseline_owt  = _evaluate_model(teacher, teacher.model, owt_eval_corpus,  batch_size=BATCH_SIZE)["perplexity"]
baseline_wiki = _evaluate_model(teacher, teacher.model, wiki_eval_corpus, batch_size=BATCH_SIZE)["perplexity"]
print(f"Baseline PPL — OWT: {baseline_owt:.2f}, WikiText-2: {baseline_wiki:.2f}")

all_results: list[dict] = []

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    _set_seed(seed)

    # ── OWT probe corpus (channel selection + holdout for polish) ──────────────
    owt_probe_spec = ProbeCorpusSpec(
        teacher=teacher, text_path=OWT_PATH,
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQUENCES,
        dataset_name="openwebtext", shuffle_seed=seed,
    )
    owt_probe_corpus = build_probe_corpus(owt_probe_spec)
    polish_tokens = owt_probe_corpus.probe_holdout
    print(f"  Polish tokens shape: {polish_tokens.shape}")

    # ── Probe FFN geometry on OWT (channel selection for BOTH conditions) ──────
    print(f"  Probing FFN geometry on OWT (2048 tokens)...")
    atlas_owt = probe_ffn_geometry(
        teacher, owt_probe_corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=N_TOKENS_REFIT, batch_size=BATCH_SIZE, jacobian_samples=4,
    )
    for layer in atlas_owt.layer_geometries:
        k = _layer_target_hidden(layer, budget)
        print(f"    Block {layer.layer_id}: k={k}, hidden={layer.hidden_dim}")

    # ── Condition A: OWT-only refit (activations from atlas_owt) ──────────────
    print(f"  Condition A: building OWT-only pack (n=2048)...")
    refit_A: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in atlas_owt.layer_geometries:
        h = layer.hidden_sample.float()
        y = layer.output_sample.float()
        refit_A[layer.layer_id] = (h[:N_TOKENS_REFIT], y[:N_TOKENS_REFIT])
        print(f"    Block {layer.layer_id}: A has {min(h.shape[0], N_TOKENS_REFIT)} tokens")

    # ── Condition B: mixed-corpus refit (512 per corpus) ──────────────────────
    print(f"  Condition B: collecting mixed-corpus activations (512 per corpus)...")
    block_hidden_B: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {
        b: {} for b in TARGET_BLOCKS
    }
    for cname, cpath in MIXED_CORPUS_PATHS.items():
        c_spec = ProbeCorpusSpec(
            teacher=teacher, text_path=cpath,
            sequence_length=SEQ_LEN,
            probe_train_sequences=16, probe_holdout_sequences=4,
            eval_val_sequences=8,
            dataset_name=cname, shuffle_seed=seed,
        )
        c_corpus = build_probe_corpus(c_spec)
        tok_tensor = c_corpus.probe_holdout
        for block_id in TARGET_BLOCKS:
            h512, y512 = collect_hidden_activations(
                teacher, tok_tensor, block_id, n_tokens=N_TOKENS_PER_CORPUS
            )
            block_hidden_B[block_id][cname] = (h512, y512)
            print(f"    Block {block_id}, {cname}: {h512.shape[0]} tokens")

    refit_B: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for block_id in TARGET_BLOCKS:
        hs = [block_hidden_B[block_id][c][0] for c in MIXED_CORPUS_PATHS]
        ys = [block_hidden_B[block_id][c][1] for c in MIXED_CORPUS_PATHS]
        h_cat = torch.cat(hs, dim=0)
        y_cat = torch.cat(ys, dim=0)
        refit_B[block_id] = (h_cat, y_cat)
        print(f"    Block {block_id}: B total tokens = {h_cat.shape[0]}")

    # ── Evaluate both conditions (unpolished + polished) ──────────────────────
    for cond_label, refit_data in [("A_owt_2048", refit_A), ("B_mixed_2048", refit_B)]:
        print(f"\n  --- Condition {cond_label} ---")
        pack = build_pack_from_refits(teacher, atlas_owt, refit_data, cond_label)
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        student.eval()

        # Unpolished eval
        owt_m  = _evaluate_model(teacher, student, owt_eval_corpus,  batch_size=BATCH_SIZE)
        wiki_m = _evaluate_model(teacher, student, wiki_eval_corpus, batch_size=BATCH_SIZE)
        ppl_owt_unpolished  = owt_m["perplexity"]
        ppl_wiki_unpolished = wiki_m["perplexity"]
        print(f"    Unpolished: OWT={ppl_owt_unpolished:.2f}, Wiki={ppl_wiki_unpolished:.2f}")

        all_results.append({
            "seed": seed,
            "condition": cond_label,
            "polish_steps": 0,
            "ppl_owt":  ppl_owt_unpolished,
            "ppl_wiki": ppl_wiki_unpolished,
        })

        # Polish 500 steps (joint W_down)
        print(f"    Polishing {POLISH_STEPS} steps (joint)...")
        losses = _polish_blocks(
            student, TARGET_BLOCKS, polish_tokens,
            n_steps=POLISH_STEPS, lr=POLISH_LR, device=teacher.device,
        )
        first_l = losses[0]  if losses else float("nan")
        last_l  = losses[-1] if losses else float("nan")
        print(f"    Polish loss: {first_l:.4f} → {last_l:.4f}")

        # Polished eval
        owt_m2  = _evaluate_model(teacher, student, owt_eval_corpus,  batch_size=BATCH_SIZE)
        wiki_m2 = _evaluate_model(teacher, student, wiki_eval_corpus, batch_size=BATCH_SIZE)
        ppl_owt_polished  = owt_m2["perplexity"]
        ppl_wiki_polished = wiki_m2["perplexity"]
        print(f"    Polished:   OWT={ppl_owt_polished:.2f}, Wiki={ppl_wiki_polished:.2f}")

        all_results.append({
            "seed": seed,
            "condition": cond_label,
            "polish_steps": POLISH_STEPS,
            "ppl_owt":  ppl_owt_polished,
            "ppl_wiki": ppl_wiki_polished,
        })

        del student, pack

    del atlas_owt, refit_A, refit_B, block_hidden_B


# ---------------------------------------------------------------------------
# Aggregate and verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")

conditions  = ["A_owt_2048", "B_mixed_2048"]
polish_vals = [0, POLISH_STEPS]

agg: dict[str, dict[int, dict]] = {c: {} for c in conditions}
for cond in conditions:
    for steps in polish_vals:
        rows = [r for r in all_results if r["condition"] == cond and r["polish_steps"] == steps]
        ppls_owt  = [r["ppl_owt"]  for r in rows]
        ppls_wiki = [r["ppl_wiki"] for r in rows]
        agg[cond][steps] = {
            "mean_owt":  statistics.mean(ppls_owt),
            "std_owt":   statistics.stdev(ppls_owt) if len(ppls_owt) > 1 else 0.0,
            "mean_wiki": statistics.mean(ppls_wiki),
            "std_wiki":  statistics.stdev(ppls_wiki) if len(ppls_wiki) > 1 else 0.0,
        }

# Print table
print(f"\n{'Condition':<20} {'Steps':>6}  {'OWT PPL':>10}  {'Wiki PPL':>10}")
for cond in conditions:
    for steps in polish_vals:
        a = agg[cond][steps]
        print(f"{cond:<20} {steps:>6}  "
              f"{a['mean_owt']:>8.2f}±{a['std_owt']:.2f}  "
              f"{a['mean_wiki']:>8.2f}±{a['std_wiki']:.2f}")

# Evaluate criteria on POLISHED results (500 steps)
a_owt_ppl  = agg["A_owt_2048"][POLISH_STEPS]["mean_owt"]
b_owt_ppl  = agg["B_mixed_2048"][POLISH_STEPS]["mean_owt"]
a_wiki_ppl = agg["A_owt_2048"][POLISH_STEPS]["mean_wiki"]
b_wiki_ppl = agg["B_mixed_2048"][POLISH_STEPS]["mean_wiki"]

# Relative changes: (B - A) / A * 100  (negative = B is better)
wiki_rel_change_pct = (b_wiki_ppl - a_wiki_ppl) / a_wiki_ppl * 100.0
owt_rel_change_pct  = (b_owt_ppl  - a_owt_ppl)  / a_owt_ppl  * 100.0

print(f"\nBaseline (uncompressed) PPL — OWT: {baseline_owt:.2f}, WikiText-2: {baseline_wiki:.2f}")
print(f"\nPolished ({POLISH_STEPS} steps):")
print(f"  Condition A (OWT-only):  OWT={a_owt_ppl:.2f}, Wiki={a_wiki_ppl:.2f}")
print(f"  Condition B (mixed):     OWT={b_owt_ppl:.2f}, Wiki={b_wiki_ppl:.2f}")
print(f"  Wiki PPL change (A→B): {wiki_rel_change_pct:+.1f}%  (negative = B better)")
print(f"  OWT  PPL change (A→B): {owt_rel_change_pct:+.1f}%  (positive = B worse)")

# Criteria
wiki_improvement_pct = -wiki_rel_change_pct   # positive if B reduces wiki PPL
owt_degradation_pct  =  owt_rel_change_pct    # positive if B degrades OWT PPL

if wiki_improvement_pct >= 10.0 and owt_degradation_pct <= 3.0:
    verdict = "confirmed"
    print("Verdict: CONFIRMED — mixed reduces Wiki PPL by >=10% AND OWT degradation <=3%")
elif owt_degradation_pct > 5.0:
    verdict = "refuted"
    print(f"Verdict: REFUTED — mixed degrades OWT PPL by >5% ({owt_degradation_pct:.1f}%)")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — Wiki improvement={wiki_improvement_pct:.1f}%, OWT degradation={owt_degradation_pct:.1f}%")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
output = {
    "baseline_ppl_owt":  baseline_owt,
    "baseline_ppl_wiki": baseline_wiki,
    "results": all_results,
    "aggregate": {
        cond: {
            str(steps): agg[cond][steps]
            for steps in polish_vals
        }
        for cond in conditions
    },
    "polished_summary": {
        "a_owt_ppl":  a_owt_ppl,
        "a_wiki_ppl": a_wiki_ppl,
        "b_owt_ppl":  b_owt_ppl,
        "b_wiki_ppl": b_wiki_ppl,
        "wiki_improvement_pct": wiki_improvement_pct,
        "owt_degradation_pct":  owt_degradation_pct,
    },
    "verdict": verdict,
    "seeds": SEEDS,
    "target_blocks": TARGET_BLOCKS,
    "compression": COMPRESSION,
    "polish_steps": POLISH_STEPS,
}
json_path = RESULTS_DIR / "cycle_087_mixed_stacking.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {json_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Cycle 087: Mixed-Corpus Stacking — GPT-2 FFN (blocks [7,5,4,1], 6x)", fontsize=12)

cond_labels = ["A: OWT-only\n2048 tok", "B: Mixed\n2048 tok\n(512/corpus)"]
colors = ["steelblue", "forestgreen"]

for row, (eval_name, ppl_key, baseline_ppl) in enumerate([
    ("OWT",        "ppl_owt",  baseline_owt),
    ("WikiText-2", "ppl_wiki", baseline_wiki),
]):
    for col, steps in enumerate([0, POLISH_STEPS]):
        ax = axes[row][col]
        mean_ppls = [agg[c][steps][f"mean_{ppl_key.split('_')[1]}"] for c in conditions]
        std_ppls  = [agg[c][steps][f"std_{ppl_key.split('_')[1]}"]  for c in conditions]
        seed_ppls = [
            [r[ppl_key] for r in all_results if r["condition"] == c and r["polish_steps"] == steps]
            for c in conditions
        ]

        x = list(range(len(conditions)))
        ax.bar(x, mean_ppls, color=colors, alpha=0.7, width=0.5, yerr=std_ppls, capsize=5)
        for i, ppls in enumerate(seed_ppls):
            ax.scatter([i] * len(ppls), ppls, color="black", s=40, zorder=3)
        ax.axhline(baseline_ppl, ls="--", color="red", alpha=0.7, label=f"Baseline={baseline_ppl:.1f}")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=8)
        ax.set_ylabel(f"{eval_name} PPL")
        ax.set_title(f"{eval_name} / {'Unpolished' if steps == 0 else f'{steps}-step polish'}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
png_path = RESULTS_DIR / "cycle_087_mixed_stacking.png"
plt.savefig(png_path, dpi=120, bbox_inches="tight")
print(f"Saved: {png_path}")
print("\nDone.")
