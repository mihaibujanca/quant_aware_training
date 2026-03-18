"""Cycle 030: Learned compressor generalization on held-out GPT-2 FFN blocks.

Train a shared low-rank bottleneck (U: 768→384, V: 384→768) on 8 of 12 GPT-2 FFN blocks.
Evaluate zero-shot on 4 held-out blocks (bias-only adaptation via mean-shift).

Baselines:
  - Geometric: top-384 channel selection + lstsq refit (per-block, no sharing)
  - Per-block distillation: separate 768→384→768 MLP per held-out block (in-distrib upper bound)

Confirmed if: shared compressor beats geometric on >= 3 held-out blocks
              AND held-out PPL <= 1.25 * train PPL (generalization ratio).
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _apply_activation,
    _evaluate_model,
    _extract_ffn_handles,
    _fit_output_projection,
    _standard_bias,
    _standard_weight,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"

TRAIN_BLOCKS = (0, 1, 2, 3, 4, 5, 6, 7)
HELD_OUT_BLOCKS = (8, 9, 10, 11)
ALL_BLOCKS = TRAIN_BLOCKS + HELD_OUT_BLOCKS
K = 384          # 8x compression: 3072 -> 384
BOTTLENECK = 384  # shared encoder/decoder dimension

SHARED_STEPS = 600
DISTILL_STEPS = 500
LR = 1e-3
BATCH_TOKENS = 128
SEED = 42


# ---------------------------------------------------------------------------
# Shared bottleneck compressor
# ---------------------------------------------------------------------------

class SharedBottleneck(nn.Module):
    """Shared encoder-decoder with per-block output bias.

    forward(h, block_idx) -> V(gelu(U(h))) + b[block_idx]
    """

    def __init__(self, input_dim: int, bottleneck: int, n_train_blocks: int):
        super().__init__()
        self.U = nn.Linear(input_dim, bottleneck, bias=False)
        self.V = nn.Linear(bottleneck, input_dim, bias=False)
        # Per-block output bias (one per train block during training)
        self.b = nn.Parameter(torch.zeros(n_train_blocks, input_dim))

    def forward(self, h: torch.Tensor, block_idx: int) -> torch.Tensor:
        return self.V(F.gelu(self.U(h))) + self.b[block_idx]

    def encode_decode(self, h: torch.Tensor) -> torch.Tensor:
        """U+V pass without any block bias (for held-out bias adaptation)."""
        return self.V(F.gelu(self.U(h)))


# ---------------------------------------------------------------------------
# Per-block distillation MLP (baseline)
# ---------------------------------------------------------------------------

class DistillMLP(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, input_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(h)))


# ---------------------------------------------------------------------------
# Hook-based activation collection
# ---------------------------------------------------------------------------

def collect_block_activations(
    model: nn.Module,
    sequences: torch.Tensor,
    block_ids: tuple[int, ...],
    activation_kind: str,
    device: torch.device,
    batch_size: int = 4,
    max_tokens: int = 4096,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Return {block_id: (h_in, h_out)} for each requested block.

    h_in is the input to the FFN MLP; h_out is the FFN MLP output.
    Both are float32 tensors of shape [N, 768].
    """
    handles_all = _extract_ffn_handles(model, activation_kind)
    handle_map = {h.layer_id: h for h in handles_all}
    selected = [handle_map[b] for b in block_ids if b in handle_map]

    inputs_dict: dict[int, list[torch.Tensor]] = {h.layer_id: [] for h in selected}
    outputs_dict: dict[int, list[torch.Tensor]] = {h.layer_id: [] for h in selected}

    hooks = []
    # Capture inputs via forward pre-hook, outputs via forward hook
    for handle in selected:
        def make_pre_hook(lid):
            def _pre(mod, args):
                inp = args[0].detach().cpu().float()
                inputs_dict[lid].append(inp.reshape(-1, inp.shape[-1]))
            return _pre

        def make_hook(lid):
            def _h(mod, inp, out):
                o = out.detach().cpu().float()
                outputs_dict[lid].append(o.reshape(-1, o.shape[-1]))
            return _h

        hooks.append(handle.mlp.register_forward_pre_hook(make_pre_hook(handle.layer_id)))
        hooks.append(handle.mlp.register_forward_hook(make_hook(handle.layer_id)))

    model.eval()
    token_count = 0
    with torch.no_grad():
        for start in range(0, sequences.shape[0], batch_size):
            if token_count >= max_tokens:
                break
            batch = sequences[start:start + batch_size, :-1].to(device)
            model(batch)
            token_count += int(batch.numel())

    for hook in hooks:
        hook.remove()

    result = {}
    for lid in [h.layer_id for h in selected]:
        h_in = torch.cat(inputs_dict[lid], dim=0)[:max_tokens]
        h_out = torch.cat(outputs_dict[lid], dim=0)[:max_tokens]
        result[lid] = (h_in, h_out)
    return result


# ---------------------------------------------------------------------------
# Geometric baseline: channel select + lstsq (per block)
# ---------------------------------------------------------------------------

def geometric_compress_block(
    model: nn.Module,
    block_id: int,
    h_in: torch.Tensor,
    h_out: torch.Tensor,
    activation_kind: str,
    k: int,
) -> tuple[nn.Module, float]:
    """Return (replacement_mlp_module, block_mse)."""
    handles = {h.layer_id: h for h in _extract_ffn_handles(model, activation_kind)}
    handle = handles[block_id]

    W_in = _standard_weight(handle.c_fc)   # [3072, 768]
    b_in = _standard_bias(handle.c_fc)     # [3072] or None

    # Preactivations
    pre = h_in @ W_in.t()
    if b_in is not None:
        pre = pre + b_in
    hidden = _apply_activation(pre, activation_kind)   # [N, 3072]

    # Score channels by std of activation
    scores = hidden.std(dim=0)  # [3072]
    selected = scores.topk(k).indices.sort().values

    W_in_k = W_in[selected]   # [k, 768]
    b_in_k = None if b_in is None else b_in[selected]
    hidden_k = hidden[:, selected]  # [N, k]

    # lstsq to fit W_out
    use_bias = _standard_bias(handle.c_proj) is not None
    W_out, b_out = _fit_output_projection(hidden_k, h_out, use_bias=use_bias)

    # Build replacement MLP (simple Linear+act+Linear)
    class GeomMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, k, bias=b_in_k is not None)
            self.fc2 = nn.Linear(k, 768, bias=b_out is not None)
            with torch.no_grad():
                self.fc1.weight.copy_(W_in_k)
                if b_in_k is not None:
                    self.fc1.bias.copy_(b_in_k)
                self.fc2.weight.copy_(W_out)
                if b_out is not None:
                    self.fc2.bias.copy_(b_out)

        def forward(self, x):
            return self.fc2(_apply_activation(self.fc1(x), activation_kind))

    mlp = GeomMLP()
    with torch.no_grad():
        pred = mlp(h_in)
        mse = F.mse_loss(pred, h_out).item()
    return mlp, mse


# ---------------------------------------------------------------------------
# Inject a replacement MLP into one block
# ---------------------------------------------------------------------------

def inject_block_mlp(model: nn.Module, block_id: int, new_mlp: nn.Module, activation_kind: str, device: torch.device) -> nn.Module:
    patched = copy.deepcopy(model)
    handles = {h.layer_id: h for h in _extract_ffn_handles(patched, activation_kind)}
    handles[block_id].block.mlp = new_mlp.to(device)
    return patched


# ---------------------------------------------------------------------------
# Evaluate one modified block's PPL
# ---------------------------------------------------------------------------

def eval_single_block_ppl(
    teacher,
    model_with_replacement: nn.Module,
    corpus,
    batch_size: int = 4,
) -> float:
    metrics = _evaluate_model(teacher, model_with_replacement, corpus, batch_size=batch_size)
    return metrics["perplexity"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    print("Loading GPT-2 teacher...")
    teacher_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device="cpu")
    teacher = load_nanochat_teacher(teacher_spec)
    model = teacher.model
    activation_kind = teacher.activation_kind  # "gelu"

    print("Building probe corpus...")
    corpus_spec = ProbeCorpusSpec(
        sequence_length=128,
        probe_train_sequences=64,
        probe_holdout_sequences=32,
        eval_val_sequences=32,
        text_path=SHAKESPEARE_PATH,
        teacher=teacher,
        dataset_name="shakespeare",
    )
    corpus = build_probe_corpus(corpus_spec)
    print(f"  Probe train: {corpus.token_counts['probe_train']} tokens")
    print(f"  Eval val: {corpus.token_counts['eval_val']} tokens")

    # Baseline PPL
    print("\nEvaluating baseline (uncompressed) PPL...")
    baseline_metrics = _evaluate_model(teacher, model, corpus)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # Collect activations for all 12 blocks
    print("\nCollecting FFN activations for all 12 blocks...")
    all_acts = collect_block_activations(
        model, corpus.probe_train, ALL_BLOCKS, activation_kind, device,
        batch_size=4, max_tokens=4096,
    )
    print(f"  Collected blocks: {list(all_acts.keys())}, tokens/block: {next(iter(all_acts.values()))[0].shape[0]}")

    dim = 768
    n_train = len(TRAIN_BLOCKS)

    # -----------------------------------------------------------------------
    # Train shared bottleneck on train blocks
    # -----------------------------------------------------------------------
    print(f"\nTraining shared bottleneck on blocks {TRAIN_BLOCKS}...")
    compressor = SharedBottleneck(dim, BOTTLENECK, n_train)
    optimizer = torch.optim.Adam(compressor.parameters(), lr=LR)

    # Build training dataset: (h_in, h_out, block_local_idx)
    train_data = []
    for local_idx, bid in enumerate(TRAIN_BLOCKS):
        h_in, h_out = all_acts[bid]
        n = h_in.shape[0]
        for i in range(n):
            train_data.append((h_in[i], h_out[i], local_idx))

    print(f"  Training pairs: {len(train_data)}")

    compressor.train()
    for step in range(SHARED_STEPS):
        # Sample a mini-batch
        indices = torch.randint(0, len(train_data), (BATCH_TOKENS,))
        h_in_batch = torch.stack([train_data[i][0] for i in indices])
        h_out_batch = torch.stack([train_data[i][1] for i in indices])
        block_idxs = [train_data[i][2] for i in indices]

        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for local_idx in range(n_train):
            mask = torch.tensor([b == local_idx for b in block_idxs])
            if mask.sum() == 0:
                continue
            pred = compressor(h_in_batch[mask], local_idx)
            loss = loss + F.mse_loss(pred, h_out_batch[mask])

        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{SHARED_STEPS}, loss={loss.item():.6f}")

    compressor.eval()

    # -----------------------------------------------------------------------
    # Compute in-distribution MSE + PPL on train blocks
    # -----------------------------------------------------------------------
    print("\nEvaluating in-distribution (train blocks)...")
    train_mses = {}
    train_ppls = {}

    with torch.no_grad():
        for local_idx, bid in enumerate(TRAIN_BLOCKS):
            h_in, h_out = all_acts[bid]
            pred = compressor(h_in, local_idx)
            mse = F.mse_loss(pred, h_out).item()
            train_mses[bid] = mse

    # PPL for all train blocks replaced simultaneously
    class SharedBottleneckModule(nn.Module):
        """Wraps SharedBottleneck for a specific block_idx as a drop-in MLP."""
        def __init__(self, net: SharedBottleneck, block_idx: int, bias: torch.Tensor | None = None):
            super().__init__()
            self.net = net
            self.block_idx = block_idx
            self.extra_bias = bias

        def forward(self, x):
            out = self.net.encode_decode(x)
            if self.extra_bias is not None:
                out = out + self.extra_bias
            return out

    # Get train block PPL (replace each individually)
    for local_idx, bid in enumerate(TRAIN_BLOCKS[:4]):  # Sample 4 for speed
        m = SharedBottleneckModule(compressor, local_idx)
        model_patched = inject_block_mlp(model, bid, m, activation_kind, device)
        ppl = eval_single_block_ppl(teacher, model_patched, corpus)
        train_ppls[bid] = ppl
        print(f"  Train block {bid}: MSE={train_mses[bid]:.6f}, PPL={ppl:.2f}")

    mean_train_ppl = sum(train_ppls.values()) / len(train_ppls)
    print(f"  Mean train PPL (sample): {mean_train_ppl:.2f}")

    # -----------------------------------------------------------------------
    # Held-out evaluation
    # -----------------------------------------------------------------------
    print(f"\nEvaluating held-out blocks {HELD_OUT_BLOCKS}...")
    geometric_ppls = {}
    distill_ppls = {}
    shared_ppls = {}
    shared_mses = {}
    geometric_mses = {}

    for bid in HELD_OUT_BLOCKS:
        h_in, h_out = all_acts[bid]
        print(f"\n  Block {bid}:")

        # 1. Shared compressor with bias-only adaptation (mean-shift)
        with torch.no_grad():
            pred_no_bias = compressor.encode_decode(h_in)
            bias_adapt = (h_out - pred_no_bias).mean(dim=0)  # [768]
            pred_adapted = pred_no_bias + bias_adapt
            shared_mse = F.mse_loss(pred_adapted, h_out).item()
        shared_mses[bid] = shared_mse

        m_shared = SharedBottleneckModule(compressor, -1, bias=bias_adapt)
        model_patched = inject_block_mlp(model, bid, m_shared, activation_kind, device)
        shared_ppl = eval_single_block_ppl(teacher, model_patched, corpus)
        shared_ppls[bid] = shared_ppl
        print(f"    Shared (bias-adapt): MSE={shared_mse:.6f}, PPL={shared_ppl:.2f}")

        # 2. Geometric baseline (channel selection + lstsq)
        geom_mlp, geom_mse = geometric_compress_block(model, bid, h_in, h_out, activation_kind, K)
        geometric_mses[bid] = geom_mse
        model_geom = inject_block_mlp(model, bid, geom_mlp, activation_kind, device)
        geom_ppl = eval_single_block_ppl(teacher, model_geom, corpus)
        geometric_ppls[bid] = geom_ppl
        print(f"    Geometric:           MSE={geom_mse:.6f}, PPL={geom_ppl:.2f}")

        # 3. Per-block distillation (in-distribution upper bound)
        distill_net = DistillMLP(dim, BOTTLENECK)
        opt_d = torch.optim.Adam(distill_net.parameters(), lr=LR)
        distill_net.train()
        n = h_in.shape[0]
        for step in range(DISTILL_STEPS):
            idx = torch.randint(0, n, (BATCH_TOKENS,))
            pred = distill_net(h_in[idx])
            loss = F.mse_loss(pred, h_out[idx])
            opt_d.zero_grad()
            loss.backward()
            opt_d.step()
        distill_net.eval()

        with torch.no_grad():
            distill_mse = F.mse_loss(distill_net(h_in), h_out).item()
        model_distill = inject_block_mlp(model, bid, distill_net, activation_kind, device)
        distill_ppl = eval_single_block_ppl(teacher, model_distill, corpus)
        distill_ppls[bid] = distill_ppl
        print(f"    Per-block distill:   MSE={distill_mse:.6f}, PPL={distill_ppl:.2f}")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    beats_geometric = sum(shared_ppls[b] < geometric_ppls[b] for b in HELD_OUT_BLOCKS)
    generalization_ratio = (
        sum(shared_ppls.values()) / len(shared_ppls)
    ) / mean_train_ppl if mean_train_ppl > 0 else float("inf")

    print(f"\n=== VERDICT ===")
    print(f"Beats geometric: {beats_geometric}/{len(HELD_OUT_BLOCKS)} blocks")
    print(f"Generalization ratio (held-out PPL / train PPL): {generalization_ratio:.3f}")
    print(f"Confirmed if: beats_geometric >= 3 AND gen_ratio <= 1.25")

    if beats_geometric >= 3 and generalization_ratio <= 1.25:
        verdict = "CONFIRMED"
    elif (sum(shared_ppls[b] > geometric_ppls[b] for b in HELD_OUT_BLOCKS) >= 3
          and generalization_ratio > 2.0):
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    print(f"Verdict: {verdict}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "baseline_ppl": baseline_ppl,
        "train_blocks": list(TRAIN_BLOCKS),
        "held_out_blocks": list(HELD_OUT_BLOCKS),
        "k_bottleneck": BOTTLENECK,
        "k_channels": K,
        "shared_steps": SHARED_STEPS,
        "distill_steps": DISTILL_STEPS,
        "train_block_mses": {str(b): train_mses[b] for b in TRAIN_BLOCKS},
        "train_block_ppls": {str(b): train_ppls[b] for b in train_ppls},
        "mean_train_ppl": mean_train_ppl,
        "held_out": {
            str(b): {
                "shared_mse": shared_mses[b],
                "shared_ppl": shared_ppls[b],
                "geometric_mse": geometric_mses[b],
                "geometric_ppl": geometric_ppls[b],
                "distill_ppl": distill_ppls[b],
                "shared_beats_geometric": bool(shared_ppls[b] < geometric_ppls[b]),
            }
            for b in HELD_OUT_BLOCKS
        },
        "beats_geometric_count": int(beats_geometric),
        "generalization_ratio": float(generalization_ratio),
        "verdict": verdict,
    }

    out_path = RESULTS_DIR / "cycle_030_generalization.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
