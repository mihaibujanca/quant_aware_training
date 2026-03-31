"""Cycle 156: Joint compress + correct training (item 166).

Train rank-64 factorization AND LoRA corrections simultaneously under
stacked CE loss. Compare to topk+corrections (PPL ~59) and sequential
joint+corrections (cycle 153, PPL 107.5).

Seed 42 screening. Escalate if promising.

Verdict:
  CONFIRMED    if PPL < 65
  REFUTED      if PPL > 75 or corrections collapse
  INCONCLUSIVE if 65-75
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    load_nanochat_teacher,
)

SEED = 42
BLOCKS = [7, 5, 4, 1]
D_MODEL = 768
D_FFN = 3072
D_BN = 64
RANK = 8
SEQ_LEN = 128
BATCH_SIZE = 4
STEPS = 2000
LR = 1e-4

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class LinearFactorFFN(nn.Module):
    def __init__(self, original_mlp, d_bn):
        super().__init__()
        self.c_fc = original_mlp.c_fc
        self.act = original_mlp.act
        self.encoder = nn.Linear(D_FFN, d_bn, bias=False)
        self.decoder = nn.Linear(d_bn, D_MODEL, bias=True)
        if hasattr(original_mlp.c_proj, 'bias') and original_mlp.c_proj.bias is not None:
            self.decoder.bias.data.copy_(original_mlp.c_proj.bias.data)
        for p in self.c_fc.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.decoder(self.encoder(self.act(self.c_fc(x))))


class LowRankCorrection(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d, r))
        self.V = nn.Parameter(torch.randn(r, d) * 0.01)

    def forward(self, x):
        return x + (x @ self.V.t()) @ self.U.t()


if __name__ == "__main__":
    _set_seed(SEED)
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    device = teacher.device

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=200, probe_holdout_sequences=200,
        eval_val_sequences=200, dataset_name="openwebtext", shuffle_seed=SEED,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    # Build model with factorized FFNs + LoRA corrections
    _set_seed(SEED)
    model = copy.deepcopy(teacher.model); model.eval()
    for bid in BLOCKS:
        model.transformer.h[bid].mlp = LinearFactorFFN(model.transformer.h[bid].mlp, D_BN)

    # Add corrections as hooks
    corrections = nn.ModuleDict({str(b): LowRankCorrection(D_MODEL, RANK) for b in BLOCKS})
    hooks = []
    for bid in BLOCKS:
        corr = corrections[str(bid)]
        def make_hook(c):
            def hook(mod, inp, out):
                return (c(out[0]),) + out[1:] if isinstance(out, tuple) else c(out)
            return hook
        hooks.append(model.transformer.h[bid].register_forward_hook(make_hook(corr)))

    # Collect ALL trainable params (factorization + corrections)
    model.requires_grad_(False)
    all_params = []
    for bid in BLOCKS:
        for p in model.transformer.h[bid].mlp.encoder.parameters():
            p.requires_grad_(True); all_params.append(p)
        for p in model.transformer.h[bid].mlp.decoder.parameters():
            p.requires_grad_(True); all_params.append(p)
    for p in corrections.parameters():
        p.requires_grad_(True); all_params.append(p)

    factorization_params = sum(p.numel() for bid in BLOCKS
                               for p in list(model.transformer.h[bid].mlp.encoder.parameters()) +
                               list(model.transformer.h[bid].mlp.decoder.parameters()))
    correction_params = sum(p.numel() for p in corrections.parameters())
    print(f"  Factorization params: {factorization_params:,}", flush=True)
    print(f"  Correction params: {correction_params:,}", flush=True)
    print(f"  Total trainable: {factorization_params + correction_params:,}", flush=True)

    # Train jointly
    opt = torch.optim.AdamW(all_params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=0)
    rng = torch.Generator(); rng.manual_seed(SEED)

    print(f"\n  Training {STEPS} steps (joint compress+correct)...", flush=True)
    losses = []
    for step in range(STEPS):
        idx = torch.randperm(corpus.probe_holdout.shape[0], generator=rng)[:BATCH_SIZE]
        batch = corpus.probe_holdout[idx].to(device)
        opt.zero_grad()
        raw = model(batch[:, :-1])
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.item()))
        if (step + 1) % 500 == 0:
            print(f"      step {step+1}/{STEPS}: loss={loss.item():.4f}", flush=True)

    model.eval(); model.requires_grad_(False)

    # Eval
    final_ppl = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)["perplexity"]
    print(f"\n  Joint compress+correct PPL: {final_ppl:.3f}", flush=True)

    # Check correction magnitude
    for bid in sorted(BLOCKS):
        corr = corrections[str(bid)]
        c_fro = (corr.U.data @ corr.V.data).norm().item()
        print(f"    Block {bid} correction ||UV||_F: {c_fro:.4f}", flush=True)

    for h in hooks:
        h.remove()

    # Verdict
    print(f"\n  topk+corrections baseline: ~59", flush=True)
    print(f"  Joint compress-only (cycle 155): ~91", flush=True)
    print(f"  Sequential joint+corr (cycle 153): 107.5", flush=True)

    if final_ppl < 65:
        verdict = "CONFIRMED"
        detail = f"PPL {final_ppl:.1f} < 65"
    elif final_ppl > 75:
        verdict = "REFUTED"
        detail = f"PPL {final_ppl:.1f} > 75"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"PPL {final_ppl:.1f} in 65-75 range"

    print(f"  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 156, "item": 166, "seed": SEED,
        "float_ppl": bl["perplexity"],
        "final_ppl": final_ppl,
        "factorization_params": factorization_params,
        "correction_params": correction_params,
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_156_joint_compress_correct.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
