"""Cycle 152: Progressive stacking — train-then-freeze (item 162).

Compress blocks [1, 4, 7, 11] sequentially. Each block trains under
stacked CE with upstream blocks already frozen+compressed.

Verdict:
  CONFIRMED    if PPL < 69 (beats joint random 69.76 by >=1%)
  REFUTED      if PPL >= 72 (>= independent stacking ~74, no benefit)
  INCONCLUSIVE if 69-72
"""
from __future__ import annotations

import copy
import json
import math
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
BLOCKS_ORDER = [1, 4, 7, 11]  # upstream to downstream
D_MODEL = 768
D_FFN = 3072
D_BN = 64
SEQ_LEN = 128
BATCH_SIZE = 4
STEPS_PER_PHASE = 500
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
        h = self.act(self.c_fc(x))
        return self.decoder(self.encoder(h))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.encoder.parameters()) + \
               sum(p.numel() for p in self.decoder.parameters())


def train_block(model, bid, tokens, n_steps, lr, device, seed):
    """Train one block's encoder+decoder under stacked CE, freeze after."""
    model.requires_grad_(False)
    mlp = model.transformer.h[bid].mlp
    params = []
    for p in mlp.encoder.parameters():
        p.requires_grad_(True); params.append(p)
    for p in mlp.decoder.parameters():
        p.requires_grad_(True); params.append(p)

    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=0)
    rng = torch.Generator(); rng.manual_seed(seed)
    losses = []

    for step in range(n_steps):
        idx = torch.randperm(tokens.shape[0], generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        opt.zero_grad()
        raw = model(batch[:, :-1])
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.item()))
        if (step + 1) % 100 == 0:
            print(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}", flush=True)

    # Freeze permanently
    model.requires_grad_(False)
    return losses


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

    # Start from float model
    model = copy.deepcopy(teacher.model)
    model.eval()

    phase_results = {}
    total_params = 0

    for phase_idx, bid in enumerate(BLOCKS_ORDER):
        phase = phase_idx + 1
        print(f"\n{'='*50}\nPHASE {phase}: Compress block {bid}\n{'='*50}", flush=True)

        # Replace this block's MLP with LinearFactorFFN
        _set_seed(SEED + phase)  # different seed per phase for init diversity
        model.transformer.h[bid].mlp = LinearFactorFFN(model.transformer.h[bid].mlp, D_BN)
        total_params += model.transformer.h[bid].mlp.n_params
        print(f"  Cumulative params: {total_params:,}", flush=True)

        # Pre-train eval
        pre = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
        print(f"  Pre-train PPL: {pre['perplexity']:.3f}", flush=True)

        # Train this block (upstream blocks already frozen)
        losses = train_block(model, bid, corpus.probe_holdout, STEPS_PER_PHASE, LR, device, SEED + phase)

        # Post-train eval
        post = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
        print(f"  Post-train PPL: {post['perplexity']:.3f}", flush=True)

        phase_results[f"phase_{phase}"] = {
            "block": bid, "pre_ppl": pre["perplexity"], "post_ppl": post["perplexity"],
            "loss_start": losses[0], "loss_end": losses[-1],
        }

    # Final eval
    final_ppl = phase_results[f"phase_{len(BLOCKS_ORDER)}"]["post_ppl"]

    # Comparison
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}", flush=True)
    print(f"  Progressive stacking:  {final_ppl:.3f} ({total_params:,} params)", flush=True)
    print(f"  Joint random (c149):   69.764 (986,112 params)", flush=True)
    print(f"  topk+polish (c149):    70.699 (1,575,936 params)", flush=True)
    print(f"  Independent AE (c138): 74.391 (986,112 params)", flush=True)
    print(f"  Float:                 {bl['perplexity']:.3f}", flush=True)

    if final_ppl < 69.0:
        verdict = "CONFIRMED"
        detail = f"Progressive ({final_ppl:.1f}) beats joint random (69.76) by >=1%"
    elif final_ppl >= 72.0:
        verdict = "REFUTED"
        detail = f"Progressive ({final_ppl:.1f}) >= 72 (no benefit over independent)"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Progressive ({final_ppl:.1f}) in 69-72 range"

    print(f"\n  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 152, "item": 162, "seed": SEED,
        "float_ppl": bl["perplexity"], "d_bn": D_BN,
        "blocks_order": BLOCKS_ORDER,
        "steps_per_phase": STEPS_PER_PHASE,
        "phases": phase_results,
        "final_ppl": final_ppl,
        "total_params": total_params,
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_152_progressive_stacking.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
