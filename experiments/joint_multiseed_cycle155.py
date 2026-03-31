"""Cycle 155: Multi-seed validation of joint training (item 165).

Run joint random-init rank-64 stacking with 8 seeds to characterize
seed-dependence. Cycle 149 got 69.76 (seed 42), cycle 153 got 91.9.

Verdict:
  CONFIRMED    if median PPL < 72 AND std < 3 (reproducible advantage)
  REFUTED      if median PPL >= 75 (69.76 was an outlier)
  INCONCLUSIVE if median 72-75 or std >= 3 (real but unreliable)
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

SEEDS = [42, 7, 137, 271, 1234, 5678, 9999, 314]
D_MODEL = 768
D_FFN = 3072
D_BN = 64
BLOCKS = [7, 5, 4, 1]
SEQ_LEN = 128
BATCH_SIZE = 4
JOINT_STEPS = 2000
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


if __name__ == "__main__":
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    device = teacher.device

    _set_seed(42)
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=200, probe_holdout_sequences=200,
        eval_val_sequences=200, dataset_name="openwebtext", shuffle_seed=42,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    ppls = {}
    for seed in SEEDS:
        print(f"\n  Seed {seed}:", flush=True)
        _set_seed(seed)

        model = copy.deepcopy(teacher.model); model.eval()
        for bid in BLOCKS:
            model.transformer.h[bid].mlp = LinearFactorFFN(model.transformer.h[bid].mlp, D_BN)

        # Joint training
        model.requires_grad_(False)
        params = []
        for bid in BLOCKS:
            for p in model.transformer.h[bid].mlp.encoder.parameters():
                p.requires_grad_(True); params.append(p)
            for p in model.transformer.h[bid].mlp.decoder.parameters():
                p.requires_grad_(True); params.append(p)

        opt = torch.optim.AdamW(params, lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=JOINT_STEPS, eta_min=0)
        rng = torch.Generator(); rng.manual_seed(seed)

        for step in range(JOINT_STEPS):
            idx = torch.randperm(corpus.probe_holdout.shape[0], generator=rng)[:BATCH_SIZE]
            batch = corpus.probe_holdout[idx].to(device)
            opt.zero_grad()
            raw = model(batch[:, :-1])
            logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
            loss.backward(); opt.step(); sched.step()

        model.eval(); model.requires_grad_(False)
        m = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
        ppl = m["perplexity"]
        ppls[seed] = ppl
        print(f"    PPL: {ppl:.3f}", flush=True)
        del model

    # Summary
    vals = list(ppls.values())
    median_ppl = float(np.median(vals))
    mean_ppl = float(np.mean(vals))
    std_ppl = float(np.std(vals))
    min_ppl = float(np.min(vals))
    max_ppl = float(np.max(vals))

    print(f"\n{'='*50}\nSUMMARY ({len(SEEDS)} seeds)\n{'='*50}", flush=True)
    print(f"  Median: {median_ppl:.3f}", flush=True)
    print(f"  Mean:   {mean_ppl:.3f} ± {std_ppl:.3f}", flush=True)
    print(f"  Min:    {min_ppl:.3f}, Max: {max_ppl:.3f}", flush=True)
    print(f"  topk baseline: 70.699", flush=True)

    if median_ppl < 72.0 and std_ppl < 3.0:
        verdict = "CONFIRMED"
        detail = f"Median {median_ppl:.1f} < 72, std {std_ppl:.1f} < 3"
    elif median_ppl >= 75.0:
        verdict = "REFUTED"
        detail = f"Median {median_ppl:.1f} >= 75 (69.76 was an outlier)"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Median {median_ppl:.1f}, std {std_ppl:.1f}"

    print(f"  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 155, "item": 165, "seeds": SEEDS,
        "float_ppl": bl["perplexity"], "topk_baseline": 70.699,
        "per_seed": ppls,
        "median": median_ppl, "mean": mean_ppl, "std": std_ppl,
        "min": min_ppl, "max": max_ppl,
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_155_joint_multiseed.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
