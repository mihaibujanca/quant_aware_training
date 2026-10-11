"""Cycle 157: Higher-rank joint training (item 167).

Test rank-128 and rank-256 to see if capacity fixes the ~91 PPL problem.
3 seeds per rank. Compare to topk baseline (70.7).

Verdict:
  CONFIRMED    if any rank achieves PPL < 72 AND std < 3 across 3 seeds
  REFUTED      if all ranks >= 75 (capacity doesn't solve the problem)
  INCONCLUSIVE if 72-75 or high variance
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

SEEDS = [42, 137, 7]
RANKS = [64, 128, 256]
BLOCKS = [7, 5, 4, 1]
D_MODEL = 768
D_FFN = 3072
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

    results = {}
    for rank in RANKS:
        for seed in SEEDS:
            key = f"r{rank}_s{seed}"
            print(f"\n  Rank {rank}, Seed {seed}:", flush=True)
            _set_seed(seed)

            model = copy.deepcopy(teacher.model); model.eval()
            for bid in BLOCKS:
                model.transformer.h[bid].mlp = LinearFactorFFN(model.transformer.h[bid].mlp, rank)

            model.requires_grad_(False)
            params = []
            for bid in BLOCKS:
                for p in model.transformer.h[bid].mlp.encoder.parameters():
                    p.requires_grad_(True); params.append(p)
                for p in model.transformer.h[bid].mlp.decoder.parameters():
                    p.requires_grad_(True); params.append(p)

            total_params = sum(p.numel() for p in params)
            opt = torch.optim.AdamW(params, lr=LR)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=0)
            rng = torch.Generator(); rng.manual_seed(seed)

            for step in range(STEPS):
                idx = torch.randperm(corpus.probe_holdout.shape[0], generator=rng)[:BATCH_SIZE]
                batch = corpus.probe_holdout[idx].to(device)
                opt.zero_grad()
                raw = model(batch[:, :-1])
                logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
                loss.backward(); opt.step(); sched.step()

            model.eval(); model.requires_grad_(False)
            ppl = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)["perplexity"]
            print(f"    PPL: {ppl:.3f} ({total_params:,} params)", flush=True)
            results[key] = {"rank": rank, "seed": seed, "ppl": ppl, "params": total_params}
            del model

    # Summary
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}", flush=True)
    best_rank = None
    for rank in RANKS:
        vals = [results[f"r{rank}_s{s}"]["ppl"] for s in SEEDS]
        m, s = np.mean(vals), np.std(vals)
        p = results[f"r{rank}_s{SEEDS[0]}"]["params"]
        print(f"  Rank {rank}: {m:.2f} ± {s:.2f} ({p:,} params)", flush=True)
        if m < 72 and s < 3:
            best_rank = rank

    if best_rank:
        verdict = "CONFIRMED"
        vals = [results[f"r{best_rank}_s{s}"]["ppl"] for s in SEEDS]
        detail = f"Rank {best_rank}: {np.mean(vals):.1f}±{np.std(vals):.1f} < 72"
    elif all(np.mean([results[f"r{r}_s{s}"]["ppl"] for s in SEEDS]) >= 75 for r in RANKS):
        verdict = "REFUTED"
        detail = "All ranks >= 75"
    else:
        verdict = "INCONCLUSIVE"
        detail = "Some ranks 72-75"

    print(f"  topk baseline: 70.7", flush=True)
    print(f"  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 157, "item": 167, "seeds": SEEDS, "ranks": RANKS,
        "float_ppl": bl["perplexity"],
        "results": results, "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_157_higher_rank_joint.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
