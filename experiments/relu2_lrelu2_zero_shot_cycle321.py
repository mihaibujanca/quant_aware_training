"""
Cycle 321 / item 330 — Zero-shot ReLU² and LeakyReLU² activation patches on
GPT-2 small via cycle 312 unified curvature framework.

Runs UnifiedCurvatureFramework with three (sigma, sigma_lin) pairs on the
same calib slice (200 wikitext-2 val × 128 tok, seed 42, GPT-2 small):
  (a) sigma=ReLU²,            sigma_lin=ReLU
  (b) sigma=LeakyReLU²(0.5),  sigma_lin=ReLU²
  (c) sigma=GELU,             sigma_lin=ReLU   (cross-reference, cycle 302
                                                reproduction)

For each pair, runs panel 1 (raw rank_95) and panel 2 (centred c_fc/Q/K/V
top-5 alignment) per block. Reports a 3×N profile and applies item 330's
verdict tree.

NB: ReLU² and LeakyReLU² are pure-functional swaps applied to the GELU-trained
GPT-2 pre-activation distribution. The pre-activation distribution z(x) is
GELU-tuned, NOT ReLU²-tuned, so this is a *predictive* measurement: it
measures what the curvature residual would be if those activations were
applied to the trained model's pre-activation distribution. Approximate but
immediate. Trainable params: 0.

Verdict (item 330 verbatim):
  CONFIRMED-distinct-profiles iff (ReLU², ReLU) and (LeakyReLU², ReLU²)
    produce per-block rank/alignment profiles measurably different from
    (GELU, ReLU) — at least one of the measurements differs by ≥30% relative
    at ≥4 blocks per pair.
  CONFIRMED-identical-profiles iff all 3 pairs give measurements within ±10%
    relative at every block.
  INCONCLUSIVE otherwise.

Note: the cycle 302 panel 5 (κ shift under sigma_lin + b_leak swap) is
omitted to keep wall under the 15 min agenda budget. Future cycle can
revisit panel 5 separately for the (ReLU², ReLU) and (LeakyReLU², ReLU²)
pairs if the panel 1/2 result is conclusive.
"""
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from experiments.curvature_residual_geometric import UnifiedCurvatureFramework

CYCLE = 321
ITEM = 330

N_CALIB = 200
SEQ_LEN = 128
SEED = 42
BLOCKS = list(range(12))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "autoresearch" / "results"
LOGS_DIR = REPO / "autoresearch" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS_DIR / f"cycle_{CYCLE}_relu2_lrelu2_zero_shot.json"
SUMMARY_PATH = RESULTS_DIR / f"cycle_{CYCLE}_summary.md"
LOG_PATH = LOGS_DIR / f"cycle_{CYCLE}_run.log"


def sigma_relu(z):
    return F.relu(z)


def sigma_relu2(z):
    r = F.relu(z)
    return r * r


def sigma_lrelu2(z):
    """LeakyReLU²(slope=0.5) — i.e. leaky_relu(z, 0.5)²."""
    h = F.leaky_relu(z, negative_slope=0.5)
    return h * h


def sigma_gelu(z):
    return F.gelu(z, approximate="tanh")


PAIRS = [
    ("relu2", "relu", sigma_relu2, sigma_relu),
    ("lrelu2", "relu2", sigma_lrelu2, sigma_relu2),
    ("gelu", "relu", sigma_gelu, sigma_relu),
]


def main():
    torch.manual_seed(SEED)
    log_lines = []
    log_lines.append(f"Cycle {CYCLE} / item {ITEM} — Zero-shot ReLU²/lrelu² activation patches")
    log_lines.append(f"Device: {DEVICE}")
    log_lines.append(f"3 (sigma, sigma_lin) pairs × 12 blocks × 2 panels (raw rank, centred alignment)")
    print(log_lines[-3], flush=True)
    print(log_lines[-2], flush=True)
    print(log_lines[-1], flush=True)

    print("Loading GPT-2 + wikitext-2 val...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
    all_ids = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        if enc["input_ids"].shape[1] >= SEQ_LEN:
            all_ids.append(enc["input_ids"][:, :SEQ_LEN])
        if len(all_ids) >= N_CALIB:
            break
    calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
    log_lines.append(f"Calib: {len(calib_ids)} seqs × {SEQ_LEN} = {N_CALIB * SEQ_LEN} tokens")
    print(log_lines[-1], flush=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    pair_results = []
    t_start = time.time()
    for sigma_name, sigma_lin_name, sigma_fn, sigma_lin_fn in PAIRS:
        print(f"\n── Pair ({sigma_name}, {sigma_lin_name}) ──", flush=True)
        log_lines.append(f"\nPair ({sigma_name}, {sigma_lin_name})")
        t0 = time.time()
        fw = UnifiedCurvatureFramework(
            model=model,
            sigma=sigma_fn,
            sigma_lin=sigma_lin_fn,
            blocks=BLOCKS,
            calib_ids=calib_ids,
            device=DEVICE,
            tokenizer=None,
            A_factors=None,
            sigma_name=sigma_name,
            sigma_lin_name=sigma_lin_name,
        )
        # Capture once (cached in fw); subsequent pairs reuse the model state
        # but each pair builds its own framework with its own (sigma, sigma_lin).
        # capture_inputs() is sigma-independent (it captures pre-MLP residual
        # stream inputs to c_fc), so it could be shared, but each fw instance
        # has its own _mlp_inputs cache. Acceptable cost.
        fw.capture_inputs()

        per_block = {}
        for l in BLOCKS:
            p1 = fw.panel_raw_rank(l)
            p2 = fw.panel_centred_alignment(l, l_next=l + 1 if l + 1 < 12 else None)
            per_block[l] = {
                "rank_95_raw": p1["rank_95_raw"],
                "rank_99_raw": p1["rank_99_raw"],
                "top5_singular_values_raw": p1["top5_singular_values_raw"],
                "delta_norm_mean": p1["delta_norm_mean"],
                "rank_95_centred": p2["rank_95_centred"],
                "rank_99_centred": p2["rank_99_centred"],
                "sigma0": p2["sigma0"],
                "sigma1": p2["sigma1"],
                "sigma2": p2["sigma2"],
                "c_fc_top5_cos_sum": p2.get("c_fc_top5_cos_sum"),
                "Q_top5_cos_sum": p2.get("Q_top5_cos_sum"),
                "K_top5_cos_sum": p2.get("K_top5_cos_sum"),
                "V_top5_cos_sum": p2.get("V_top5_cos_sum"),
            }
            fw.free_block(l)
            print(
                f"  block {l}: rank_95_raw={p1['rank_95_raw']} "
                f"delta_norm_mean={p1['delta_norm_mean']:.3f} "
                f"c_fc_top5_cos_sum={p2.get('c_fc_top5_cos_sum')}",
                flush=True,
            )

        wall = time.time() - t0
        log_lines.append(f"  pair wall {wall:.1f}s")
        pair_results.append({
            "sigma": sigma_name,
            "sigma_lin": sigma_lin_name,
            "per_block": per_block,
            "wall_sec": wall,
        })
        del fw  # free framework caches

    total_wall = time.time() - t_start

    # ── Verdict (item 330 verbatim) ───────────────────────────────────────────
    def get_pair(name_pair):
        for p in pair_results:
            if (p["sigma"], p["sigma_lin"]) == name_pair:
                return p
        return None

    relu2_relu = get_pair(("relu2", "relu"))
    lrelu2_relu2 = get_pair(("lrelu2", "relu2"))
    gelu_relu = get_pair(("gelu", "relu"))

    def rel_diff(a, b):
        denom = max(abs(b), 1e-12)
        return abs(a - b) / denom

    def block_metric_diffs(target_pair, ref_pair, metric):
        diffs = []
        for l in BLOCKS:
            a = target_pair["per_block"][l].get(metric)
            r = ref_pair["per_block"][l].get(metric)
            if a is None or r is None:
                diffs.append(None)
            else:
                diffs.append(rel_diff(a, r))
        return diffs

    def n_blocks_diff_ge(diffs, threshold):
        return sum(1 for d in diffs if d is not None and d >= threshold)

    metrics = ["rank_95_raw", "delta_norm_mean", "c_fc_top5_cos_sum"]

    relu2_vs_gelu = {m: block_metric_diffs(relu2_relu, gelu_relu, m) for m in metrics}
    lrelu2_vs_gelu = {m: block_metric_diffs(lrelu2_relu2, gelu_relu, m) for m in metrics}

    relu2_distinct = any(n_blocks_diff_ge(relu2_vs_gelu[m], 0.30) >= 4 for m in metrics)
    lrelu2_distinct = any(n_blocks_diff_ge(lrelu2_vs_gelu[m], 0.30) >= 4 for m in metrics)
    distinct = relu2_distinct and lrelu2_distinct

    relu2_within_10 = all(
        all((d is None or d <= 0.10) for d in relu2_vs_gelu[m]) for m in metrics
    )
    lrelu2_within_10 = all(
        all((d is None or d <= 0.10) for d in lrelu2_vs_gelu[m]) for m in metrics
    )
    identical = relu2_within_10 and lrelu2_within_10

    if distinct:
        verdict = "CONFIRMED-distinct-profiles"
    elif identical:
        verdict = "CONFIRMED-identical-profiles"
    else:
        verdict = "INCONCLUSIVE"

    result = {
        "cycle": CYCLE,
        "item": ITEM,
        "config": {
            "N_CALIB": N_CALIB,
            "SEQ_LEN": SEQ_LEN,
            "SEED": SEED,
            "BLOCKS": BLOCKS,
            "DEVICE": DEVICE,
            "model": "gpt2",
            "n_pairs": len(PAIRS),
            "pairs": [(s, sl) for s, sl, _, _ in PAIRS],
            "panels_run": ["panel_raw_rank", "panel_centred_alignment"],
            "panel5_kappa_shift": "OMITTED (would 3x wall budget)",
        },
        "pair_results": pair_results,
        "verdict_logic": {
            "relu2_vs_gelu_relu_relative_diffs": relu2_vs_gelu,
            "lrelu2_vs_gelu_relu_relative_diffs": lrelu2_vs_gelu,
            "relu2_distinct_any_metric_ge4_blocks_ge30pc": relu2_distinct,
            "lrelu2_distinct_any_metric_ge4_blocks_ge30pc": lrelu2_distinct,
            "relu2_within_10pc_all_blocks_all_metrics": relu2_within_10,
            "lrelu2_within_10pc_all_blocks_all_metrics": lrelu2_within_10,
        },
        "verdict": verdict,
        "total_wall_sec": total_wall,
    }
    JSON_PATH.write_text(json.dumps(result, indent=2))

    # ── Summary markdown ──────────────────────────────────────────────────────
    lines = []
    lines.append(f"# Cycle {CYCLE} — Item {ITEM}: Zero-shot ReLU² / LeakyReLU² patches on GPT-2 small")
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- GPT-2 small, wikitext-2 val, {N_CALIB} seqs × {SEQ_LEN} tokens, seed {SEED}.")
    lines.append("- Three (σ, σ_lin) pairs: (ReLU², ReLU), (LeakyReLU²(0.5), ReLU²), (GELU, ReLU).")
    lines.append("- Panels run: panel_raw_rank (rank_95(Δy)), panel_centred_alignment (c_fc/Q/K/V top-5 cos sums).")
    lines.append("- Panel 5 (κ shift under sigma_lin + b_leak swap) OMITTED to keep wall under budget.")
    lines.append("- Trainable params: 0 (zero-shot patches).")
    lines.append("")

    for p in pair_results:
        lines.append(f"## Pair ({p['sigma']}, {p['sigma_lin']})")
        lines.append("")
        lines.append("| block | rank_95_raw | rank_99_raw | top1_sv | ||Δy||/N | rank_95_centred | c_fc_top5_cos_sum |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for l in BLOCKS:
            b = p["per_block"][l]
            cfc = b["c_fc_top5_cos_sum"]
            cfc_str = f"{cfc:.3f}" if cfc is not None else "N/A"
            lines.append(
                f"| {l} | {b['rank_95_raw']} | {b['rank_99_raw']} | "
                f"{b['top5_singular_values_raw'][0]:.3f} | {b['delta_norm_mean']:.3f} | "
                f"{b['rank_95_centred']} | {cfc_str} |"
            )
        lines.append("")

    lines.append("## Verdict logic (item 330 verbatim)")
    lines.append(
        "- CONFIRMED-distinct-profiles iff (ReLU², ReLU) and (LeakyReLU², ReLU²) produce per-block rank/alignment profiles measurably different from (GELU, ReLU) — at least one measurement differs by ≥30% relative at ≥4 blocks per pair."
    )
    lines.append("- CONFIRMED-identical-profiles iff all 3 pairs give measurements within ±10% relative at every block.")
    lines.append("- INCONCLUSIVE otherwise.")
    lines.append("")
    lines.append(f"- relu2 vs gelu_relu — distinct (≥4 blocks ≥30% in any metric): {relu2_distinct}")
    lines.append(f"- lrelu2 vs gelu_relu — distinct (≥4 blocks ≥30% in any metric): {lrelu2_distinct}")
    lines.append(f"- relu2 vs gelu_relu — identical (within ±10% all blocks all metrics): {relu2_within_10}")
    lines.append(f"- lrelu2 vs gelu_relu — identical (within ±10% all blocks all metrics): {lrelu2_within_10}")
    lines.append("")
    lines.append(f"**Wall:** {total_wall:.1f} s")
    SUMMARY_PATH.write_text("\n".join(lines))

    LOG_PATH.write_text("\n".join(log_lines))

    print(f"\n[cycle {CYCLE}] verdict = {verdict}", flush=True)
    print(f"[cycle {CYCLE}] wrote {JSON_PATH}", flush=True)
    print(f"[cycle {CYCLE}] wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
