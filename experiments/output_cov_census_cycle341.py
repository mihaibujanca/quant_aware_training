"""
Cycle 341 — Item 350: Output-covariance spectrum census on GPT-2.

Builds on cycle 278 (identical measurement: all 12 blocks, 200 wikitext-2 val seqs, 128 tokens,
seed 42, hook on mlp.c_proj output, covariance C = Y^T Y / N, eigh, exp(entropy) eff_rank).

Cycle 278 stored top-20 eigenvalues + total_energy per block, so top-10 and top-13 energy
can be computed from stored data. Combines with cycle 338 weight SVD for comparison table.

Identifies LASER-viable blocks: top-5 output-covariance energy >= 50% (strong) or >= 30% (moderate).
"""
import json
import math

# ── Load cycle 278 output-covariance data ──
with open("autoresearch/results/cycle_278_all_blocks_eff_rank.json") as f:
    c278 = json.load(f)

# ── Load cycle 338 weight SVD data ──
with open("autoresearch/results/cycle_338_spectrum_census.json") as f:
    c338 = json.load(f)

per_block_278 = c278["per_block"]
per_block_338 = {str(b["block"]): b for b in c338["per_block"]}

# ── Compute top-k energy fractions from stored eigenvalues ──
results = []
for b in range(12):
    bkey = str(b)
    d278 = per_block_278[bkey]
    d338 = per_block_338[bkey]

    eigvals = d278["top20_mlp_eigvals"]  # top-20 eigenvalues, descending
    total = d278["total_energy"]

    # Output-covariance energy fractions
    top1_out = sum(eigvals[:1]) / total
    top5_out = sum(eigvals[:5]) / total
    top10_out = sum(eigvals[:10]) / total
    top13_out = sum(eigvals[:13]) / total
    eff_rank = d278["eff_rank"]

    # Verify top-1 and top-5 match stored values
    stored_top1 = d278["topk_energy"]["1"]
    stored_top5 = d278["topk_energy"]["5"]
    assert abs(top1_out - stored_top1) < 1e-3, f"block {b} top1 mismatch: {top1_out:.4f} vs {stored_top1:.4f}"
    assert abs(top5_out - stored_top5) < 1e-3, f"block {b} top5 mismatch: {top5_out:.4f} vs {stored_top5:.4f}"

    # Weight SVD energy fractions (from cycle 338)
    top1_wt = d338["top1_pct"] / 100.0
    top5_wt = d338["top5_pct"] / 100.0
    top13_wt = d338["top13_pct"] / 100.0

    # LASER viability
    if top5_out >= 0.50:
        laser_label = "STRONG"
    elif top5_out >= 0.30:
        laser_label = "MODERATE"
    else:
        laser_label = "WEAK"

    row = {
        "block": b,
        "out_cov_eff_rank": round(eff_rank, 2),
        "out_cov_top1_pct": round(top1_out * 100, 2),
        "out_cov_top5_pct": round(top5_out * 100, 2),
        "out_cov_top10_pct": round(top10_out * 100, 2),
        "out_cov_top13_pct": round(top13_out * 100, 2),
        "wt_svd_top1_pct": d338["top1_pct"],
        "wt_svd_top5_pct": d338["top5_pct"],
        "wt_svd_top13_pct": d338["top13_pct"],
        "laser_viability": laser_label,
    }
    results.append(row)

# ── Print comparison table ──
print("=" * 120)
print(f"{'Block':>5} | {'EffRank':>8} | {'Out-Cov top1%':>13} | {'Out-Cov top5%':>13} | {'Out-Cov top10%':>14} | {'Wt-SVD top5%':>12} | {'LASER':>8}")
print("-" * 120)
for r in results:
    print(f"{r['block']:>5} | {r['out_cov_eff_rank']:>8.2f} | {r['out_cov_top1_pct']:>12.2f}% | {r['out_cov_top5_pct']:>12.2f}% | {r['out_cov_top10_pct']:>13.2f}% | {r['wt_svd_top5_pct']:>11.2f}% | {r['laser_viability']:>8}")
print("=" * 120)

# ── Count LASER-viable blocks ──
strong = [r for r in results if r["laser_viability"] == "STRONG"]
moderate = [r for r in results if r["laser_viability"] == "MODERATE"]
weak = [r for r in results if r["laser_viability"] == "WEAK"]

print(f"\nLASER-viable blocks (top-5 output-covariance energy >= 50%): {len(strong)}")
for r in strong:
    print(f"  block {r['block']}: eff_rank={r['out_cov_eff_rank']}, top5={r['out_cov_top5_pct']:.1f}%")
print(f"\nModerate (30-50%): {len(moderate)}")
for r in moderate:
    print(f"  block {r['block']}: eff_rank={r['out_cov_eff_rank']}, top5={r['out_cov_top5_pct']:.1f}%")
print(f"\nWeak (<30%): {len(weak)}")
for r in weak:
    print(f"  block {r['block']}: eff_rank={r['out_cov_eff_rank']}, top5={r['out_cov_top5_pct']:.1f}%")

# ── Verdict ──
n_strong = len(strong)
only_b1_strong = (n_strong == 1 and strong[0]["block"] == 1)
all_others_below_30 = all(r["out_cov_top5_pct"] < 30 for r in results if r["block"] != 1)

if n_strong >= 3:
    verdict = "CONFIRMED"
    verdict_detail = f"{n_strong} blocks have top-5 output-covariance energy >= 50%"
elif only_b1_strong and all_others_below_30:
    verdict = "REFUTED"
    verdict_detail = "Only b1 has top-5 >= 50%, all others below 30%"
else:
    verdict = "INCONCLUSIVE"
    verdict_detail = f"{n_strong} strong blocks, {len(moderate)} moderate blocks"

print(f"\nVERDICT: {verdict}")
print(f"  {verdict_detail}")

# ── Concentration ratio: out-cov top-5 / wt-svd top-5 ──
print("\n--- Concentration amplification: output-cov / weight-SVD ---")
for r in results:
    ratio = r["out_cov_top5_pct"] / r["wt_svd_top5_pct"] if r["wt_svd_top5_pct"] > 0 else float('inf')
    print(f"  block {r['block']}: {ratio:.1f}x  (out-cov {r['out_cov_top5_pct']:.1f}% vs wt-svd {r['wt_svd_top5_pct']:.1f}%)")

# ── Save JSON ──
output = {
    "cycle": 341,
    "item": 350,
    "verdict": verdict,
    "verdict_detail": verdict_detail,
    "n_strong_laser": n_strong,
    "n_moderate_laser": len(moderate),
    "n_weak_laser": len(weak),
    "strong_blocks": [r["block"] for r in strong],
    "moderate_blocks": [r["block"] for r in moderate],
    "weak_blocks": [r["block"] for r in weak],
    "per_block": results,
    "source_data": {
        "output_covariance": "cycle_278 (200 wikitext-2 val seqs, 128 tokens, seed 42, mlp.c_proj hook)",
        "weight_svd": "cycle_338 (c_proj.weight SVD, 768 singular values)",
    },
}
out_path = "autoresearch/results/cycle_341_output_cov_census.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
