"""Cycle 216 — W_down column geometry census.

Item 230: For each of the 12 GPT-2 small blocks, treat mlp.c_proj.weight
(Conv1D, shape (3072, 768)) as a stack of 3072 columns of W_down in R^768.
Measure: pairwise cosine structure, anti-podal pair count, top-20 singular
values, k=32 KMeans cluster quality, cross-block principal-angle alignment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from sklearn.cluster import KMeans
from transformers import GPT2LMHeadModel


REPO = Path("/Users/mihai/Projects/quant_aware_training")
OUT_JSON = REPO / "autoresearch/results/cycle_216_wdown_geometry.json"
OUT_MD = REPO / "autoresearch/results/cycle_216_summary.md"

N_BLOCKS = 12
N_COLS = 3072
D = 768
KMEANS_K = 32
KMEANS_N_INIT = 5
THRESHOLDS = [-0.95, -0.90, -0.85, -0.80]
SEEDS = [42, 137, 2024]


@torch.no_grad()
def analyze_block(W_down_cols: torch.Tensor, block_idx: int) -> dict:
    """W_down_cols: shape (3072, 768). Each row is one W_down column in R^768."""
    assert W_down_cols.shape == (N_COLS, D)

    # Unit-normalize each row
    cols_norm = W_down_cols / W_down_cols.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # Pairwise cosine matrix
    cos_mat = cols_norm @ cols_norm.T
    iu = torch.triu_indices(N_COLS, N_COLS, offset=1)
    cos_upper = cos_mat[iu[0], iu[1]]

    anti_podal = {}
    for tau in THRESHOLDS:
        count = int((cos_upper < tau).sum().item())
        fraction = min(1.0, 2.0 * count / N_COLS)
        anti_podal[f"{tau}"] = {"count": count, "fraction": fraction}

    cos_stats = {
        "min": float(cos_upper.min().item()),
        "mean": float(cos_upper.mean().item()),
        "std": float(cos_upper.std().item()),
    }

    # SVD of W_down (shape (3072, 768))
    svdvals = torch.linalg.svdvals(W_down_cols)  # length min(3072, 768) = 768
    top20_sv = svdvals[:20].tolist()
    cumsum = svdvals.cumsum(0) / svdvals.sum()
    rank_95 = int((cumsum < 0.95).sum().item() + 1)

    # KMeans k=32 on unit columns
    cols_np = cols_norm.cpu().numpy()
    km = KMeans(n_clusters=KMEANS_K, n_init=KMEANS_N_INIT, random_state=0).fit(cols_np)
    labels = torch.from_numpy(km.labels_.astype("int64"))

    within_vals = []
    within_weights = []
    sizes = []
    for k in range(KMEANS_K):
        mask = labels == k
        n_k = int(mask.sum().item())
        sizes.append(n_k)
        if n_k < 2:
            continue
        pts = cols_norm[mask]
        cm = pts @ pts.T
        iu_k = torch.triu_indices(n_k, n_k, offset=1)
        mean_cos = float(cm[iu_k[0], iu_k[1]].mean().item())
        within_vals.append(mean_cos)
        within_weights.append(n_k)

    within_weights_t = torch.tensor(within_weights, dtype=torch.float64)
    within_vals_t = torch.tensor(within_vals, dtype=torch.float64)
    if within_weights_t.sum() > 0:
        within_cos_mean = float(
            (within_vals_t * within_weights_t).sum() / within_weights_t.sum()
        )
    else:
        within_cos_mean = float("nan")

    C = torch.from_numpy(km.cluster_centers_).to(torch.float32)
    C_norm = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C_cos = C_norm @ C_norm.T
    iu_c = torch.triu_indices(KMEANS_K, KMEANS_K, offset=1)
    between_centroid_cos_mean = float(C_cos[iu_c[0], iu_c[1]].mean().item())

    sizes_sorted = sorted(sizes)
    size_min = int(min(sizes))
    size_max = int(max(sizes))
    size_median = int(sizes_sorted[len(sizes_sorted) // 2])

    claim = (
        f"Block {block_idx}: {100.0 * anti_podal['-0.9']['fraction']:.1f}% of columns have "
        f"an anti-podal twin (cos<-0.9); {rank_95} singular values capture 95% of W_down "
        f"energy; KMeans k=32 within/between = {within_cos_mean:.2f}/"
        f"{between_centroid_cos_mean:.2f}."
    )

    return {
        "block": block_idx,
        "anti_podal": anti_podal,
        "cos_stats": cos_stats,
        "top20_singular_values": top20_sv,
        "rank_95": rank_95,
        "kmeans": {
            "within_cos_mean": within_cos_mean,
            "between_centroid_cos_mean": between_centroid_cos_mean,
            "cluster_sizes": {
                "min": size_min,
                "median": size_median,
                "max": size_max,
                "all": sizes,
            },
        },
        "claim_one_sentence": claim,
    }


@torch.no_grad()
def random_baseline() -> dict:
    """Sample 3072 unit vectors in R^768, compute anti-podal stats at each threshold."""
    per_seed_counts = {f"{tau}": [] for tau in THRESHOLDS}
    min_cos_per_seed = []
    iu = torch.triu_indices(N_COLS, N_COLS, offset=1)
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        X = torch.randn(N_COLS, D, generator=g)
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cos_mat = X @ X.T
        cos_upper = cos_mat[iu[0], iu[1]]
        min_cos_per_seed.append(float(cos_upper.min().item()))
        for tau in THRESHOLDS:
            per_seed_counts[f"{tau}"].append(int((cos_upper < tau).sum().item()))

    out = {"seeds": SEEDS, "min_cos_per_seed": min_cos_per_seed}
    for tau in THRESHOLDS:
        counts = torch.tensor(per_seed_counts[f"{tau}"], dtype=torch.float64)
        count_mean = float(counts.mean().item())
        count_std = float(counts.std(unbiased=False).item())
        frac_mean = min(1.0, 2.0 * count_mean / N_COLS)
        frac_std = 2.0 * count_std / N_COLS
        out[f"anti_podal_at_{tau}"] = {
            "counts_per_seed": per_seed_counts[f"{tau}"],
            "count_mean": count_mean,
            "count_std": count_std,
            "fraction_mean": frac_mean,
            "fraction_std": frac_std,
        }
    return out


@torch.no_grad()
def cross_block_alignment(model) -> list:
    """For l in 0..10: principal angles between V_l (top-20 right SVs of
    c_proj in block l, the W_down writing subspace in R^768) and U_{l+1}
    (top-20 left SVs of c_fc in block l+1, the W_up reading subspace in R^768).
    """
    out = []
    for l in range(N_BLOCKS - 1):
        W_down = model.transformer.h[l].mlp.c_proj.weight.detach().float()  # (3072, 768)
        W_up_next = model.transformer.h[l + 1].mlp.c_fc.weight.detach().float()  # (768, 3072)

        # Right singular vectors of W_down (shape (3072, 768)):
        # svd returns Vh of shape (768, 768); V_l = Vh[:20, :].T  shape (768, 20)
        U_d, S_d, Vh_d = torch.linalg.svd(W_down, full_matrices=False)
        V_l = Vh_d[:20, :].T  # (768, 20)

        # Left singular vectors of W_up_next (shape (768, 3072)):
        # U has shape (768, 768); take U_next[:, :20]  shape (768, 20)
        U_u, S_u, Vh_u = torch.linalg.svd(W_up_next, full_matrices=False)
        U_next = U_u[:, :20]  # (768, 20)

        M = V_l.T @ U_next  # (20, 20)
        sv = torch.linalg.svdvals(M)  # length 20, in [0, 1]
        sv_list = sv.tolist()
        out.append(
            {
                "l": l,
                "lp1": l + 1,
                "principal_cosines_top3": sv_list[:3],
                "principal_cosines_top5_mean": float(sv[:5].mean().item()),
                "principal_cosines_mean": float(sv.mean().item()),
                "principal_cosines_all": sv_list,
            }
        )
    return out


def compute_verdict(per_block: list) -> str:
    key_blocks = [1, 2, 3, 4, 5]
    confirmed = all(
        per_block[l]["anti_podal"]["-0.9"]["fraction"] >= 0.20 for l in key_blocks
    ) and all(
        per_block[l]["kmeans"]["within_cos_mean"] > 0.5 for l in key_blocks
    ) and all(
        per_block[l]["kmeans"]["between_centroid_cos_mean"] < 0.2 for l in key_blocks
    )
    refuted = all(
        per_block[l]["anti_podal"]["-0.9"]["fraction"] < 0.05 for l in range(N_BLOCKS)
    ) and all(
        per_block[l]["kmeans"]["within_cos_mean"] < 0.2 for l in range(N_BLOCKS)
    )
    if confirmed:
        return "CONFIRMED"
    if refuted:
        return "REFUTED"
    return "INCONCLUSIVE"


def write_summary_md(data: dict, path: Path) -> None:
    per_block = data["blocks"]
    verdict = data["verdict"]
    rb = data["random_baseline"]
    rb09 = rb["anti_podal_at_-0.9"]
    rb08 = rb["anti_podal_at_-0.8"]
    xblock = data["cross_block_alignment"]

    # Headline fractions for blocks 1-5
    b15_frac = [per_block[l]["anti_podal"]["-0.9"]["fraction"] for l in range(1, 6)]
    b15_max = max(b15_frac)
    b15_min = min(b15_frac)

    lines = []
    lines.append(f"# Cycle 216 — W_down Column Geometry Census (Item 230)\n")
    lines.append(f"**Verdict: {verdict}**\n\n")
    lines.append(
        "## Summary\n\n"
        "GPT-2 small c_proj.weight in each of the 12 blocks has shape (3072, 768). "
        "We treat the 3072 rows as column vectors of W_down in R^768 and measure "
        "pairwise cosine structure, near-anti-podal pair counts at thresholds "
        "{-0.95, -0.9, -0.85, -0.8}, top-20 singular values, and KMeans k=32 "
        "cluster quality. We also compute principal angles between block l's W_down "
        "right-singular (writing) subspace and block l+1's c_fc left-singular (reading) "
        "subspace. A random baseline samples Gaussian unit vectors in R^768 with seeds "
        f"{SEEDS}.\n\n"
    )
    lines.append(
        f"**Blocks 1-5 anti-podal fraction at cos<-0.9:** "
        f"min={100.0 * b15_min:.2f}%, max={100.0 * b15_max:.2f}%. "
        f"**Random baseline (cos<-0.9):** count_mean={rb09['count_mean']:.2f} "
        f"(fraction={100.0 * rb09['fraction_mean']:.4f}%), "
        f"count_std={rb09['count_std']:.2f}. "
        f"**Random baseline (cos<-0.8):** count_mean={rb08['count_mean']:.2f} "
        f"(fraction={100.0 * rb08['fraction_mean']:.4f}%).\n\n"
    )
    mean_xb_top5 = sum(x["principal_cosines_top5_mean"] for x in xblock) / len(xblock)
    mean_xb_all = sum(x["principal_cosines_mean"] for x in xblock) / len(xblock)
    lines.append(
        f"**Cross-block alignment headline:** across the 11 consecutive pairs, mean of "
        f"top-5 principal-angle cosines = {mean_xb_top5:.3f}, mean of all 20 cosines = "
        f"{mean_xb_all:.3f}. (Cosine=1 means the subspaces are identical; 0 means orthogonal.)\n\n"
    )

    lines.append("## Per-block table\n\n")
    lines.append("| block | %twin@-0.9 | %twin@-0.8 | within_cos | between_cos | rank_95 | xblock_top5 |\n")
    lines.append("|------:|----------:|----------:|-----------:|------------:|--------:|------------:|\n")
    for l in range(N_BLOCKS):
        bl = per_block[l]
        frac09 = bl["anti_podal"]["-0.9"]["fraction"]
        frac08 = bl["anti_podal"]["-0.8"]["fraction"]
        w = bl["kmeans"]["within_cos_mean"]
        b = bl["kmeans"]["between_centroid_cos_mean"]
        r95 = bl["rank_95"]
        if l < N_BLOCKS - 1:
            xb = xblock[l]["principal_cosines_top5_mean"]
            xb_str = f"{xb:.3f}"
        else:
            xb_str = "—"
        lines.append(
            f"| {l} | {100.0 * frac09:6.2f}% | {100.0 * frac08:6.2f}% | {w:+.3f} | "
            f"{b:+.3f} | {r95:4d} | {xb_str} |\n"
        )
    lines.append("\n")

    lines.append("## Per-block one-sentence claims\n\n")
    for l in range(N_BLOCKS):
        lines.append(f"- {per_block[l]['claim_one_sentence']}\n")
    lines.append("\n")

    lines.append("## Test claim and verdict\n\n")
    lines.append(
        "> \"GPT-2 W_down contains enough anti-podal column pairs that GELU's bump "
        "effectively doubles channel capacity via bidirectional pushes.\"\n\n"
    )
    lines.append(
        f"**Verdict: {verdict}**. Confirmed required (a) ≥20% anti-podal fraction at "
        f"cos<-0.9 in every block in {{1..5}}, (b) within-cluster cos>0.5 in every "
        f"block in {{1..5}}, and (c) between-centroid cos<0.2 in every block in "
        f"{{1..5}}. Refuted required <5% anti-podal fraction at cos<-0.9 in ALL 12 "
        f"blocks AND within-cluster cos<0.2 in ALL 12 blocks.\n"
    )
    path.write_text("".join(lines))


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    print("[cycle216] loading gpt2 ...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    W_d0 = model.transformer.h[0].mlp.c_proj.weight
    W_u0 = model.transformer.h[0].mlp.c_fc.weight
    assert W_d0.shape == (N_COLS, D), f"c_proj shape {tuple(W_d0.shape)} != (3072, 768)"
    assert W_u0.shape == (D, N_COLS), f"c_fc shape {tuple(W_u0.shape)} != (768, 3072)"
    print(f"[cycle216] shapes OK  c_proj={tuple(W_d0.shape)}  c_fc={tuple(W_u0.shape)}")

    per_block = []
    for l in range(N_BLOCKS):
        t_l = time.time()
        W_down_cols = model.transformer.h[l].mlp.c_proj.weight.detach().float()
        rec = analyze_block(W_down_cols, l)
        per_block.append(rec)
        print(
            f"[cycle216] block {l:2d} done in {time.time() - t_l:5.1f}s — "
            f"{rec['claim_one_sentence']}"
        )

    print("[cycle216] cross-block alignment ...")
    xblock = cross_block_alignment(model)
    print("[cycle216] random baseline ...")
    rb = random_baseline()

    verdict = compute_verdict(per_block)
    print(f"[cycle216] VERDICT = {verdict}")

    config = {
        "model": "gpt2",
        "n_blocks": N_BLOCKS,
        "n_cols": N_COLS,
        "d": D,
        "kmeans_k": KMEANS_K,
        "kmeans_n_init": KMEANS_N_INIT,
        "thresholds": THRESHOLDS,
        "random_seeds": SEEDS,
    }

    data = {
        "cycle": 216,
        "item": 230,
        "verdict": verdict,
        "config": config,
        "blocks": per_block,
        "cross_block_alignment": xblock,
        "random_baseline": rb,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2))
    print(f"[cycle216] wrote {OUT_JSON}")

    write_summary_md(data, OUT_MD)
    print(f"[cycle216] wrote {OUT_MD}")

    print(f"[cycle216] total wall time {time.time() - t0:.1f}s")
    print(f"[cycle216] VERDICT = {verdict}")


if __name__ == "__main__":
    main()
