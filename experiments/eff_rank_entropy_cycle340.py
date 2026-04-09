"""
Cycle 340 — Recompute eff_rank with exp(entropy) formula.
Loads cycle 332 checkpoints, hooks mlp.c_proj output, computes covariance eigenvalues,
then exp(-Σ p_i ln p_i) where p_i = λ_i / Σλ.
"""
import json, math, os, sys
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F

# Import model + data from cycle 331/332
sys.path.insert(0, os.path.dirname(__file__))
from tiny_gpt_act_compare_cycle331 import (
    TinyGPT, load_data, make_batches,
    N_LAYER, N_EMBD, N_INNER, BLOCK_SIZE, BATCH_SIZE,
    N_CALIB_SEQ, CKPT_DIR, ACTIVATIONS, SEEDS, DEVICE
)

RESULTS_JSON = "autoresearch/results/cycle_340_eff_rank_entropy.json"
os.makedirs("autoresearch/results", exist_ok=True)


def exp_entropy_eff_rank(eigenvalues: torch.Tensor) -> float:
    """Compute exp(entropy) effective rank from eigenvalues.
    p_i = λ_i / Σλ, H = -Σ p_i ln p_i, eff_rank = exp(H).
    """
    # Clamp to non-negative (numerical noise)
    eigs = eigenvalues.clamp(min=0)
    total = eigs.sum()
    if total < 1e-12:
        return 1.0
    p = eigs / total
    # Avoid log(0): only sum where p > 0
    mask = p > 1e-30
    H = -(p[mask] * p[mask].log()).sum()
    return math.exp(H.item())


def participation_ratio(eigenvalues: torch.Tensor) -> float:
    """Participation ratio for comparison: (Σλ)² / Σλ²."""
    eigs = eigenvalues.clamp(min=0)
    s1 = eigs.sum()
    s2 = (eigs ** 2).sum()
    if s2 < 1e-12:
        return 1.0
    return (s1 ** 2 / s2).item()


def measure_eff_rank(model, calib_x):
    """Hook c_proj outputs, run calib data, compute covariance eigenvalues."""
    model.eval()
    n_blocks = len(model.blocks)
    activations = {i: [] for i in range(n_blocks)}

    hooks = []
    for i, block in enumerate(model.blocks):
        def make_hook(idx):
            def hook_fn(module, input, output):
                # output shape: (B, T, N_EMBD) — c_proj output
                activations[idx].append(output.detach().cpu())
            return hook_fn
        h = block.mlp.c_proj.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for batch_start in range(0, calib_x.shape[0], BATCH_SIZE):
            batch = calib_x[batch_start:batch_start + BATCH_SIZE].to(DEVICE)
            model(batch)

    for h in hooks:
        h.remove()

    results = {}
    for i in range(n_blocks):
        # Concatenate all activations: (N, T, D) -> (N*T, D)
        act = torch.cat(activations[i], dim=0)
        act = act.reshape(-1, act.shape[-1]).float()  # (N*T, D)
        # Covariance
        act_centered = act - act.mean(dim=0, keepdim=True)
        cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
        # Eigenvalues on CPU (MPS eigh can be flaky)
        eigs = torch.linalg.eigvalsh(cov.cpu())
        eigs = eigs.flip(0)  # descending

        results[i] = {
            "exp_entropy": exp_entropy_eff_rank(eigs),
            "participation_ratio": participation_ratio(eigs),
        }
    return results


def main():
    print("Loading data...")
    train_ids, val_ids = load_data()
    calib_x, _ = make_batches(val_ids, N_CALIB_SEQ, BLOCK_SIZE, BATCH_SIZE)

    all_results = {"exp_entropy": {}, "participation_ratio": {}}
    for act in ACTIVATIONS:
        all_results["exp_entropy"][act] = {}
        all_results["participation_ratio"][act] = {}

    for act in ACTIVATIONS:
        for seed in SEEDS:
            ckpt_path = os.path.join(CKPT_DIR, f"{act}_s{seed}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {ckpt_path} (not found)")
                continue
            print(f"Loading {act} seed={seed}...")
            model = TinyGPT(act).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(sd)
            block_results = measure_eff_rank(model, calib_x)

            for block_idx, metrics in block_results.items():
                block_key = str(block_idx)
                if block_key not in all_results["exp_entropy"][act]:
                    all_results["exp_entropy"][act][block_key] = []
                    all_results["participation_ratio"][act][block_key] = []
                all_results["exp_entropy"][act][block_key].append(metrics["exp_entropy"])
                all_results["participation_ratio"][act][block_key].append(metrics["participation_ratio"])
            del model
            torch.mps.empty_cache() if DEVICE.type == "mps" else None

    # Compute means and format
    summary = {"exp_entropy": {}, "participation_ratio": {}, "comparison": {}}
    for metric in ["exp_entropy", "participation_ratio"]:
        for act in ACTIVATIONS:
            summary[metric][act] = {}
            for block_key in sorted(all_results[metric][act].keys(), key=int):
                vals = all_results[metric][act][block_key]
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                summary[metric][act][block_key] = {
                    "mean": round(mean, 2),
                    "std": round(std, 2),
                    "per_seed": [round(v, 2) for v in vals],
                }

    # Print table
    print("\n" + "=" * 80)
    print("EXP(ENTROPY) EFFECTIVE RANK — per block, mean ± std across 3 seeds")
    print("=" * 80)
    header = f"{'Block':>6}"
    for act in ACTIVATIONS:
        header += f"  {act:>20}"
    print(header)
    print("-" * 70)
    for b in range(N_LAYER):
        row = f"{b:>6}"
        for act in ACTIVATIONS:
            d = summary["exp_entropy"][act][str(b)]
            row += f"  {d['mean']:>8.1f} ± {d['std']:>5.1f}   "
        print(row)

    print("\nPARTICIPATION RATIO (recomputed) — for comparison")
    print("-" * 70)
    for b in range(N_LAYER):
        row = f"{b:>6}"
        for act in ACTIVATIONS:
            d = summary["participation_ratio"][act][str(b)]
            row += f"  {d['mean']:>8.1f} ± {d['std']:>5.1f}   "
        print(row)

    # Verdict check
    print("\n" + "=" * 80)
    print("VERDICT CHECK: squared > GELU by >=10% relative at >=2 blocks?")
    print("=" * 80)
    blocks_pass = {"relu2": 0, "lrelu2": 0}
    for b in range(N_LAYER):
        gelu_mean = summary["exp_entropy"]["gelu"][str(b)]["mean"]
        for act in ["relu2", "lrelu2"]:
            act_mean = summary["exp_entropy"][act][str(b)]["mean"]
            rel_diff = (act_mean - gelu_mean) / gelu_mean * 100
            tag = "PASS" if rel_diff >= 10 else ("FAIL" if rel_diff < 5 else "MARGINAL")
            print(f"  Block {b}: {act} vs GELU = {rel_diff:+.1f}% [{tag}]")
            if rel_diff >= 10:
                blocks_pass[act] += 1

    confirmed = any(v >= 2 for v in blocks_pass.values())
    verdict = "CONFIRMED" if confirmed else "INCONCLUSIVE"
    print(f"\nBlocks passing 10% threshold: relu2={blocks_pass['relu2']}, lrelu2={blocks_pass['lrelu2']}")
    print(f"VERDICT: {verdict}")

    summary["verdict"] = verdict
    summary["blocks_passing_10pct"] = blocks_pass

    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
