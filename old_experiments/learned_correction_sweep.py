"""
Correction experiments across seeds/widths/depths/bits.

Trains a float model, evaluates naive quantization and oracle correction,
then trains a learned correction head using accumulated quantization error.
"""

import os
import csv
import time
import itertools
from datetime import datetime

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from aleph.datasets import make_spirals, embed_dataset_in_high_dimensional_space
from aleph.models import MLPWithCorrection, MLPWithLearnedCorrection
from aleph.quantization import calibrate_quantization
from aleph.visualization import (
    plot_decision_boundary,
    QuantizedWrapper,
    OracleWrapper,
    LearnedCorrectionWrapper,
)

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
SEEDS = [42, 123, 999]
WIDTHS = [4, 8, 16, 32, 64, 128]
DEPTHS = [2, 4, 6, 8, 10]
BIT_WIDTHS = [8, 4]
EPOCHS = 3000
LEARNED_EPOCHS = 1000
LR = 1e-3
LEARNED_LR = 1e-4
TARGET_DIM = 10000
LOG_EVERY = 200
CORRECTION_EVERY = 3
CORRECTION_HIDDEN = 32

# Output directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"runs/correction_experiment_{TIMESTAMP}"
PLOT_DIR = f"plots/correction_experiment_{TIMESTAMP}"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =============================================================================
# DATA SETUP (fixed across all runs)
# =============================================================================
print("Setting up data...")
X_2d, y = make_spirals(n_samples=2000, noise=0.3, n_turns=3, random_state=42)
X_high, embedding = embed_dataset_in_high_dimensional_space(X_2d, target_dim=TARGET_DIM, random_state=42)
X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)
X_train_high = embedding.transform(X_train_2d)
X_test_high = embedding.transform(X_test_2d)

X_train_t = torch.tensor(X_train_high, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_high, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

print(f"Data: {X_2d.shape} -> {X_high.shape}")
print(f"Train: {len(y_train)}, Test: {len(y_test)}")


def evaluate_float(model):
    model.eval()
    with torch.no_grad():
        acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()
    return acc


def evaluate_quant(model, scale_factors, zero_points, bits):
    model.eval()
    with torch.no_grad():
        logits = model.forward_quantized(X_test_t, scale_factors, zero_points, num_bits=bits)
        acc = (logits.argmax(1) == y_test_t).float().mean().item()
    return acc


def evaluate_oracle(model, scale_factors, zero_points, bits):
    model.eval()
    with torch.no_grad():
        logits, _ = model.forward_with_oracle_correction(
            X_test_t, scale_factors, zero_points, correct_every_n=CORRECTION_EVERY, num_bits=bits
        )
        acc = (logits.argmax(1) == y_test_t).float().mean().item()
    return acc


def evaluate_learned(model, scale_factors, zero_points, bits):
    model.eval()
    with torch.no_grad():
        logits = model.forward_quantized_with_correction(
            X_test_t, scale_factors, zero_points, num_bits=bits
        )
        acc = (logits.argmax(1) == y_test_t).float().mean().item()
    return acc


def run_experiment(seed, width, depth, bits):
    """Run a single experiment configuration."""
    run_name = f"s{seed}_w{width}_d{depth}_b{bits}"
    writer = SummaryWriter(log_dir=f"{LOG_DIR}/{run_name}")

    writer.add_hparams(
        {
            "seed": seed,
            "width": width,
            "depth": depth,
            "bits": bits,
            "lr": LR,
            "learned_lr": LEARNED_LR,
            "correction_every": CORRECTION_EVERY,
            "correction_hidden": CORRECTION_HIDDEN,
        },
        {},
        run_name=run_name,
    )

    torch.manual_seed(seed)
    model = MLPWithCorrection(
        input_size=TARGET_DIM,
        hidden_size=width,
        output_size=2,
        depth=depth,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_EVERY == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(X_train_t).argmax(1) == y_train_t).float().mean().item()
                test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            model.train()

    scale_factors, zero_points = calibrate_quantization(model, X_train_t, num_bits=bits)

    acc_float = evaluate_float(model)
    acc_quant = evaluate_quant(model, scale_factors, zero_points, bits)
    acc_oracle = evaluate_oracle(model, scale_factors, zero_points, bits)

    learned_model = MLPWithLearnedCorrection(
        input_size=TARGET_DIM,
        hidden_size=width,
        output_size=2,
        depth=depth,
        correction_every_n=CORRECTION_EVERY,
        correction_hidden=CORRECTION_HIDDEN,
    )
    learned_model.load_state_dict(model.state_dict(), strict=False)

    learned_optimizer = torch.optim.Adam(learned_model.parameters(), lr=LEARNED_LR)
    learned_model.train()
    for epoch in range(LEARNED_EPOCHS):
        learned_optimizer.zero_grad()
        logits = learned_model.forward_quantized_with_correction(
            X_train_t, scale_factors, zero_points, num_bits=bits
        )
        loss = criterion(logits, y_train_t)
        loss.backward()
        learned_optimizer.step()

        if (epoch + 1) % LOG_EVERY == 0:
            learned_model.eval()
            with torch.no_grad():
                test_acc = evaluate_learned(learned_model, scale_factors, zero_points, bits)
            writer.add_scalar("Learned/Loss", loss.item(), epoch)
            writer.add_scalar("Learned/Accuracy", test_acc, epoch)
            learned_model.train()

    acc_learned = evaluate_learned(learned_model, scale_factors, zero_points, bits)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    plot_decision_boundary(
        model, X_test_2d, y_test, f"FP32 (acc={acc_float:.3f})", axes[0], embedding=embedding
    )

    quant_wrapper = QuantizedWrapper(model, scale_factors, zero_points, bits)
    oracle_wrapper = OracleWrapper(model, scale_factors, zero_points, CORRECTION_EVERY, bits)
    learned_wrapper = LearnedCorrectionWrapper(learned_model, scale_factors, zero_points, bits)

    plot_decision_boundary(
        quant_wrapper, X_test_2d, y_test, f"INT{bits} naive (acc={acc_quant:.3f})", axes[1], embedding=embedding
    )
    plot_decision_boundary(
        oracle_wrapper,
        X_test_2d,
        y_test,
        f"INT{bits} oracle (acc={acc_oracle:.3f})",
        axes[2],
        embedding=embedding,
    )
    plot_decision_boundary(
        learned_wrapper,
        X_test_2d,
        y_test,
        f"INT{bits} learned (acc={acc_learned:.3f})",
        axes[3],
        embedding=embedding,
    )

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{run_name}.png", dpi=100)
    plt.close()

    writer.close()

    return {
        "seed": seed,
        "width": width,
        "depth": depth,
        "bits": bits,
        "acc_float": acc_float,
        "acc_quant": acc_quant,
        "acc_oracle": acc_oracle,
        "acc_learned": acc_learned,
    }


def main():
    configs = list(itertools.product(SEEDS, WIDTHS, DEPTHS, BIT_WIDTHS))
    total_runs = len(configs)

    print(f"\n{'='*60}")
    print(f"Starting correction experiment: {total_runs} runs")
    print(f"Seeds: {SEEDS}")
    print(f"Widths: {WIDTHS}")
    print(f"Depths: {DEPTHS}")
    print(f"Bit widths: {BIT_WIDTHS}")
    print(f"Epochs per run: {EPOCHS}")
    print(f"Learned epochs per run: {LEARNED_EPOCHS}")
    print(f"Logs: {LOG_DIR}")
    print(f"Plots: {PLOT_DIR}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    for i, (seed, width, depth, bits) in enumerate(configs):
        run_start = time.time()
        print(f"[{i+1}/{total_runs}] seed={seed}, width={width}, depth={depth}, bits={bits}")

        result = run_experiment(seed, width, depth, bits)
        results.append(result)

        run_time = time.time() - run_start
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total_runs - i - 1)

        print(f"         -> acc_float={result['acc_float']:.4f}, "
              f"acc_quant={result['acc_quant']:.4f}, "
              f"acc_oracle={result['acc_oracle']:.4f}, "
              f"acc_learned={result['acc_learned']:.4f}, "
              f"time={run_time:.1f}s, ETA={eta/60:.1f}min")

    csv_path = f"{LOG_DIR}/summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print("Correction experiment complete!")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Summary saved to: {csv_path}")
    print(f"TensorBoard logs: tensorboard --logdir {LOG_DIR}")
    print(f"Decision boundary plots: {PLOT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
