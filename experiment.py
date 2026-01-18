"""
Comprehensive QAT rounding mode experiment.

Varies: seeds, widths, depths, rounding modes
Logs to TensorBoard + saves decision boundaries + CSV summary
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
from aleph.models import MLP
from aleph.quantization import prepare_qat_with_rounding
from aleph.visualization import plot_decision_boundary

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
SEEDS = [42, 123, 999]
WIDTHS = [4, 8, 16, 32, 64, 128]
DEPTHS = [2, 4, 6, 8, 10]
ROUNDING_MODES = [None, "nearest", "floor", "ceil"]  # None = baseline (no quant)
EPOCHS = 3000
LR = 1e-4
TARGET_DIM = 100
LOG_EVERY = 50  # Log metrics every N epochs

# Output directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"runs/experiment_{TIMESTAMP}"
PLOT_DIR = f"plots/experiment_{TIMESTAMP}"
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


def compute_prediction_entropy(model, X):
    """Compute entropy of predictions - low entropy means predicting same class."""
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)
        preds = probs.argmax(1)
        # Proportion predicting each class
        p0 = (preds == 0).float().mean().item()
        p1 = 1 - p0
        # Entropy (0 = all same class, 1 = perfectly balanced)
        if p0 == 0 or p1 == 0:
            return 0.0
        return -(p0 * torch.log2(torch.tensor(p0)) + p1 * torch.log2(torch.tensor(p1))).item()


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def run_experiment(seed, width, depth, rounding_mode):
    """Run a single experiment configuration."""
    mode_name = rounding_mode if rounding_mode else "baseline"
    run_name = f"s{seed}_w{width}_d{depth}_{mode_name}"

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"{LOG_DIR}/{run_name}")

    # Log hyperparameters
    writer.add_hparams(
        {"seed": seed, "width": width, "depth": depth, "rounding": mode_name, "lr": LR},
        {},  # metrics will be added at the end
        run_name=run_name
    )

    # Create model
    torch.manual_seed(seed)
    model = MLP(input_size=TARGET_DIM, hidden_size=width, output_size=2,
                depth=depth, activation_fn=torch.nn.ReLU)

    if rounding_mode is not None:
        prepare_qat_with_rounding(model, rounding_mode=rounding_mode, backend="qnnpack")

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Track metrics
    best_test_acc = 0.0
    final_metrics = {}

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()

        # Compute gradient norm before optimizer step
        grad_norm = compute_gradient_norm(model)

        optimizer.step()

        # Log every N epochs
        if (epoch + 1) % LOG_EVERY == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(X_train_t).argmax(1) == y_train_t).float().mean().item()
                test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()
                test_loss = criterion(model(X_test_t), y_test_t).item()

            pred_entropy = compute_prediction_entropy(model, X_test_t)

            # TensorBoard logging
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("Gradient/norm", grad_norm, epoch)
            writer.add_scalar("Prediction/entropy", pred_entropy, epoch)

            best_test_acc = max(best_test_acc, test_acc)
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train_acc = (model(X_train_t).argmax(1) == y_train_t).float().mean().item()
        final_test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()
        final_loss = criterion(model(X_test_t), y_test_t).item()

    final_metrics = {
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
        "final_loss": final_loss,
    }

    # Save decision boundary plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_decision_boundary(model, X_test_2d, y_test,
                          f"{mode_name} (w={width}, d={depth}, s={seed})\nAcc={final_test_acc:.3f}",
                          ax, embedding=embedding)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{run_name}.png", dpi=100)
    plt.close()

    writer.close()

    return {
        "seed": seed,
        "width": width,
        "depth": depth,
        "rounding": mode_name,
        **final_metrics
    }


def main():
    # All combinations
    configs = list(itertools.product(SEEDS, WIDTHS, DEPTHS, ROUNDING_MODES))
    total_runs = len(configs)

    print(f"\n{'='*60}")
    print(f"Starting experiment: {total_runs} runs")
    print(f"Seeds: {SEEDS}")
    print(f"Widths: {WIDTHS}")
    print(f"Depths: {DEPTHS}")
    print(f"Rounding modes: {ROUNDING_MODES}")
    print(f"Epochs per run: {EPOCHS}")
    print(f"Logs: {LOG_DIR}")
    print(f"Plots: {PLOT_DIR}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    for i, (seed, width, depth, rounding_mode) in enumerate(configs):
        mode_name = rounding_mode if rounding_mode else "baseline"
        run_start = time.time()

        print(f"[{i+1}/{total_runs}] seed={seed}, width={width}, depth={depth}, rounding={mode_name}")

        result = run_experiment(seed, width, depth, rounding_mode)
        results.append(result)

        run_time = time.time() - run_start
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total_runs - i - 1)

        print(f"         -> test_acc={result['final_test_acc']:.4f}, "
              f"time={run_time:.1f}s, ETA={eta/60:.1f}min")

    # Save summary CSV
    csv_path = f"{LOG_DIR}/summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Summary saved to: {csv_path}")
    print(f"TensorBoard logs: tensorboard --logdir {LOG_DIR}")
    print(f"Decision boundary plots: {PLOT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
