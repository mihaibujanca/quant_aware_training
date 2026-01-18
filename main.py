import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from aleph.datasets import make_spirals, embed_dataset_in_high_dimensional_space
from aleph.models import MLP
from aleph.quantization import prepare_qat_with_rounding, quantize_model
from aleph.visualization import plot_decision_boundary

# Generate 2D spiral dataset
X_2d, y = make_spirals(n_samples=2000, noise=0.3, n_turns=3, random_state=42)

# Embed into high-dimensional space (100D) - makes quantization effects more visible
target_dim = 100
X_high, embedding = embed_dataset_in_high_dimensional_space(X_2d, target_dim=target_dim, random_state=42)
print(f"Data: {X_2d.shape} -> {X_high.shape}")

# Split data (keep both 2D for visualization and high-D for training)
X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)
X_train_high = embedding.transform(X_train_2d)
X_test_high = embedding.transform(X_test_2d)

# Convert to tensors (use high-D for training)
X_train_t = torch.tensor(X_train_high, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_high, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)


def create_model(rounding_mode=None, seed=42):
    """Create a model, optionally prepared for QAT with specific rounding mode."""
    # Same seed = same initialization for fair comparison
    torch.manual_seed(seed)
    model = MLP(input_size=target_dim, hidden_size=64, output_size=2, depth=5, activation_fn=torch.nn.ReLU)
    if rounding_mode is not None:
        prepare_qat_with_rounding(model, rounding_mode=rounding_mode, backend="qnnpack")
    return model


def train_model(model, epochs=100):
    """Train the model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(X_train_t).argmax(1) == y_train_t).float().mean()
                test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean()
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
            model.train()

    return model


def evaluate_model(model, name):
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        acc = (model(X_test_t).argmax(1) == y_test_t).float().mean()
    print(f"{name}: Test Accuracy = {acc:.4f}")
    return acc.item()

NUM_EPOCHS = 1000
# Also train a baseline model WITHOUT any quantization
print("=" * 60)
print("Training BASELINE model (no quantization)...")
print("=" * 60)
model_baseline = create_model(rounding_mode=None, seed=123)
model_baseline = train_model(model_baseline, epochs=NUM_EPOCHS)
model_baseline.eval()
print()

# Train 3 models with QAT using different rounding modes DURING training
models = {}

for rounding_mode in ["nearest", "floor", "ceil"]:
    print("=" * 60)
    print(f"Training model with QAT (rounding={rounding_mode})...")
    print("=" * 60)

    model = create_model(rounding_mode=rounding_mode)
    model = train_model(model, epochs=NUM_EPOCHS)
    model.eval()
    models[rounding_mode] = model
    print()

# Evaluate all models - use QAT models directly (fake quantization is still active in eval mode)
# This ensures training and inference use the SAME rounding mode
print("=" * 60)
print("Model Accuracies (Fair comparison - same rounding at train & inference):")
print("=" * 60)
print("\nBaseline (no quantization):")
evaluate_model(model_baseline, "  Float32 (no quant)     ")

for rounding_mode in ["nearest", "floor", "ceil"]:
    print(f"\nQAT with {rounding_mode} rounding:")
    evaluate_model(models[rounding_mode], f"  Fake-quantized ({rounding_mode})")

# Plot decision boundaries - just the QAT models (fair comparison)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

plot_decision_boundary(model_baseline, X_test_2d, y_test,
                       "Baseline (no quant)", axes[0], embedding=embedding)
for i, rounding_mode in enumerate(["nearest", "floor", "ceil"]):
    plot_decision_boundary(models[rounding_mode], X_test_2d, y_test,
                           f"QAT-{rounding_mode}", axes[i+1], embedding=embedding)

plt.tight_layout()
plt.savefig("decision_boundaries.png", dpi=150)
print("\nSaved decision boundary plot to 'decision_boundaries.png'")
