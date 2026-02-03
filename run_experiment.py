"""
Unified experiment runner for quantization correction experiments.

Usage:
    # Single run (autoencoder on MNIST, 4-bit)
    python run_experiment.py experiment=autoencoder bits=4

    # Single run (transformer on Shakespeare)
    python run_experiment.py experiment=transformer bits=4

    # Single run (classification on spirals)
    python run_experiment.py experiment=classification bits=8

    # Sweep over seeds and bit widths
    python run_experiment.py -m seed=42,123,999 bits=2,4,8

    # Override model architecture
    python run_experiment.py experiment=autoencoder model.hidden_sizes=[512,256,128]
"""

import logging
import os

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from aleph.datasets import load_mnist_flat, load_shakespeare, make_spirals, embed_dataset_in_high_dimensional_space
from aleph.models import AutoencoderWithCorrection, MLPWithCorrection, MLPWithLearnedCorrection, TransformerWithCorrection
from aleph.qgeom.transformer_analysis import collect_transformer_layer_reports

from aleph.quantization import calibrate_model

log = logging.getLogger(__name__)


def get_device(cfg):
    """Get the device to use for training."""
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def should_skip_correction(correction_every, model_depth, model_name):
    if correction_every > model_depth:
        log.warning(
            f"{model_name}: correction_every ({correction_every}) > model depth ({model_depth}); "
            "skipping correction."
        )
        return True
    return False


# =============================================================================
# Autoencoder experiment
# =============================================================================


def run_autoencoder(cfg: DictConfig):
    """Run autoencoder reconstruction experiment."""
    device = get_device(cfg)
    torch.manual_seed(cfg.seed)

    # Data
    train_loader, test_loader = load_mnist_flat(batch_size=cfg.batch_size)
    test_X, _ = next(iter(test_loader))
    test_X = test_X.to(device)

    # Model
    model = AutoencoderWithCorrection(
        input_size=784,
        hidden_sizes=list(cfg.model.hidden_sizes),
        latent_size=cfg.model.latent_size,
        correction_every_n=cfg.correction_every,
        correction_hidden=cfg.correction_hidden,
    ).to(device)

    log.info(f"Model: hidden={cfg.model.hidden_sizes}, latent={cfg.model.latent_size}")
    log.info(f"Correction: every {cfg.correction_every} layers, hidden={cfg.correction_hidden}")
    log.info(f"Correction layers: {len(model.correction_layers)}")

    skip_correction = should_skip_correction(
        cfg.correction_every, len(cfg.model.hidden_sizes), "Autoencoder"
    )

    # Train float model
    log.info("Training float model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    for epoch in range(cfg.epochs):
        model.train()
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()

    # Calibrate
    X_calib, _ = next(iter(train_loader))
    X_calib = X_calib.to(device)
    scale_factors, zero_points = calibrate_model(model, X_calib, num_bits=cfg.bits)

    # Evaluate float and quantized
    model.eval()
    with torch.no_grad():
        mse_float = F.mse_loss(model(test_X), test_X).item()
        mse_quant = F.mse_loss(
            model.forward_quantized(test_X, scale_factors, zero_points, num_bits=cfg.bits),
            test_X
        ).item()

    log.info(f"Float MSE: {mse_float:.6f}")
    log.info(f"Quantized MSE: {mse_quant:.6f}")

    # Train correction layers with distillation (match float model outputs)
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    if not skip_correction and len(list(model.correction_layers.parameters())) > 0:
        corr_optimizer = torch.optim.Adam(model.correction_layers.parameters(), lr=cfg.learned_lr)

        log.info("Training correction layers (distillation)...")
        for epoch in range(cfg.learned_epochs):
            model.train()
            for X_batch, _ in train_loader:
                X_batch = X_batch.to(device)

                # Get float model output (teacher)
                with torch.no_grad():
                    float_output = model(X_batch)

                # Get corrected output (student)
                corrected_output = model.forward_quantized_with_correction(
                    X_batch, scale_factors, zero_points, num_bits=cfg.bits
                )

                # Distillation loss: match float output
                corr_optimizer.zero_grad()
                loss = F.mse_loss(corrected_output, float_output)
                loss.backward()
                corr_optimizer.step()

    # Final evaluation
    if skip_correction:
        mse_corrected = mse_quant
    else:
        model.eval()
        with torch.no_grad():
            mse_corrected = F.mse_loss(
                model.forward_quantized_with_correction(test_X, scale_factors, zero_points, num_bits=cfg.bits),
                test_X
            ).item()

    quant_gap = mse_quant - mse_float
    recovery = (mse_quant - mse_corrected) / quant_gap * 100 if quant_gap > 0 else 0

    log.info(f"Corrected MSE: {mse_corrected:.6f}")
    log.info(f"Recovery: {recovery:.1f}%")

    return {
        "mse_float": mse_float,
        "mse_quant": mse_quant,
        "mse_corrected": mse_corrected,
        "recovery_pct": recovery,
    }


# =============================================================================
# Transformer experiment
# =============================================================================


def run_transformer(cfg: DictConfig):
    """Run transformer language modeling experiment."""
    device = get_device(cfg)
    torch.manual_seed(cfg.seed)

    # Data
    train_X, train_Y, test_X, test_Y, vocab_size, _, _ = load_shakespeare(seq_len=cfg.data.seq_len)
    test_X, test_Y = test_X.to(device), test_Y.to(device)

    log.info(f"Vocab size: {vocab_size}")
    log.info(f"Train: {len(train_X)}, Test: {len(test_X)}")

    # Model
    model = TransformerWithCorrection(
        vocab_size=vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        correction_every_n=cfg.correction_every,
        correction_hidden=cfg.correction_hidden,
    ).to(device)

    log.info(f"Model: d={cfg.model.d_model}, h={cfg.model.n_heads}, L={cfg.model.n_layers}")
    log.info(f"Correction layers: {len(model.correction_layers)}")

    skip_correction = should_skip_correction(
        cfg.correction_every, cfg.model.n_layers * 2, "Transformer"
    )

    # Train float model
    log.info("Training float model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    n_batches = len(train_X) // cfg.batch_size

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(train_X))
        train_X_shuffled = train_X[perm]
        train_Y_shuffled = train_Y[perm]

        for i in range(n_batches):
            batch_X = train_X_shuffled[i*cfg.batch_size:(i+1)*cfg.batch_size].to(device)
            batch_Y = train_Y_shuffled[i*cfg.batch_size:(i+1)*cfg.batch_size].to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch_Y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Calibrate
    X_calib = train_X[:cfg.batch_size].to(device)
    scale_factors, zero_points = calibrate_model(model, X_calib, num_bits=cfg.bits)
    geometry_report = None

    if cfg.get("geometry_report", False):
        report_batch = test_X[: min(len(test_X), cfg.batch_size)].to(device)
        geometry_report = collect_transformer_layer_reports(
            model,
            report_batch,
            scale_factors,
            zero_points,
            num_bits=cfg.bits,
            task_name="transformer_shakespeare",
        )
        with open("geometry_report.json", "w") as f:
            import json

            json.dump(geometry_report.to_dict(), f, indent=2)

    # Evaluate
    model.eval()
    with torch.no_grad():
        loss_float = F.cross_entropy(
            model(test_X).reshape(-1, vocab_size),
            test_Y.reshape(-1)
        ).item()
        loss_quant = F.cross_entropy(
            model.forward_quantized(test_X, scale_factors, zero_points, num_bits=cfg.bits).reshape(-1, vocab_size),
            test_Y.reshape(-1)
        ).item()

    log.info(f"Float loss: {loss_float:.4f}, PPL: {torch.exp(torch.tensor(loss_float)).item():.1f}")
    log.info(f"Quantized loss: {loss_quant:.4f}, PPL: {torch.exp(torch.tensor(loss_quant)).item():.1f}")

    # Train correction layers with distillation (match float model outputs)
    for name, param in model.named_parameters():
        if 'correction_layers' not in name:
            param.requires_grad = False

    if not skip_correction and len(list(model.correction_layers.parameters())) > 0:
        corr_optimizer = torch.optim.Adam(model.correction_layers.parameters(), lr=cfg.learned_lr)

        log.info("Training correction layers (distillation)...")
        for epoch in range(cfg.learned_epochs):
            model.train()
            perm = torch.randperm(len(train_X))
            train_X_shuffled = train_X[perm]

            for i in range(n_batches):
                batch_X = train_X_shuffled[i*cfg.batch_size:(i+1)*cfg.batch_size].to(device)

                # Get float model output (teacher)
                with torch.no_grad():
                    float_logits = model(batch_X)

                # Get corrected output (student)
                corrected_logits = model.forward_quantized_with_correction(
                    batch_X, scale_factors, zero_points, num_bits=cfg.bits
                )

                # Distillation loss: match float logits
                corr_optimizer.zero_grad()
                loss = F.mse_loss(corrected_logits, float_logits)
                loss.backward()
                corr_optimizer.step()

    # Final evaluation
    if skip_correction:
        loss_corrected = loss_quant
    else:
        model.eval()
        with torch.no_grad():
            loss_corrected = F.cross_entropy(
                model.forward_quantized_with_correction(test_X, scale_factors, zero_points, num_bits=cfg.bits).reshape(-1, vocab_size),
                test_Y.reshape(-1)
            ).item()

    quant_gap = loss_quant - loss_float
    recovery = (loss_quant - loss_corrected) / quant_gap * 100 if quant_gap > 0 else 0

    log.info(f"Corrected loss: {loss_corrected:.4f}, PPL: {torch.exp(torch.tensor(loss_corrected)).item():.1f}")
    log.info(f"Recovery: {recovery:.1f}%")

    return {
        "loss_float": loss_float,
        "loss_quant": loss_quant,
        "loss_corrected": loss_corrected,
        "ppl_float": torch.exp(torch.tensor(loss_float)).item(),
        "ppl_quant": torch.exp(torch.tensor(loss_quant)).item(),
        "ppl_corrected": torch.exp(torch.tensor(loss_corrected)).item(),
        "recovery_pct": recovery,
        "geometry_report_path": "geometry_report.json" if geometry_report is not None else None,
    }


# =============================================================================
# Classification experiment
# =============================================================================


def run_classification(cfg: DictConfig):
    """Run classification experiment on spiral dataset."""
    device = get_device(cfg)
    torch.manual_seed(cfg.seed)

    # Data
    X_2d, y = make_spirals(
        n_samples=cfg.data.n_samples,
        noise=cfg.data.noise,
        n_turns=cfg.data.n_turns,
        random_state=cfg.seed
    )
    X_high, embedding = embed_dataset_in_high_dimensional_space(
        X_2d, target_dim=cfg.data.target_dim, random_state=cfg.seed
    )

    X_train_2d, X_test_2d, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.2, random_state=cfg.seed
    )
    X_train = torch.tensor(embedding.transform(X_train_2d), dtype=torch.float32).to(device)
    X_test = torch.tensor(embedding.transform(X_test_2d), dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    log.info(f"Data: {X_2d.shape} -> {X_high.shape}")
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Model (for oracle correction)
    model = MLPWithCorrection(
        input_size=cfg.data.target_dim,
        hidden_size=cfg.model.hidden_size,
        output_size=2,
        depth=cfg.model.depth
    ).to(device)

    log.info(f"Model: depth={cfg.model.depth}, width={cfg.model.hidden_size}")

    skip_correction = should_skip_correction(
        cfg.correction_every, cfg.model.depth, "MLP"
    )

    # Train float model
    log.info("Training float model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Calibrate
    from aleph.quantization import calibrate_quantization
    scale_factors, zero_points = calibrate_quantization(model, X_train, num_bits=cfg.bits)

    # Evaluate
    model.eval()
    with torch.no_grad():
        acc_float = (model(X_test).argmax(1) == y_test).float().mean().item()
        acc_quant = (model.forward_quantized(X_test, scale_factors, zero_points, num_bits=cfg.bits).argmax(1) == y_test).float().mean().item()
        if skip_correction:
            acc_oracle = acc_quant
        else:
            acc_oracle, _ = model.forward_with_oracle_correction(
                X_test, scale_factors, zero_points, correct_every_n=cfg.correction_every, num_bits=cfg.bits
            )
            acc_oracle = (acc_oracle.argmax(1) == y_test).float().mean().item()

    log.info(f"Float accuracy: {acc_float:.4f}")
    log.info(f"Quantized accuracy: {acc_quant:.4f}")
    log.info(f"Oracle accuracy: {acc_oracle:.4f}")

    # Train learned correction
    if skip_correction:
        acc_learned = acc_quant
    else:
        learned_model = MLPWithLearnedCorrection(
            input_size=cfg.data.target_dim,
            hidden_size=cfg.model.hidden_size,
            output_size=2,
            depth=cfg.model.depth,
            correction_every_n=cfg.correction_every,
            correction_hidden=cfg.correction_hidden,
        ).to(device)
        learned_model.load_state_dict(model.state_dict(), strict=False)

        # Freeze backbone, only train correction layers
        for name, param in learned_model.named_parameters():
            if 'correction_layers' not in name:
                param.requires_grad = False

        log.info("Training learned correction (distillation)...")
        learned_optimizer = torch.optim.Adam(learned_model.correction_layers.parameters(), lr=cfg.learned_lr)

        # Get float model output (teacher) - only need to compute once for full-batch
        model.eval()
        with torch.no_grad():
            float_logits = model(X_train)

        for epoch in range(cfg.learned_epochs):
            learned_model.train()

            # Get corrected output (student)
            corrected_logits = learned_model.forward_quantized_with_correction(
                X_train, scale_factors, zero_points, num_bits=cfg.bits
            )

            # Distillation loss: match float logits
            learned_optimizer.zero_grad()
            loss = F.mse_loss(corrected_logits, float_logits)
            loss.backward()
            learned_optimizer.step()

        # Final evaluation
        learned_model.eval()
        with torch.no_grad():
            acc_learned = (learned_model.forward_quantized_with_correction(
                X_test, scale_factors, zero_points, num_bits=cfg.bits
            ).argmax(1) == y_test).float().mean().item()

    quant_gap = acc_float - acc_quant
    recovery = (acc_learned - acc_quant) / quant_gap * 100 if quant_gap > 0 else 0

    log.info(f"Learned accuracy: {acc_learned:.4f}")
    log.info(f"Recovery: {recovery:.1f}%")

    return {
        "acc_float": acc_float,
        "acc_quant": acc_quant,
        "acc_oracle": acc_oracle,
        "acc_learned": acc_learned,
        "recovery_pct": recovery,
    }


# =============================================================================
# Main entry point
# =============================================================================


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = get_device(cfg)
    log.info(f"Device: {device}")
    log.info(f"Seed: {cfg.seed}")
    log.info(f"Bits: {cfg.bits}")

    # Run the appropriate experiment
    exp_type = cfg.experiment.type
    if exp_type == "autoencoder":
        results = run_autoencoder(cfg)
    elif exp_type == "transformer":
        results = run_transformer(cfg)
    elif exp_type == "classification":
        results = run_classification(cfg)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    if results:
        log.info(f"Results: {results}")

        # Save results
        import json
        with open("results.json", "w") as f:
            json.dump({
                "config": OmegaConf.to_container(cfg, resolve=True),
                "results": results
            }, f, indent=2)

    return results.get("recovery_pct", 0) if results else 0


if __name__ == "__main__":
    main()
