#!/bin/bash
# Overnight sweep experiments for quantization correction
# Run with: ./run_sweeps.sh 2>&1 | tee sweep_log.txt
#
# Priority order: classification (fastest) -> autoencoder -> transformer (slowest)

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="runs/sweep_${TIMESTAMP}"
SEED=42

echo "============================================================"
echo "Starting overnight sweeps at $(date)"
echo "Results will be saved to: ${BASE_DIR}"
echo "Seed: ${SEED}"
echo "============================================================"

# ===========================================
# CLASSIFICATION (fastest, ~2 hours)
# ===========================================
# Architecture: hidden_size × depth
# Quantization: bits
# Correction: correction_every × correction_hidden

echo ""
echo "=== CLASSIFICATION (seed=${SEED}) ==="
echo "Started at $(date)"

# Baseline (no correction)
uv run python run_experiment.py -m \
    experiment=classification \
    bits=2,4 \
    correction_every=999 \
    correction_hidden=0 \
    model.hidden_size=32,64,128 \
    model.depth=4,6,8 \
    seed=${SEED} \
    hydra.sweep.dir=${BASE_DIR}/classification_baseline

# With correction
uv run python run_experiment.py -m \
    experiment=classification \
    bits=2,4 \
    correction_every=1,2,4 \
    correction_hidden=0,32 \
    model.hidden_size=32,64,128 \
    model.depth=4,6,8 \
    seed=${SEED} \
    hydra.sweep.dir=${BASE_DIR}/classification_correction

echo "Classification completed at $(date)"

# ===========================================
# AUTOENCODER (~6 hours)
# ===========================================
# Architecture: hidden_sizes × latent_size
# Quantization: bits
# Correction: correction_every × correction_hidden

echo ""
echo "=== AUTOENCODER (seed=${SEED}) ==="
echo "Started at $(date)"

# Baseline (no correction)
uv run python run_experiment.py -m \
    experiment=autoencoder \
    bits=2,4 \
    correction_every=999 \
    correction_hidden=0 \
    model.hidden_sizes="[256,128]","[512,256,128]","[128,64]" \
    model.latent_size=16,32,64 \
    seed=${SEED} \
    hydra.sweep.dir=${BASE_DIR}/autoencoder_baseline

# With correction
uv run python run_experiment.py -m \
    experiment=autoencoder \
    bits=2,4 \
    correction_every=1,2,4 \
    correction_hidden=0,32 \
    model.hidden_sizes="[256,128]","[512,256,128]","[128,64]" \
    model.latent_size=16,32,64 \
    seed=${SEED} \
    hydra.sweep.dir=${BASE_DIR}/autoencoder_correction

echo "Autoencoder completed at $(date)"

# ===========================================
# TRANSFORMER (slowest, ~16 hours)
# ===========================================
# Architecture: d_model × n_layers (d_ff = 4 × d_model)
# Quantization: bits
# Correction: correction_every × correction_hidden

echo ""
echo "=== TRANSFORMER (seed=${SEED}) ==="
echo "Started at $(date)"

# Baseline and correction for each d_model (d_ff tied to d_model)
for D_MODEL in 64 128 256; do
    D_FF=$((D_MODEL * 4))
    echo "  d_model=${D_MODEL}, d_ff=${D_FF}"

    # Baseline (no correction)
    uv run python run_experiment.py -m \
        experiment=transformer \
        bits=2,4 \
        correction_every=999 \
        correction_hidden=0 \
        model.d_model=${D_MODEL} \
        model.n_layers=2,4,6 \
        model.n_heads=4 \
        model.d_ff=${D_FF} \
        seed=${SEED} \
        hydra.sweep.dir=${BASE_DIR}/transformer_baseline_d${D_MODEL}

    # With correction
    uv run python run_experiment.py -m \
        experiment=transformer \
        bits=2,4 \
        correction_every=1,2,4 \
        correction_hidden=0,32 \
        model.d_model=${D_MODEL} \
        model.n_layers=2,4,6 \
        model.n_heads=4 \
        model.d_ff=${D_FF} \
        seed=${SEED} \
        hydra.sweep.dir=${BASE_DIR}/transformer_correction_d${D_MODEL}
done

echo "Transformer completed at $(date)"

echo ""
echo "============================================================"
echo "All sweeps completed at $(date)"
echo "============================================================"
echo ""
echo "To aggregate results:"
echo "  python aggregate_results.py ${BASE_DIR}/*"
