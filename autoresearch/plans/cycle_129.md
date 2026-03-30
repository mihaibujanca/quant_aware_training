# Cycle 129 Plan — Item 100: Is FFN isotropy architectural or learned?

## Pre-flights
- **P1 (duplicate):** No duplicate. Only item 100 tests architectural vs learned isotropy.
- **P2 (dependency):** Item 99 (cycle 086) ran, verdict INCONCLUSIVE (ratios 0.004–0.016). Reuse its metric and blocks.
- **P3 (falsifiable):** Yes. Random-init isotropic → REFUTED. Random-init anisotropic + trained isotropic → CONFIRMED.
- **P4 (single question):** Yes — one binary question (architectural vs learned).
- **P5 (null baseline):** Random-init IS the null baseline (architecture + init, no learning).
- **P6 (GELU-specific):** Test 2 is qualitative only (see below). Test 1 is activation-agnostic.
- **P7 (power):** Same corpus (16384 tokens). 3 seeds for random-init → mean ± std.

## Approach

### Test 1: Random-init GPT-2 (critical)
- Load GPT-2 config, initialize with `GPT2LMHeadModel(config)` using **3 seeds: [42, 123, 7]**.
- Per seed: run probe corpus (4 sources × 32 seqs × 128 tokens = 16384 tokens).
- Collect post-GELU hidden activations h at blocks [7, 5, 4, 1].
- Compute `h_isotropy_ratio = λ_512 / λ_1` from covariance eigenspectrum.
- Report per-seed ratios + **mean ± std** across seeds.
- Compare mean random-init ratio to cycle 086 trained values: block 1 = 0.0043, block 7 = 0.0151.

### Test 2: Post-hoc ReLU swap (qualitative only)
- Load trained GPT-2. Replace `gelu` with `relu` in block 7 only.
- Measure same isotropy metric. Report as **qualitative observation only** — excluded from decision logic.
- Caveat: ReLU hard zeros mechanically kill ~50% of activation variance, making isotropy ratio incomparable to GELU baseline. Direct ratio comparison is unreliable.

### Decision logic (Test 1 only)
- Mean random-init ratio ≈ trained ratio (within 2×) → **REFUTED** (isotropy is architectural)
- Mean random-init ratio >> trained ratio (>10×) → **CONFIRMED** (isotropy is learned)
- Mean random-init ratio 2–10× trained → **INCONCLUSIVE**

## Experiment file
`experiments/ffn_isotropy_architectural_gpt2.py`

Reuse from cycle 086: `compute_eigenspectrum`, `build_probe_corpus`, `load_nanochat_teacher`.
Add: `GPT2Config` + `GPT2LMHeadModel` for random-init (loop over 3 seeds); GELU→ReLU hook.

## Artifacts
- `autoresearch/results/cycle_129_isotropy_architectural.json`: per-seed + mean±std isotropy ratios for random-init; trained baseline; ReLU-swap (qualitative).
- One comparison plot: bar chart of isotropy ratios across conditions × 4 blocks (error bars for random-init).

## Runtime
~15 min (3× random-init forward passes + 1 trained + 1 ReLU-swap, CPU only).
