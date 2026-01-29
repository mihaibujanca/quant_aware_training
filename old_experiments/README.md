# Old Experiments

Archived experiment scripts and analysis code. Superseded by `experiments/` and newer sweeps.

| File | Description |
|------|-------------|
| `oracle_correction_experiment.py` | Original oracle correction experiment. Runs float and quantized paths in parallel, corrects with exact error. Proved that perfect correction always recovers accuracy. |
| `learned_correction_sweep.py` | Learned correction sweep across seeds/widths/depths/bits on spiral classification. Pre-dates zero-init fix. Results in `old_runs/`. |
| `rounding_mode_sweep.py` | Floor/ceil/nearest rounding mode sweep (630 runs). Conclusion: use nearest. See `docs/rounding_exploration.md`. |
| `analyze_rounding.py` | Analysis script for rounding sweep results. |
| `rounding_analysis.ipynb` | Notebook version of rounding analysis. |
| `autoencoder_experiment.py` | Single-config autoencoder correction test on MNIST. |
| `autoencoder_sweep.py` | Autoencoder correction sweep varying architecture, bits, correction params. |
| `transformer_correction_sweep.py` | Transformer correction sweep on Shakespeare char-level LM (243 runs, ~55hrs). Results documented in `docs/transformer_correction_sweep.md`. |
