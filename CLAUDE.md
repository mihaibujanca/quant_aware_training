# Claude Code — Repo Guide

## Repo layout

- `aleph/` — main Python package (quantization, models, datasets, visualization)
- `aleph/qgeom/` — quantization geometry library (core, 2D/3D geometry, manifolds)
- `notebooks/` — Jupyter notebooks (paired with .py percent scripts via Jupytext)
- `experiments/` — experiment configs and results
- `configs/` — Hydra config files
- `scripts/` — shell scripts (pairing, session tooling, sweeps)
- `run_experiment.py` — main experiment entry point
- `analyze_sweeps.py` — sweep analysis

## Session context

Before starting work, read:
- `.context/SESSION_BRIEF.md` — recent commits, git status, active experiments

Available on-demand (read when needed, not at session start):
- `docs/ARCHITECTURE.md` — package structure and module roles
- `.context/REPO_MAP.md` — directory tree and public API signatures

Regenerate with: `make brief repomap`

## Project-specific rules

- **Think geometrically.** Geometry is foundational to this project. Reason about geometric objects — transforms, spaces, projections, manifolds — not just numerical quantities. Ask "what does this mean geometrically?" before reporting a result.
- **Always verify task performance before analysis.** Before running quantization analysis on a model, confirm it actually learned the task well. Analyzing quantization error on an undertrained model is meaningless.

## Notebooks

- Edit the `.py` percent script, not the `.ipynb`.
- A Claude Code hook auto-syncs `.py` → `.ipynb` on edit (no manual sync needed).
- Outputs are stripped on commit (nbstripout).
- Save plots to `plots/` (relative to notebook cwd, i.e. `notebooks/plots/`).

## Compact instructions

When compacting, preserve: current experiment state, active hypothesis, key quantization findings, notebook config values. Drop: full file reads, verbose training output, old experiment results already committed.
