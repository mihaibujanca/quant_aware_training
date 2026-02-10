# Claude Code — Repo Guide

## Repo layout

- `aleph/` — main Python package (quantization, models, datasets, visualization)
- `aleph/qgeom/` — quantization geometry library (core, 2D/3D geometry, manifolds)
- `notebooks/` — Jupyter notebooks (paired with .py percent scripts via Jupytext)
- `experiments/` — experiment configs and results
- `configs/` — Hydra config files
- `scripts/` — shell scripts (pairing, session tooling, sweeps)
- `docs/` — documentation
- `run_experiment.py` — main experiment entry point
- `analyze_sweeps.py` — sweep analysis

## Session context

Before starting work, read:
- `.context/SESSION_BRIEF.md` — recent commits, git status, active experiments
- `docs/ARCHITECTURE.md` — package structure and module roles
- `.context/REPO_MAP.md` — directory tree and public API signatures
- `.context/NEXT_STEPS.md` - at the end of each session, we have a discussion on what the next steps are. At the beginning of each session, you read this document, summarize it to me, and discuss the plan before starting any implementation.

Regenerate with: `make brief repomap`

## Rules

0. **Always use PyTorch** for numerical code unless there's a very specific reason for numpy (e.g. scipy interop, matplotlib mesh grids). All new library code, notebooks, and analysis should use `torch.Tensor`, not `np.ndarray`.
0b. **Think geometrically.** Geometry is foundational to this project. When designing experiments, writing documentation, or explaining results, always reason about what's happening in terms of geometric objects — transforms, spaces, projections, manifolds — not just numerical quantities. Ask "what does this mean geometrically?" before reporting a number.
0c. **Always verify task performance before analysis.** Before running quantization analysis on a model, confirm it actually learned the task well (e.g. check accuracy, loss, PPL against reasonable baselines). Always store performance metrics in the logs. Analyzing quantization error on a model that hasn't learned the task is meaningless — the weights are undertrained and the error dynamics tell us nothing about real-world quantization behavior.
1. **Never scan the entire repo.** Only open files I point you to or that you must edit.
2. **Always use `rg`, not `grep` to locate symbols.** Open only the relevant snippet, not whole files.
3. **Always use `fd` instead of `find` to identify files.**
4. **Output unified diffs only.** Do not paste full file contents.
5. **Do not reread or restate code you just wrote.**
6. **Prefer touching <= 3 files per change.** If more are needed, propose the file list first.
7. **Do not rescan the repo** between turns unless the task changes.

## Counterintuitive results

Never take counterintuitive results at face value. When something surprising appears (e.g. a model with less information outperforms one with more):

1. **Review code and methodology carefully.** Look for bugs, architecture limitations, unfair comparisons, and training dynamics artifacts. Most counterintuitive results are mistakes.
2. **Test with different configurations.** Vary depth, width, dataset, hyperparameters. If the result only appears in one config, it's fragile and likely an artifact.
3. **Only if confirmed across configs**, document and explain thoroughly. A genuine counterintuitive finding is valuable — but only after ruling out mundane explanations.

## Workflow

When asked to implement something:
1. Propose a minimal list of files to touch.
2. Show unified diffs (or use edit tools).
3. If tests exist, run them.

## Notebooks

- Edit the `.py` percent script, not the `.ipynb`.
- A Claude Code hook auto-syncs `.py` → `.ipynb` on edit (no manual sync needed).
- Outputs are stripped on commit (nbstripout).
- Save plots to `plots/` (relative to notebook cwd, i.e. `notebooks/plots/`). This is cheap and exploratory. `docs/figures/` is only for final, validated figures destined for reports/papers.
