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
- `.context/REPO_MAP.md` — directory tree and public API signatures
- `docs/ARCHITECTURE.md` — package structure and module roles

Regenerate with: `make brief repomap`

## Rules

1. **Never scan the entire repo.** Only open files I point you to or that you must edit.
2. **Use `rg` / `grep` to locate symbols.** Open only the relevant snippet, not whole files.
3. **Output unified diffs only.** Do not paste full file contents.
4. **Do not reread or restate code you just wrote.**
5. **Prefer touching <= 3 files per change.** If more are needed, propose the file list first.
6. **Do not rescan the repo** between turns unless the task changes.

## Workflow

When asked to implement something:
1. Propose a minimal list of files to touch.
2. Show unified diffs (or use edit tools).
3. If tests exist, run them.

## Notebooks

- Edit the `.py` percent script, not the `.ipynb`.
- Sync: `jupytext --sync notebooks/<name>.py`
- Outputs are stripped on commit (nbstripout).
