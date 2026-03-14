# Repo Audit

Date: 2026-03-14

## Executive Summary

This repository currently contains two substantial research programs:

1. Quantization / correction work
2. Geometric compression work

The quantization work is still the larger and more mature codebase. The geometric compression work is newer, more focused, and has its own tests, pipeline, and experiment scripts.

The main clarity problem is not that the repo is random. The problem is that the project narrative has moved faster than the file structure:

- [`README.md`](/Users/mihai/Projects/quant_aware_training/README.md) now describes geometric compression as the main story.
- [`docs/ARCHITECTURE.md`](/Users/mihai/Projects/quant_aware_training/docs/ARCHITECTURE.md) still describes the repo as a quantization-aware training repo.
- [`aleph/__init__.py`](/Users/mihai/Projects/quant_aware_training/aleph/__init__.py) still exports the quantization stack as the package identity.
- The top-level runnable entry point is still [`run_experiment.py`](/Users/mihai/Projects/quant_aware_training/run_experiment.py), which belongs to the older quantization branch.

So the repo is not yet organized around its current thesis.

## What Is In The Repo

### 1. Quantization / correction stack

Primary code:

- [`aleph/quantization.py`](/Users/mihai/Projects/quant_aware_training/aleph/quantization.py)
- [`aleph/training.py`](/Users/mihai/Projects/quant_aware_training/aleph/training.py)
- [`aleph/models.py`](/Users/mihai/Projects/quant_aware_training/aleph/models.py)
- [`aleph/qgeom/`](/Users/mihai/Projects/quant_aware_training/aleph/qgeom)
- [`run_experiment.py`](/Users/mihai/Projects/quant_aware_training/run_experiment.py)
- [`configs/`](/Users/mihai/Projects/quant_aware_training/configs)

This is the original backbone of the repo. It includes:

- fake quantization utilities
- correction networks
- quantized MLP / autoencoder / transformer experiments
- geometry analysis of quantization error
- policy scoring for correction placement

This branch is still healthy code, not dead junk. It has tests and internal structure.

### 2. Geometric compression stack

Primary code:

- [`aleph/geometric_compression.py`](/Users/mihai/Projects/quant_aware_training/aleph/geometric_compression.py)
- [`aleph/differential.py`](/Users/mihai/Projects/quant_aware_training/aleph/differential.py)
- [`aleph/mesh_datasets.py`](/Users/mihai/Projects/quant_aware_training/aleph/mesh_datasets.py)
- [`scripts/run_shape_sweep.py`](/Users/mihai/Projects/quant_aware_training/scripts/run_shape_sweep.py)
- [`scripts/calibrate_speed_tiers.py`](/Users/mihai/Projects/quant_aware_training/scripts/calibrate_speed_tiers.py)
- [`experiments/kan_last_stand_overnight.py`](/Users/mihai/Projects/quant_aware_training/experiments/kan_last_stand_overnight.py)
- [`experiments/projected_kan_sweep.py`](/Users/mihai/Projects/quant_aware_training/experiments/projected_kan_sweep.py)
- [`experiments/jacobian_annealing_sweep.py`](/Users/mihai/Projects/quant_aware_training/experiments/jacobian_annealing_sweep.py)

This branch is coherent and reasonably self-contained:

- sample SDF data from meshes
- train a source MLP
- profile Jacobians / ReLU-boundary density
- fit a compressed model, mainly KAN variants
- compare against distilled KAN / scratch KAN / small MLP

Important caveat: the docs already say the original KAN direction largely failed on high-dimensional inputs. That means this branch is partly active pipeline and partly postmortem evidence.

### 3. Exploratory notebook layer

Main location:

- [`notebooks/`](/Users/mihai/Projects/quant_aware_training/notebooks)

This is the largest local directory in the repo checkout. It contains both `.ipynb` and paired `.py` files, covering both research branches.

The notebook layer is useful for understanding the project trajectory, but it also creates most of the “mishmash” feeling because:

- it mixes active and superseded ideas
- filenames are close to experiment names but not consistently canonical
- the directory is much larger than the package code

### 4. Archived and meta-research material

- [`old_experiments/`](/Users/mihai/Projects/quant_aware_training/old_experiments)
- [`autoresearch/`](/Users/mihai/Projects/quant_aware_training/autoresearch)
- [`animations/`](/Users/mihai/Projects/quant_aware_training/animations)
- [`docs/`](/Users/mihai/Projects/quant_aware_training/docs)

These folders are informative, but they are not all serving the same purpose:

- `old_experiments/` is a real archive
- `autoresearch/` is a research-planning layer
- `animations/` is presentation material
- `docs/` mixes durable architecture docs, experiment reports, scratch notes, and theory reviews

## What Looks Current

Most current geometric-compression code path:

1. [`README.md`](/Users/mihai/Projects/quant_aware_training/README.md)
2. [`aleph/geometric_compression.py`](/Users/mihai/Projects/quant_aware_training/aleph/geometric_compression.py)
3. [`scripts/run_shape_sweep.py`](/Users/mihai/Projects/quant_aware_training/scripts/run_shape_sweep.py)
4. [`docs/high_d_embedding.md`](/Users/mihai/Projects/quant_aware_training/docs/high_d_embedding.md)
5. [`experiments/kan_last_stand_overnight.py`](/Users/mihai/Projects/quant_aware_training/experiments/kan_last_stand_overnight.py)
6. [`tests/test_geometric_compression.py`](/Users/mihai/Projects/quant_aware_training/tests/test_geometric_compression.py)
7. [`tests/test_geometric_compression_speed.py`](/Users/mihai/Projects/quant_aware_training/tests/test_geometric_compression_speed.py)

Most current quantization-geometry code path:

1. [`run_experiment.py`](/Users/mihai/Projects/quant_aware_training/run_experiment.py)
2. [`aleph/models.py`](/Users/mihai/Projects/quant_aware_training/aleph/models.py)
3. [`aleph/quantization.py`](/Users/mihai/Projects/quant_aware_training/aleph/quantization.py)
4. [`aleph/qgeom/`](/Users/mihai/Projects/quant_aware_training/aleph/qgeom)
5. [`docs/quant_aware_training.md`](/Users/mihai/Projects/quant_aware_training/docs/quant_aware_training.md)
6. [`docs/geometry_overnight_12h_report.md`](/Users/mihai/Projects/quant_aware_training/docs/geometry_overnight_12h_report.md)
7. [`tests/test_models.py`](/Users/mihai/Projects/quant_aware_training/tests/test_models.py)
8. [`tests/test_qgeom_policy.py`](/Users/mihai/Projects/quant_aware_training/tests/test_qgeom_policy.py)

## What Looks Historical Or Superseded

- [`old_experiments/`](/Users/mihai/Projects/quant_aware_training/old_experiments) is correctly archived.
- [`docs/geometric_compression.md`](/Users/mihai/Projects/quant_aware_training/docs/geometric_compression.md) is explicitly marked obsolete.
- [`notebooks/high_d_embedding_backup.ipynb`](/Users/mihai/Projects/quant_aware_training/notebooks/high_d_embedding_backup.ipynb) and [`notebooks/high_d_embedding_backup.py`](/Users/mihai/Projects/quant_aware_training/notebooks/high_d_embedding_backup.py) are obvious backups and should not sit in the main notebook namespace long-term.
- Several docs in [`docs/`](/Users/mihai/Projects/quant_aware_training/docs) are notes, reviews, or diaries rather than stable project docs.

## Structural Mismatches

### Narrative mismatch

The main repo story says “geometric compression,” but a large part of the package and top-level interface still says “quant-aware-training.”

Examples:

- project name in [`pyproject.toml`](/Users/mihai/Projects/quant_aware_training/pyproject.toml) is `quant-aware-training`
- package docstring in [`aleph/__init__.py`](/Users/mihai/Projects/quant_aware_training/aleph/__init__.py) is quantization-first
- architecture doc in [`docs/ARCHITECTURE.md`](/Users/mihai/Projects/quant_aware_training/docs/ARCHITECTURE.md) is out of date

### Entry-point mismatch

There is no single canonical top-level “start here” path for the current repo state.

Current candidates point to different worlds:

- [`README.md`](/Users/mihai/Projects/quant_aware_training/README.md): geometric compression
- [`run_experiment.py`](/Users/mihai/Projects/quant_aware_training/run_experiment.py): quantization experiments
- [`experiments/README.md`](/Users/mihai/Projects/quant_aware_training/experiments/README.md): mostly quantization-era framing

### Docs taxonomy mismatch

[`docs/`](/Users/mihai/Projects/quant_aware_training/docs) currently mixes:

- architecture
- active briefs
- experiment reports
- theory notes
- reviews
- personal notes

That makes the folder useful historically but poor as a current control panel.

### Notebook sprawl

The notebook layer has many paired `.ipynb` and `.py` files plus generated outputs nearby. Even with jupytext, it is still doing too many jobs in one place:

- active exploration
- canonical reports
- backups
- rendered outputs

## Repo Health

The repo is not broken.

Verification:

- test command used: `./.venv/bin/python -m pytest -q`
- result: `36 passed in 3.67s`

That matters because it means the cleanup should focus on structure and truthfulness, not emergency repair.

## Recommended Mental Model

Treat the repository as:

1. A mature quantization-geometry lab
2. A newer geometric-compression branch that grew out of it
3. A large research journal around both

That framing fits the code better than pretending the old branch is gone.

## Recommended Cleanup Plan

### Phase 1: Make the repo honest

1. Update [`docs/ARCHITECTURE.md`](/Users/mihai/Projects/quant_aware_training/docs/ARCHITECTURE.md) so it describes both research branches and marks one as current focus.
2. Add a “Current focus / Archived directions / Historical context” section to [`README.md`](/Users/mihai/Projects/quant_aware_training/README.md).
3. Update [`experiments/README.md`](/Users/mihai/Projects/quant_aware_training/experiments/README.md) so it separates quantization experiments from compression experiments.
4. Update [`aleph/__init__.py`](/Users/mihai/Projects/quant_aware_training/aleph/__init__.py) docstring so package identity matches reality.

### Phase 2: Separate active from archive

1. Create an `archive/` or `research_archive/` area for:
   - obsolete docs
   - backup notebooks
   - fully superseded experimental scripts
2. Keep [`old_experiments/`](/Users/mihai/Projects/quant_aware_training/old_experiments) as-is or fold it into that archive area.
3. Move diary / note-style documents out of the main docs namespace.

### Phase 3: Give each branch a home

Recommended target structure:

- `aleph/quantization/` or keep current files but document as “quantization branch”
- `aleph/geometric_compression.py` or later `aleph/compression/`
- `docs/active/`
- `docs/reports/`
- `docs/archive/`
- `notebooks/active/`
- `notebooks/archive/`

The important part is not the exact names. The important part is isolating:

- active code
- active docs
- historical evidence
- scratch exploration

### Phase 4: Reduce top-level ambiguity

Choose one of these explicitly:

1. Keep the repository as a broad “geometry of neural networks” lab.
2. Split geometric compression into a new repository.

Given the amount of shared geometry machinery, option 1 is still defensible. But if the compression branch becomes the real main project, then keeping it inside a package still named `quant-aware-training` will keep creating friction.

## Suggested Canonical “Start Here” Paths

If the current focus is geometric compression:

- story: [`README.md`](/Users/mihai/Projects/quant_aware_training/README.md)
- implementation: [`aleph/geometric_compression.py`](/Users/mihai/Projects/quant_aware_training/aleph/geometric_compression.py)
- batch runner: [`scripts/run_shape_sweep.py`](/Users/mihai/Projects/quant_aware_training/scripts/run_shape_sweep.py)
- result interpretation: [`docs/high_d_embedding.md`](/Users/mihai/Projects/quant_aware_training/docs/high_d_embedding.md)

If revisiting the original quantization program:

- entry point: [`run_experiment.py`](/Users/mihai/Projects/quant_aware_training/run_experiment.py)
- package core: [`aleph/models.py`](/Users/mihai/Projects/quant_aware_training/aleph/models.py)
- geometry tooling: [`aleph/qgeom/__init__.py`](/Users/mihai/Projects/quant_aware_training/aleph/qgeom/__init__.py)
- research brief: [`docs/quant_aware_training.md`](/Users/mihai/Projects/quant_aware_training/docs/quant_aware_training.md)

## Bottom Line

The repo is not a random pile. It is a layered research repo with an unfinished pivot.

The immediate job is not “delete lots of stuff.” The immediate job is:

1. declare the current focus clearly
2. mark the older branch as still valid but no longer primary
3. move archival and note-like material out of the critical path
4. make the canonical entry points unambiguous
