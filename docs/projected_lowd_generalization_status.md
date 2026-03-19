# Projected Low-D Generalization Status

Date: 2026-03-17

## Goal

Check whether the current projected geometric compression branch is only a `2D`
demo or whether it survives when the intrinsic dimensionality increases.

The method tested here is the generic projected local-RBF hybrid:

- train ambient ReLU teacher on embedded inputs
- recover the intrinsic subspace with `manifold_pca`
- fit a projected low-dimensional student with:
  - a global cubic polynomial backbone
  - local RBF atoms on teacher-identified hard regions
- compare against matched-budget `MLP` controls

All runs below use:

- `3` seeds: `42, 43, 44`
- long budget: `1000` source epochs and `1000` student epochs
- observed dimensions: `100`, `1000`
- complexity mode: `projected_field`

Artifacts:

- [projected_local_rbf_hybrid_rough_spirals_3seed_long.json](/Users/mihai/Projects/quant_aware_training/runs/projected_local_rbf_hybrid_rough_spirals_3seed_long.json)
- [projected_local_rbf_hybrid_rough_hypershell_3d_3seed_long.json](/Users/mihai/Projects/quant_aware_training/runs/projected_local_rbf_hybrid_rough_hypershell_3d_3seed_long.json)
- [projected_local_rbf_hybrid_rough_hypershell_5d_3seed_long.json](/Users/mihai/Projects/quant_aware_training/runs/projected_local_rbf_hybrid_rough_hypershell_5d_3seed_long.json)

## Benchmarks

- `rough_spirals`: intrinsic `2D`
- `rough_hypershell_3d`: intrinsic `3D`
- `rough_hypershell_5d`: intrinsic `5D`

All are embedded into `100D` and `1000D` observed input space with a fixed affine
embedding.

## Projection quality

For all three benchmarks, the recovered projected subspace aligns essentially
perfectly with the true latent subspace.

| Dataset | Observed dim | Mean projection alignment |
|---|---:|---:|
| `rough_spirals` | `100` | `1.0000` |
| `rough_spirals` | `1000` | `1.0000` |
| `rough_hypershell_3d` | `100` | `1.0000` |
| `rough_hypershell_3d` | `1000` | `1.0000` |
| `rough_hypershell_5d` | `100` | `1.0000` |
| `rough_hypershell_5d` | `1000` | `1.0000` |

This means the remaining differences are not coming from projection failure.

## Main table

### `rough_spirals` (intrinsic `2D`)

| Observed dim | Source | Ambient small `MLP` | Projected same-budget `MLP` | Projected global cubic | Hybrid local RBF |
|---|---:|---:|---:|---:|---:|
| `100` | `0.9174` | `0.6972` | `0.5069` | `0.5333` | `0.7410` |
| `1000` | `0.8833` | `0.6521` | `0.5111` | `0.5389` | `0.6632` |

Hybrid budget:

- `382` trainable params
- `754` stored numbers

Interpretation:

- The hybrid clearly beats the matched-budget projected `MLP`.
- It also beats the simple projected global cubic baseline.
- This is still the cleanest case where the local high-curvature modules matter.

### `rough_hypershell_3d` (intrinsic `3D`)

| Observed dim | Source | Ambient small `MLP` | Projected same-budget `MLP` | Projected global cubic | Hybrid local RBF |
|---|---:|---:|---:|---:|---:|
| `100` | `0.9611` | `0.9479` | `0.6840` | `0.8618` | `0.8514` |
| `1000` | `0.9563` | `0.9431` | `0.8785` | `0.8875` | `0.8833` |

Hybrid budget:

- `384` trainable params
- `748` stored numbers

Interpretation:

- The hybrid still beats the tiny matched-budget ambient control comfortably.
- But the story changes relative to `2D`.
- The local RBF layer is no longer the main win:
  - it loses to the much larger ambient small `MLP`
  - it is approximately tied with, or slightly below, the projected global cubic
    baseline
- In `1000D`, even the same-budget projected `MLP` becomes strong once the latent
  task is smoother and higher-dimensional.

### `rough_hypershell_5d` (intrinsic `5D`)

| Observed dim | Source | Ambient small `MLP` | Projected same-budget `MLP` | Projected global cubic | Hybrid local RBF |
|---|---:|---:|---:|---:|---:|
| `100` | `0.9451` | `0.9243` | `0.6264` | `0.8708` | `0.8639` |
| `1000` | `0.9368` | `0.8993` | `0.5826` | `0.9042` | `0.9014` |

Hybrid budget:

- `380` trainable params
- `704` stored numbers

Interpretation:

- Same pattern as `3D`, even more clearly.
- The geometric hybrid continues to crush the tiny matched-budget projected `MLP`.
- But the projected global cubic already explains almost all of the value.
- The local RBF correction is effectively neutral relative to the global cubic
  baseline.

## Cross-dimensional conclusion

The current branch **does** generalize beyond intrinsic `2D`, but not in the most
interesting way.

Two statements are true at once:

1. **Positive result**
   - Across `2D`, `3D`, and `5D`, the projected geometric student consistently beats
     a tiny matched-budget projected `MLP`.
   - So the method is not just exploiting an accidental `2D` artifact.

2. **More important limitation**
   - As intrinsic dimension rises from `2D` to `3D` and `5D`, the advantage no
     longer comes from the local high-curvature replacement.
   - A simple projected global cubic model becomes very strong and captures most of
     the benefit.
   - The local RBF layer is only clearly valuable in the `2D` rough-spirals case.

So the current method generalizes in the weak sense:

- "project to the true low-dimensional chart and fit a compact model there" works

But it does **not** yet generalize in the strong sense we actually care about:

- "replace dense local ReLU partitions in hard regions with a better local
  representation" remains clearly demonstrated only in the `2D` case

## What this means for the next iteration

The next question is no longer:

> can the projected geometric branch work above intrinsic `2D`?

That answer is yes.

The next real question is:

> can we find `3D` and `5D` tasks where a low-order global projected model is not
> already enough, so that local geometric replacement is forced to show a benefit?

That implies two concrete next moves:

1. **Benchmark redesign**
   - Build intrinsically `3D` and `5D` tasks with localized hard regions that a
     global cubic cannot absorb.
   - The current hypershells are too smooth globally once projected.

2. **Representation redesign**
   - Move from isotropic local RBFs to a more structured local model:
     anisotropic kernels, boundary-aligned patches, or low-rank local polynomial
     charts.
   - The local module needs to add something that the global projected polynomial
     cannot already express.

## Current verdict

- `2D`: genuine local-geometric win
- `3D`: projected geometry still helps, but local-RBF correction is not the key
- `5D`: same as `3D`, more strongly

This is a useful and nontrivial result:

- the geometric branch is not confined to intrinsic `2D`
- but the current local replacement mechanism is still too weak to claim a broad
  high-curvature compression story beyond the `2D` case
