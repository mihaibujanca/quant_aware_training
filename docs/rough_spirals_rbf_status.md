# Rough-Spirals RBF Status

Date: 2026-03-17

## Scope

This note captures the current state of the high-D geometric compression branch on
`rough_spirals`, using:

- source ReLU teacher in ambient `100D` or `1000D`
- fixed extracted `2D` projection
- projected geometric student with a global cubic polynomial and local RBF atoms
- matched-budget `MLP` controls

The relevant run artifacts are:

- [projection_to_ground_truth_check.json](/Users/mihai/Projects/quant_aware_training/runs/rough_spirals_projection_to_ground_truth_check.json)
- [projection_distortion_check.json](/Users/mihai/Projects/quant_aware_training/runs/rough_spirals_projection_distortion_check.json)
- [stored_budget_mlp_compare_5000steps.json](/Users/mihai/Projects/quant_aware_training/runs/rough_spirals_local_rbf_hybrid_384q75_3seed_5000step_stored_budget_mlp_compare.json)

## Result 1: The extracted `2D` is basically the true latent plane

The phrase "fixed projected `2D`" means:

1. fit the source teacher in ambient space
2. extract a rank-2 projection `P`
3. freeze `P`
4. train the student only on `z = (x - mean) @ P`

The important check was whether this recovered `2D` is actually close to the true
latent `X_low`. It is.

After fitting the best affine map from extracted `z` back to `X_low`, both
`manifold_pca` and `jacobian_svd` recover the latent `2D` almost perfectly.

| Observed dim | Projection | Mean subspace alignment | Mean test `R^2` to true `X_low` | Mean test RMSE |
|---|---|---:|---:|---:|
| `100D` | `manifold_pca` | `1.000` | `1.000000000000` | `6.35e-08` |
| `100D` | `jacobian_svd` | `0.383` | `1.000000000000` | `1.49e-07` |
| `1000D` | `manifold_pca` | `1.000` | `1.000000000000` | `1.83e-07` |
| `1000D` | `jacobian_svd` | `0.490` | `1.000000000000` | `2.06e-07` |

Interpretation:

- The extracted `2D` is not wrong in the sense of losing the latent structure.
- The right statement is: it recovers the latent plane up to an affine transform.

## Result 2: `manifold_pca` wins because the projected geometry is well-conditioned

The real difference between `manifold_pca` and `jacobian_svd` is not recoverability.
It is distortion.

We measured the singular values of the induced map from true latent coordinates into
the extracted `2D`, and the associated condition number.

| Observed dim | Projection | Mean singular values | Mean condition number |
|---|---|---|---:|
| `100D` | `manifold_pca` | `(7.59, 6.43)` | `1.18` |
| `100D` | `jacobian_svd` | `(4.64, 0.63)` | `8.00` |
| `1000D` | `manifold_pca` | `(22.59, 21.93)` | `1.03` |
| `1000D` | `jacobian_svd` | `(19.70, 2.02)` | `10.14` |

Interpretation:

- `manifold_pca` produces a nearly isotropic latent chart.
- `jacobian_svd` produces a heavily sheared / squashed chart.
- This matters because the current local geometry model uses isotropic RBF atoms and
  Euclidean neighborhoods in projected space.

So the current evidence is:

- projection recovery itself is fine
- the metric structure of the projected chart matters a lot

## Result 3: At tiny budget, the geometric model clearly beats matched-budget `MLP`s

The strongest apples-to-apples test so far is the `5000`-step stored-budget
comparison.

Hybrid budget:

- `382` trainable parameters
- `754` stored numbers

For the control, we gave the plain `MLP` the same fixed projected `2D` input and
roughly the same total stored budget as learned parameters.

### Stored-budget comparison (`5000` steps, best checkpoint)

| Regime | Model | Params | Stored # | Mean test acc |
|---|---|---:|---:|---:|
| `100D` | Source MLP (`96x6`) | `56,353` | `56,353` | `0.9250` |
| `100D` | Ambient small MLP (`32x4`) | `6,433` | `6,433` | `0.8292` |
| `100D` | Hybrid RBF | `382` | `754` | `0.7174` |
| `100D` | Projected stored-budget MLP | `753` | `753` | `0.5250` |
| `100D` | Ambient near-budget MLP | `739` | `739` | `0.5632` |
| `1000D` | Source MLP (`96x6`) | `142,753` | `142,753` | `0.9021` |
| `1000D` | Ambient small MLP (`32x4`) | `35,233` | `35,233` | `0.6701` |
| `1000D` | Hybrid RBF | `382` | `754` | `0.6521` |
| `1000D` | Projected stored-budget MLP | `753` | `753` | `0.5569` |
| `1000D` | Ambient near-budget MLP | `1,003` | `1,003` | `0.5125` |

### Delta vs matched-budget `MLP`

| Regime | Hybrid minus projected stored-budget `MLP` | Hybrid minus ambient near-budget `MLP` |
|---|---:|---:|
| `100D` | `+0.1924` | `+0.1542` |
| `1000D` | `+0.0951` | `+0.1396` |

Interpretation:

- At tiny budget, the geometric model is doing real work.
- This is not just "project to `2D` and train any small student."
- A same-representation, same-budget `MLP` is much worse.

## Result 4: The current method is still not good enough on the larger practical Pareto point

The same table also shows what is missing.

- In `100D`, the much larger ambient small `32x4` MLP still beats the hybrid by
  about `11.2` accuracy points.
- In `1000D`, the gap is much smaller: about `1.8` points.
- The source still sits far above everything, so there is substantial remaining
  headroom.

So the honest read is:

- strong positive result at tiny budget
- not yet a win against the practical larger `32x4` small MLP in `100D`
- much closer in `1000D`

## Interpretation of the `100D` vs `1000D` difference

If the method depended only on the recovered latent `2D`, then `100D` and `1000D`
should be almost identical after projection. They are not, because the current
pipeline still depends on the ambient teacher after projection.

The current hybrid is built from:

1. ambient teacher logits
2. teacher-derived complexity scores
3. teacher-derived atom placement
4. teacher-guided initialization and weighted training

The key complexity components currently include:

- projected Jacobian variation
- activation churn from ambient ReLU codes
- projected boundary density
- inverse teacher margin

That means ambient dimension still leaks into the student through the teacher's
internal partitioning and margin structure, even when the projected coordinates are
essentially the same latent chart.

This is a weakness of the current method, not a mystery in the benchmark.

## What We Are Doing Next

The next branch should make the geometric student depend more directly on projected
function geometry and less on ambient-teacher artifacts.

### Next experiment: projected-field complexity only

Replace the current mixed complexity score with a score computed purely in projected
space from the teacher's scalar field `g(z)`.

Candidate components:

- Hessian norm / curvature of `g(z)`
- boundary proximity from `|g(z)|`
- local residual of coarse global polynomial fit

Do not use:

- ambient activation churn
- ambient ReLU code changes

Why this is the right next step:

- it directly addresses the current `100D` vs `1000D` inconsistency
- if the recovered `2D` is the same up to affine change, the atom placement should
  become much closer across observed dimensions
- it is still geometric and teacher-derived, but now from the projected field itself

### Immediate evaluation target

For the next branch, the minimum question is:

> Holding the projected `2D` chart fixed, can projected-field complexity close the
> `100D`/`1000D` gap and improve over the current ambient-derived atom placement?

Success would look like:

- similar atom placement in `100D` and `1000D`
- improved `1000D` low-complexity performance
- equal or better accuracy than the current hybrid at the same `382 / 754` budget

### After that

If projected-field complexity helps, the next geometric move is:

- anisotropic or boundary-aligned local atoms in projected space

The current isotropic RBFs are still too blunt. They help hard regions, but they
still spill error into easy regions.
