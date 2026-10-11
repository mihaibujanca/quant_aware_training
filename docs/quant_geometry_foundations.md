# Quantization geometry: technical reference

**Start with [Watch the shape, then name the error](quant_geometry_visual_guide.md)** for the geometric explanation: the same grid and circle deform, ReLU clips them, and distinct points can collapse together. This document records the equations, implementation and numerical checks after that picture is established.

Implementation of the first three next steps in [the progress tracker](quant_geometry_progress.md), guided by [the research plan](quantization_geometry_plan.md). These are deterministic algebra and instrumentation examples, not evidence of task accuracy or compression gains.

## 1. Bounded weight quantization and map decomposition

`aleph/qgeom/weight_quantizer.py` adds a deterministic, per-tensor symmetric weight quantizer. It returns detached original and dequantized weights plus bit width, scale, bounds and rounding metadata. For `bits=b`, the integer bounds are `[-(2**(b-1)-1), +(2**(b-1)-1)]`; one signed code is unused to keep the grid symmetric. Scale is `absmax/qmax`; a zero matrix uses scale 1. Rounding is nearest with ties to even. The runner leaves biases and activations floating point. This is inspectable fake quantization, not packed integer execution.

`aleph/qgeom/map_decomposition.py` implements the network-independent split, using **row-vector weights `[input, output]`**:

\[
R=I+W_q^\dagger(W-W_q),\qquad U=(W-W_q)-W_qW_q^\dagger(W-W_q).
\]

Let `W,Wq` have shape `[n,m]`, so `R` has shape `[m,m]` and `U` has shape `[n,m]`. Writing `D=W-Wq`, reconstruction follows directly:

\[
W_qR+U=W_q+W_qW_q^\dagger D+(I-W_qW_q^\dagger)D=W_q+D=W.
\]

For the **exact Moore–Penrose inverse**, `P=Wq Wq†` is the orthogonal projector onto the column space of `Wq` in input space. Then `U=(I-P)W` and `Wq.T @ U=0`. `R` recovers the projection `PW` through the quantized **preactivation**, while `U` maps input information unavailable from `xWq`. If `U!=0`, no linear decoder can reconstruct `xW` from `xWq` for all inputs. Restricted data distributions can change that conclusion, and ReLU can hide a missing direction's output effect.

**Numerical truncation is a different statement.** The implementation uses a tolerance-truncated inverse `Wq†_τ`: singular values at or below `τ=rtol*σmax` are omitted. With `Pτ=Wq Wq†_τ`, the implemented remainder is `Uτ=(I-Pτ)(W-Wq)`. Reconstruction still holds, and `Uτ` is orthogonal to the **retained** column space. However, it need not satisfy `Wq.T @ Uτ=0` or equal `(I-Pτ)W`. A discarded nonzero singular direction is still present in the original quantized preactivation; recovering it may require a very large gain. Accordingly, a nonzero `Uτ` is not by itself proof of information loss from the full `Wq`.

The pseudoinverse tolerance is exposed (`rtol`, default `max(shape)*dtype_epsilon`). Diagnostics report numerical rank, retained-spectrum conditioning, all three singular spectra, `max(abs(Wq.T @ U))`, and reconstruction residual. The reported full-map orthogonality residual need not vanish under truncation; it is not an acceptance criterion in that case. `U_rank` uses an absolute threshold scaled to the original maps so that tiny floating-point leftovers are not counted as meaningful rank.

![Map decomposition and singular spectra](figures/quant_geometry/map_decomposition.png)

**Figure 1.** A bounded 2-bit quantizer maps `W=[[1,-0.35],[0.28,0.22]]` to `Wq=[[1,0],[0,0]]`. The resulting `R=[[1,-0.35],[0,1]]` recovers the first input row. `U=[[0,0],[0.28,0.22]]` retains the second row, which is absent from `Wq`. This example has an exact null direction, so its missing-information interpretation is valid. A full-row-rank widening map has `U=0` in exact arithmetic when no nonzero singular values are discarded.

## 2. Three exact error components, with float bias

`aleph/qgeom/ledger.py` computes actual component vectors for **every sample**. Every local ledger compares the FP and quantized layer on the **same input**. Let

\[
z_q=xW_q+b,\quad h=\phi(z_q),\quad y=\phi(xW+b),\quad A(v)=vR+b-bR.
\]

The affine offset preserves the shared float bias:

\[
A(z_q)+xU=(xW_q+b)R+b-bR+xU=x(W_qR+U)+b=xW+b.
\]

Apply the same affine correction after activation when defining the ledger:

\[
e_{\rm uncomp}=h-A(h),\qquad
 e_{\rm NL}=A(h)-\phi(A(z_q)),\qquad
 e_{\rm range}=\phi(A(z_q))-y.
\]

Their sum telescopes:

\[
[h-A(h)]+[A(h)-\phi(A(z_q))]+[\phi(A(z_q))-y]=h-y.
\]

The **total** is signed Q-minus-FP; each component is a signed difference between successive counterfactual states, not individually a Q-minus-FP error. With `b=0`, these reduce to the plan's equations. With the identity activation, `h=zq` and `A(h)=phi(A(zq))`, so nonlinear error is zero even with bias.

These mechanisms are an **accounting convention**, not a unique causal decomposition. The choice `R=I+Wq†(W-Wq)` fixes an identity-centered extension. For an exact pseudoinverse, adding any `N` with `Wq N=0` leaves the reconstructed linear map unchanged. But ReLU can move `h` outside the row space of `Wq`, so `hN` can be nonzero and change the uncompensated/nonlinear terms. The implementation fixes `R`; comparisons across different rank policies or correction extensions must state those choices.

The API exposes preactivations, postactivations, `recoverable_pre=A(zq)`, `recoverable_post=phi(A(zq))`, and `transported_post=A(h)`. `ErrorLedger.compute` checks both affine reconstruction and telescoping coordinate by coordinate and raises on failed checks. Default absolute/relative tolerances are `1e-5` for float32 and `1e-10` for float64; callers may override them. Ill-conditioned maps may require float64 or an explicitly changed rank tolerance rather than silently relaxing the check.

### Uncompensated recoverable deformation

`ledger.uncomp = h-A(h)` is the discrepancy caused by leaving the available affine correction unapplied. It can be large even when the representation retains enough information to repair the linear map. It is signed Q-minus-corrected, so the correction itself points in the opposite direction.

![Uncompensated deformation: input footprint, signed vectors and example](figures/quant_geometry/component_uncomp.png)

**Figure 2.** The negative shear in `R` would move a positive quantized first coordinate into a negative second coordinate. The blue arrows are `h-A(h)` (from corrected state to uncorrected state), not correction arrows. The right panel compares the two states and their signed difference at the largest example. These corrected-after-ReLU coordinates may be negative; they are intermediate counterfactual states, not ordinary ReLU outputs.

### Nonlinearity-induced error

`ledger.nonlinear = A(h)-phi(A(zq))` measures failure of the recoverable affine correction to commute with the activation. Correcting the map after ReLU and correcting it before ReLU generally give different results. This identifies the failure of this correction family/order; it does not prove that an unrestricted oracle or every other decoder fails.

![Nonlinearity-induced error: input footprint, signed vectors and example](figures/quant_geometry/component_nonlinear.png)

**Figure 3.** For a positive first input coordinate, ReLU clips the negative second coordinate created by the shear; the orange and blue components cancel. For a negative first coordinate (the selected example in the figure), applying the shear before ReLU creates a positive second coordinate, whereas applying ReLU first removes the input needed to create it. The orange vectors capture both cases. Nonlinear error can therefore cancel another component rather than increase the total norm.

### Linearly inaccessible range error

`ledger.range = phi(A(zq))-y` measures the output effect of adding back `xU`. It depends on the input and activation: a nonzero `U` can be masked by ReLU, while another sample exposes it. For an exact null-space split, the missing input contribution cannot be reconstructed linearly from `xWq` for all inputs. Under numerical truncation, this term can instead reflect intentionally uncorrected, small nonzero singular directions. Both remain correctable by an oracle with FP access.

![Range error: input footprint, signed vectors and example](figures/quant_geometry/component_range.png)

**Figure 4.** The removed second input row changes the FP output while leaving the quantized preactivation unchanged for inputs with the same first coordinate. The purple arrows compare the recovered affine output with the actual FP output. This isolates missing input information from the recoverable shear.

### Exactness and interaction

For the specific zero-bias input `x=(1,2)`, the states are:

| State / vector | Value |
|---|---|
| Quantized output `h` | `(1, 0)` |
| Transported output `hR` | `(1, -0.35)` |
| Activated recovered preactivation | `(1, 0)` |
| FP output `y` | `(1.56, 0.09)` |
| Uncompensated | `(0, +0.35)` |
| Nonlinear | `(0, -0.35)` |
| Range | `(-0.56, -0.09)` |
| Total Q-minus-FP | `(-0.56, -0.09)` |

The first two vectors cancel at this input; they do not cancel over the whole grid. For this particular fixture, writing the first input coordinate as `s` and `s_+=max(s,0)` gives

\[
e_{\rm uncomp}=(0,0.35s_+),\qquad
e_{\rm NL}=(0,-0.35|s|).
\]

Thus they cancel for `s>=0`, but for `x=(-1,0)`, uncompensated error is zero and nonlinear error is `(0,-0.35)`. Reporting individual energies as additive fractions of total energy would be misleading. `ledger.gram()` provides `Gij=mean(<ei,ej>)`, with

\[
\mathbb E\|e_{\rm total}\|^2=\sum_iG_{ii}+2\sum_{i<j}G_{ij}=\sum_{i,j}G_{ij}.
\]

Each off-diagonal pair contributes **twice** to total energy. This is an uncentered second-moment Gram matrix, not a covariance matrix or unique allocation of error percentages.

![Ledger norms, interaction matrix and exactness](figures/quant_geometry/ledger_accounting.png)

**Figure 5.** Norms are shown separately, the Gram matrix exposes negative cross terms, and the final panel shows the per-sample telescoping residual. In the saved float64 grid experiment its maximum is approximately `5.55e-17`.

## 3. Shared MLP traces and residual oracle

`aleph/qgeom/instrumented_mlp.py` extracts the reusable forward/oracle responsibilities previously embedded in `notebooks/canonical_error_correction.py`. It reuses `ForwardTrace` from `aleph/qgeom/canonical.py`; it does not replace the existing canonical-coordinate analysis. The new runner adapts the old oracle concept to explicit post-activation FP-minus-Q residuals, preserving the raw damaged states.

`InstrumentedMLP` snapshots an ordinary float `nn.Sequential` chain of Linear/ReLU modules or an unprepared `aleph.models.MLP`. Hidden linear layers require ReLU; the final layer may be linear or ReLU. Other modules, prepared QAT models and unsupported wrappers are rejected. PyTorch `[output,input]` weights are transposed at the adapter boundary. The snapshot is detached, independent of later edits to the model, and supports float32/float64 on the model's device.

Each `LayerTrace` contains:

- Layer index/name, activation, quantization metadata and selected status.
- Actual run input and reference FP input.
- Reference FP and raw Q pre/post states; consumed post state after any correction.
- Exact oracle residual, correction-applied flag and ReLU disagreement mask against reference FP.
- Same-input creation ledger, propagated vector and raw total vector.

For a run input `a` that may already differ from the FP input, the runner defines local creation as `phi(aWq+b)-phi(aW+b)` and propagated error as `phi(aW+b)-h_fp`. These vectors telescope to raw total error. Norms do not necessarily add. This is a local-versus-incoming accounting identity; it does not separately propagate the three mechanisms through nonlinear downstream layers.

`ActivationTrace.fp` is the reference `ForwardTrace`; `.raw_quantized` contains pre/post states before each intervention; `.consumed_post_acts` contains states passed to the next layer. A raw trace in an oracle run is conditioned on earlier corrected inputs, so it must not be confused with ordinary quantized inference. `.output` is the final consumed state.

`ResidualOracle.residual(hq, layer)` returns `h_fp-hq`, the **opposite sign** to the ledger. A `CorrectionBridge(source=i,target=i)` restores the output of linear layer `i` before layer `i+1` consumes it. A later target restores at that later layer. Indices are zero-based **linear-layer indices**, not Sequential module positions. An exact bridge restores all accumulated discrepancy at its target; it is a source-specific intervention only when the named source is the sole perturbed layer.

![Independent layer quantization and raw/consumed states](figures/quant_geometry/activation_trace.png)

**Figure 6.** Only layer 0 of a seeded, untrained `2→5→3→2` MLP is quantized. Ordinary Q inference propagates the damage. The immediate bridge leaves raw damage visible at layer 0, restores its consumed activation, and returns the FP downstream trajectory. The other panels expose the individual sample states, local/propagated accounting, and local component magnitudes. Float biases and activations are preserved.

![All-layer immediate correction and isolated local ledgers](figures/quant_geometry/oracle_trace.png)

**Figure 7.** Quantizing every layer while immediately restoring every output gives every layer the FP input. Each raw ledger still records nonzero damage, while consumed states match FP. The saved example's final discrepancy is zero in float32; tests use explicit numerical tolerances rather than demanding universal bitwise equality.

### API example

```python
import torch
from torch import nn
from aleph.qgeom import (
    WeightQuantizer, MapDecomposition, ErrorLedger,
    InstrumentedMLP, CorrectionBridge,
)

# Geometry modules use row-vector convention.
W = torch.tensor([[1., -.35], [.28, .22]], dtype=torch.float64)
x = torch.tensor([[1., 2.]], dtype=W.dtype)
quantizer = WeightQuantizer(bits=2)
q = quantizer(W)
decomp = MapDecomposition.from_weights(W, q.quantized)
ledger = ErrorLedger.compute(x, decomp)
assert torch.allclose(ledger.uncomp + ledger.nonlinear + ledger.range, ledger.total)

# Runner adapts nn.Linear weights and keeps all biases/activations float.
model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 2))
runner = InstrumentedMLP(model, quantizer=quantizer)
ordinary = runner.run(x.float(), quantized_layers=[0])
corrected = runner.run(x.float(), quantized_layers=[0],
                       bridges=[CorrectionBridge(source=0, target=0)])
```

## Reproduction, validation and contribution record

From repository root:

```sh
.venv/bin/python -m experiments.quant_geometry_foundations
.venv/bin/python -m pytest tests/test_qgeom_exact_ledger.py tests/test_qgeom_instrumented_mlp.py tests/test_qgeom_metrics.py tests/test_qgeom_policy.py -q
```

The experiment saves eight figures plus [the run manifest](figures/quant_geometry/run_manifest.json). The manifest records revision/dirty state, seed, package versions, exact model weights and biases, dataset/grid, precision, quantization scales, sites, tolerances and measured residuals. The grid example uses no trained or fitted component, so no train/test split is claimed. The rank-policy audit uses separately prescribed matrices, explicitly identified in the manifest; they are not outputs of the 2-bit quantizer.

| Contribution | Files | Verification / associated visual |
|---|---|---|
| Bounded weight-only quantization and metadata | `aleph/qgeom/weight_quantizer.py` | Bit bounds, ties, zero weights, deterministic output and snapshot ownership; Figure 1 |
| Recoverable/range map split and diagnostics | `aleph/qgeom/map_decomposition.py` | Rectangular, zero, rank-deficient and explicitly truncated maps; Figure 1 |
| Bias-aware exact creation ledger and interaction summary | `aleph/qgeom/ledger.py` | Per-sample telescoping, affine reconstruction, identity activation, ReLU cancellation and Gram identity; Figures 2–5 |
| Shared traces and model adapter | `aleph/qgeom/instrumented_mlp.py` | FP equivalence, arbitrary layer selection, detached snapshots, biasless/final-ReLU models and unsupported architecture checks; Figure 6 |
| Explicit residual oracle and correction bridges | `aleph/qgeom/instrumented_mlp.py` | Immediate correction at each selected source, all-layer reset of inputs, delayed target sanity check, raw-state preservation and residual sign; Figures 6–7 |
| Reusable plotting and reproducible example | `aleph/qgeom/ledger_plots.py`, `experiments/quant_geometry_foundations.py` | Generated artifacts and manifest; Figures 1–8 |

This implements the plan's single-layer foundation and shared MLP/oracle apparatus. The causal tests also establish immediate single-layer and all-layer restoration. Delayed-site plumbing has a sanity test, but fitted correction families, held-out recoverability curves, ordered component propagation and source-specific tracking remain future work. Existing notebook research claims were not revalidated by retraining here.

## Mathematical audit: numerical rank and attribution conventions

The reconstruction, affine-bias, telescoping, Gram-energy, and local/propagated identities above were checked algebraically and numerically. The zero-bias table and a biased example now have independent expected-coordinate tests. Two interpretation limits needed clarification:

**Truncation can create a residual without losing information.** For prescribed `W=I` and `Wq=diag(1,1e-8)`, a cutoff of `1e-6` gives `R=I` and `U=diag(0,1-1e-8)`. Retaining the small singular value instead gives `R=diag(1,1e8)` and `U=0`. The second input coordinate is weakly represented, not absent. In the truncated case, `Wq.T @ U=diag(0,1e-8*(1-1e-8))` is nonzero; only retained-space orthogonality holds.

![Numerical truncation versus an invertible high-gain correction](figures/quant_geometry/rank_policy_audit.png)

**Figure 8.** Singular values and both cutoffs, the resulting remainder norms, and recovery of the second output. The rank-2 decoder recovers the FP coordinate from Q alone at gain `1e8`; the truncated correction needs the remainder. This figure distinguishes a numerical policy decision from actual loss through an exact null direction.

**The three components depend on the chosen extension of R.** Take `W=Wq=[[1,-1],[0,0]]` and `x=(1,0)`. The implemented identity-centered choice gives `R=I`, `U=0`, and all errors zero. Alternatively, let `N` have every entry `0.5`. Since `Wq N=0`, `R'=I+N` reconstructs the same weights. Yet `h=(1,0)` gives `hR'=(1.5,0.5)`: uncompensated error is `(-0.5,-0.5)` and nonlinear error is `(0.5,0.5)`. They cancel and total error remains zero. Nonzero components alone therefore do not establish physical information destruction; the declared correction convention matters.

The existing Gram figure (Figure 5) illustrates why signed interaction must be reported alongside component magnitudes. No numerical correction was needed to the original `(1,2)` table or the affine-bias equations.
