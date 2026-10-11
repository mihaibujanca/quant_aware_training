# ANN Geometry for Quantization — Research & Implementation Plan

## 0. Project intent

Build a reusable framework for understanding **weight-only neural-network quantization as geometric deformation**, with two initial capabilities:

1. **Explanatory / error-attribution layer**
   - Decompose quantization error into interpretable mechanisms.
   - Track where error is created, how much is recoverable, and how it changes across layers.
   - Produce diagnostics and visualizations that expose structure rather than only scalar loss/accuracy deltas.

2. **Correction oracle**
   - Run an explicit residual/correction stream alongside the quantized network.
   - Initially correct error immediately from one layer to the next.
   - Later allow corrections to skip layers, become sparse, low-rank, or learned.
   - Use the oracle to study *how much correction is actually necessary* and *how long an error remains recoverable*.

This should be implemented as a reusable research toolkit, not as one-off notebook code.

---

# 1. Initial scope

Start deliberately narrow.

## Network assumptions

- ReLU-based feed-forward networks first.
- **Weight quantization only.**
- Activations remain float32.
- Biases remain float32 initially.
- Start with MLPs / small fully connected networks before CNNs or transformers.
- Keep the first implementation exact and inspectable rather than optimized.

## Why this scope

Weight-only quantization gives a clean causal setting:

\[
W_q = Q(W)
\]

while the activation arithmetic itself is unchanged. Any representation error therefore originates in the changed linear maps and their interaction with nonlinearities.

This lets us separate:

- deformation of the affine/linear map,
- information that remains linearly recoverable,
- information made inaccessible by quantization,
- error introduced by nonlinearities,
- downstream amplification / attenuation / correction.

Do **not** make boundary motion the primary object. ReLU boundary changes matter, but they are one consequence of deformation of the full map.

---

# 2. Core geometric view

For one layer, using row-vector convention,

\[
z_q = xW_q.
\]

Decompose the full-precision map as

\[
W = W_qR + U.
\]

A convenient construction is

\[
R = I + W_q^\dagger(W-W_q),
\]

\[
U = (I-W_qW_q^\dagger)(W-W_q).
\]

Then

\[
xW = z_qR + xU.
\]

Interpretation:

- \(R\): the part of the full-precision transformation recoverable by applying a linear correction to the quantized representation.
- \(U\): the component of the original map outside what can be reconstructed through \(W_q\); this is linearly inaccessible information.

This gives the first major conceptual split:

\[
\boxed{
\text{recoverable map deformation}
\quad\text{vs}\quad
\text{linearly inaccessible information loss}
}
\]

---

# 3. Error-attribution ledger

For the initial weight-only / float-activation setting, let

\[
h = \phi(z_q)
\]

and

\[
y = \phi(xW)=\phi(z_qR+xU).
\]

The exact output discrepancy at this layer can be written as

\[
h-y
=
e_{\mathrm{uncomp}}
+
e_{\mathrm{NL}}
+
e_{\mathrm{range}},
\]

where

\[
e_{\mathrm{uncomp}}
=
h-hR,
\]

\[
e_{\mathrm{NL}}
=
hR-\phi(z_qR),
\]

\[
e_{\mathrm{range}}
=
\phi(z_qR)-\phi(z_qR+xU).
\]

These are the first three named mechanisms in the ledger.

## 3.1 Uncompensated recoverable deformation

\[
e_{\mathrm{uncomp}} = h-hR
\]

This is error that exists because the quantized representation has **not yet been corrected** by the recoverable linear map \(R\).

It is not necessarily information destruction.

Question:

> How much of the observed quantization error is merely a coordinate/map deformation that could be removed by a simple correction?

---

## 3.2 Nonlinearity-induced error

\[
e_{\mathrm{NL}} = hR-\phi(z_qR)
\]

Even if \(R\) recovers the relevant linear map before activation, in general

\[
\phi(z_q)R \neq \phi(z_qR).
\]

The nonlinearity can therefore turn otherwise recoverable linear deformation into error that is no longer recoverable by the same correction after the activation.

Question:

> How much recoverability is destroyed specifically by the nonlinearity?

---

## 3.3 Range / inaccessible error

\[
e_{\mathrm{range}}
=
\phi(z_qR)-\phi(z_qR+xU)
\]

This is the effect of the part \(U\) that was already inaccessible at the linear-map level.

Question:

> How much of the error corresponds to genuinely missing information rather than an uncorrected deformation?

---

# 4. Two ledgers, not one

Keep these two ideas separate.

## A. Error creation

At the layer where an error first appears:

- uncompensated recoverable deformation,
- nonlinearity-induced error,
- linearly inaccessible/range error.

Record the actual error vector per datapoint.

## B. Downstream fate

Once an error exists, track what later layers do to it:

- attenuate it,
- amplify it,
- rotate it,
- project it away,
- turn it into a more harmful direction,
- cancel it against another error source,
- preserve it while keeping it recoverable,
- make it harder to recover.

Do not collapse these into “layer \(l\) caused X output error.”

The source of an error and its eventual downstream consequence are different questions.

---

# 5. Oracle residual stream

The second core piece of infrastructure is an explicit correction stream running alongside the quantized network.

For layer \(l\),

\[
r_l = h_l^{\mathrm{FP}} - h_l^{\mathrm{Q}}.
\]

Initially use the exact oracle:

\[
h_l^{\mathrm{Q}} + r_l = h_l^{\mathrm{FP}}.
\]

This lets us quantize a layer, measure the damage it creates, and then erase that damage before the next layer sees it.

## First mode: immediate correction

Correction is applied from layer \(l\) directly into layer \(l+1\).

Purpose:

- isolate error creation from error propagation,
- validate the attribution ledger,
- establish ground-truth correction targets,
- determine how much of the oracle residual is explained by each ledger component.

## Later modes

Allow the residual to survive for multiple layers:

\[
l \rightarrow l+1 \rightarrow l+2 \rightarrow \cdots \rightarrow k
\]

and inject correction only at depth \(k\).

This creates an empirical **recoverability trajectory**.

For an error created at layer \(l\), define the required correction at layer \(k\):

\[
r_{l\rightarrow k}
=
h_k^{\mathrm{FP}}
-
h_k^{\mathrm{Q},\,l\text{-error}}.
\]

Study how this correction changes with depth.

---

# 6. Recoverability as an object of study

With an omniscient additive oracle, everything is trivially correctable. Therefore “recoverable” must eventually mean recoverable under a constrained correction family.

Examples:

- linear,
- affine,
- diagonal,
- low-rank,
- sparse,
- small MLP,
- low-bit side channel,
- correction folded into a neighboring weight matrix.

For a correction family \(\mathcal C\),

\[
C^\star_k
=
\arg\min_{C\in\mathcal C}
\mathbb E_x
\left[
\|h_k^{\mathrm{FP}} - C(h_k^{\mathrm{Q}})\|^2
\right].
\]

Residual after best allowed correction:

\[
\rho_k(x)
=
h_k^{\mathrm{FP}}
-
C_k^\star(h_k^{\mathrm{Q}}).
\]

This gives a notion of **recoverability horizon**:

> How many layers can a particular error component travel before a correction from the allowed family can no longer recover it within tolerance?

Eventually define something like

\[
H(v)
=
\min\{k:\mathcal E_k(v)>\epsilon\},
\]

where \(v\) is an error direction/component and \(\mathcal E_k\) is residual error after the best allowed correction at depth \(k\).

---

# 7. Why this may improve quantization

The long-term goal is not merely to explain quantization error but to discover **structure in which errors are cheap versus expensive**.

Two equal-norm perturbations may behave very differently:

- one is easily linearly corrected;
- one is projected away downstream;
- one is amplified along a sensitive direction;
- one produces nonlinearity-induced irrecoverability;
- one survives for many layers but remains low-rank and easy to repair.

The useful optimization target may therefore be:

\[
\text{shape error into harmless / correctable directions}
\]

rather than simply

\[
\text{minimize error magnitude}.
\]

A future objective may look like

\[
\min_Q e^\top M e,
\]

where \(M\) encodes the downstream cost of different error directions.

A particularly interesting possibility is that \(M\) or the correction residuals have low-dimensional structure:

\[
M \approx U\Lambda U^\top.
\]

Then only a small subspace of representation-space error may actually be dangerous.

Potential practical consequences:

- smarter mixed precision,
- weight rounding biased toward recoverable directions,
- learned low-rank corrections,
- cross-layer compensation,
- sparse correction points,
- jointly quantized layer groups,
- precision allocated by recoverability horizon rather than by local norm alone.

---

# 8. Reusable software architecture

Avoid burying logic in notebooks. Notebooks should orchestrate experiments and visualization; core mechanisms should live in reusable modules.

Suggested layout:

```text
quant_geometry/
    models/
        instrumented_mlp.py
        wrappers.py

    quantization/
        weight_quantizer.py
        schemes.py

    geometry/
        map_decomposition.py
        recoverability.py

    attribution/
        ledger.py
        source_tracker.py
        interaction.py

    oracle/
        residual_oracle.py
        correction_bridge.py
        correction_families.py

    tracing/
        activation_trace.py
        hooks.py

    analysis/
        propagation.py
        metrics.py
        gram.py

    viz/
        layer_plots.py
        error_vectors.py
        recoverability_plots.py

    experiments/
        configs/
        runners.py

    tests/
        test_map_decomposition.py
        test_ledger_exactness.py
        test_oracle.py
        test_propagation.py

notebooks/
    01_single_layer_ledger.ipynb
    02_mlp_immediate_oracle.ipynb
    03_recoverability_depth.ipynb
```

---

# 9. Core module responsibilities

## 9.1 `WeightQuantizer`

Responsibilities:

- quantize weights only;
- preserve float32 activations;
- expose original and quantized weights;
- support several simple schemes eventually, but start with one;
- deterministic operation for reproducibility.

Initial interface idea:

```python
quantizer = WeightQuantizer(bits=8, symmetric=True)
Wq, meta = quantizer(W)
```

Do not optimize this subsystem prematurely.

---

## 9.2 `MapDecomposition`

Given \(W\) and \(W_q\), compute:

- \(R\),
- \(U\),
- reconstruction diagnostics,
- numerical rank,
- singular spectra,
- conditioning information.

Interface:

```python
decomp = MapDecomposition.from_weights(W, Wq)

R = decomp.R
U = decomp.U

report = decomp.diagnostics()
```

Must numerically verify:

\[
W \approx W_qR + U.
\]

This module should be independent of any network.

---

## 9.3 `ErrorLedger`

Given:

- input \(x\),
- FP weight \(W\),
- quantized weight \(W_q\),
- activation function \(\phi\),
- map decomposition \(R,U\),

compute exact per-sample vectors:

```python
ledger = ErrorLedger.compute(...)
ledger.uncomp
ledger.nonlinear
ledger.range
ledger.total
ledger.reconstruction_error
```

Must verify:

\[
e_{\mathrm{uncomp}}
+
e_{\mathrm{NL}}
+
e_{\mathrm{range}}
\approx
h-y.
\]

Do this per datapoint, not only after averaging.

The first hard test should fail loudly if the ledger does not telescope numerically.

---

## 9.4 `ActivationTrace`

For every relevant layer, record:

- FP32 preactivation,
- quantized preactivation,
- FP32 post-activation,
- quantized post-activation,
- oracle-corrected activation,
- optional masks / ReLU states,
- layer metadata.

Keep tracing separate from attribution logic.

---

## 9.5 `ResidualOracle`

Basic oracle:

```python
r = h_fp - h_q
```

Support:

```python
oracle.residual(layer)
oracle.correct(h_q, layer)
```

Modes:

1. **Immediate**
   - correct immediately after each selected layer.

2. **Delayed**
   - carry the quantized trajectory for \(d\) layers, then correct.

3. **Component-wise**
   - maintain residuals corresponding to named error sources.

4. **Constrained correction**
   - later replace exact residual with best correction from a specified family.

---

## 9.6 `CorrectionBridge`

Represent a correction edge:

```text
source layer  --->  correction target layer
```

For example:

```python
CorrectionBridge(
    source=3,
    target=4,
    mode="exact"
)
```

Later:

```python
CorrectionBridge(
    source=3,
    target=7,
    family=LowRankLinear(rank=4)
)
```

This abstraction matters because the long-term experiment is likely to become a **sparse correction graph** over the network.

---

## 9.7 `SourceTracker`

Track an error source independently through downstream layers.

For source \(i\) created at layer \(l\), retain:

- source mechanism,
- source vector,
- propagated representation discrepancy,
- correction needed at later layers,
- norm,
- cosine / angular change,
- effective rank across dataset,
- recoverability under each correction family.

Important: distinguish attribution from propagation.

---

# 10. Component-wise residual streams

Eventually maintain separate streams for

\[
r^{\mathrm{uncomp}},
\quad
r^{\mathrm{NL}},
\quad
r^{\mathrm{range}}.
\]

However, after nonlinearities, separate components do **not** propagate additively in general.

If

\[
r=r_1+r_2,
\]

then typically

\[
F(h+r_1+r_2)-F(h)
\neq
[F(h+r_1)-F(h)]
+
[F(h+r_2)-F(h)].
\]

Therefore do not assume independent component propagation remains exact.

Initial solution:

- use a fixed-order counterfactual telescoping intervention;
- add one component at a time;
- define each downstream contribution as the difference between successive interventions.

This preserves exact accounting but introduces ordering dependence.

Do not implement Shapley-style attribution initially.

---

# 11. Error interaction / Gram ledger

When aggregating error sources, do not assume orthogonality.

For error sources \(e_i\), define

\[
G_{ij}
=
\mathbb E_x
\left[
\langle e_i(x),e_j(x)\rangle
\right].
\]

Interpretation:

- diagonal \(G_{ii}\): energy of source \(i\),
- positive off-diagonal: reinforcement,
- negative off-diagonal: cancellation.

This should be a reusable analysis primitive, but it is not required for the very first implementation milestone.

---

# 12. Initial visualizations

The first visualizations should explain the ledger, not merely decorate it.

Useful plots:

### Per layer

- norm of each error component;
- fraction of total error attributable to each mechanism;
- reconstruction/telescoping residual;
- singular spectrum of \(R\);
- singular spectrum / rank diagnostics of \(U\).

### Per datapoint

- component norms,
- angles between components,
- whether large total error comes from one mechanism or interaction,
- optional ReLU-state changes.

### Across layers

- error source magnitude vs depth,
- correction magnitude vs depth,
- linear-correction residual vs depth,
- effective rank of correction vs depth,
- recoverability horizon.

### Toy low-dimensional examples

For 2D/3D networks, explicitly visualize:

- original map,
- quantized map,
- recoverable deformation,
- inaccessible component,
- post-ReLU error.

These are especially useful for building intuition and for eventual exposition.

---

# 13. First implementation milestones

## Milestone 1 — exact single-layer decomposition

Build:

- `WeightQuantizer`
- `MapDecomposition`
- `ErrorLedger`

Test on one linear + ReLU layer.

Acceptance criteria:

- exact decomposition \(W \approx W_qR+U\),
- exact telescoping of ledger within numerical tolerance,
- per-sample component vectors exposed,
- basic plots produced.

No multi-layer propagation yet.

---

## Milestone 2 — instrumented MLP

Build:

- small ReLU MLP,
- FP and quantized copies,
- activation tracing,
- hooks/wrappers that expose per-layer state.

Acceptance criteria:

- arbitrary selected layer can be quantized independently,
- FP and quantized trajectories can be compared at every layer,
- ledger can be evaluated layer-by-layer.

---

## Milestone 3 — immediate oracle

Add exact residual correction after a selected quantized layer.

Experiment:

1. Run FP network.
2. Quantize layer \(l\).
3. Measure its error.
4. Inject exact residual before layer \(l+1\).
5. Verify that downstream discrepancy caused by that layer disappears up to numerical precision.

This establishes a clean causal intervention primitive.

---

## Milestone 4 — immediate oracle on all layers

Quantize all layers, but immediately correct after each one.

Purpose:

- isolate the error each layer creates when its input is reset to the FP representation,
- build a clean layer-by-layer ledger without contamination from upstream quantization,
- compare isolated error creation against ordinary end-to-end quantized behaviour.

This becomes the baseline “fully instrumented” mode.

---

## Milestone 5 — delayed oracle

For an error source at layer \(l\), delay correction until

\[
l+1,\; l+2,\; l+3,\ldots
\]

Measure:

- required exact residual,
- best linear/affine correction,
- residual after constrained correction,
- rank / spectrum of the correction,
- direction changes,
- downstream amplification.

This produces the first recoverability curves.

---

# 14. First experiments

Use a small trained ReLU MLP initially.

Possible datasets:

1. synthetic 2D classification for visualization,
2. MNIST once the instrumentation is stable.

For each layer:

1. quantize weights,
2. compute \(R,U\),
3. compute the three ledger components,
4. verify exact accounting,
5. measure per-sample distributions,
6. run immediate oracle correction,
7. delay correction one layer,
8. compare correction complexity.

Questions to answer first:

- What fraction of local error is uncompensated-but-recoverable?
- What fraction is nonlinearity-induced?
- What fraction comes from inaccessible \(U\)?
- Do these fractions vary sharply by layer?
- Do large local errors necessarily remain large downstream?
- Does a correction become simpler or more complex after one layer?
- Are some error directions naturally suppressed?
- Is downstream correction low-rank even when local error is not?

---

# 15. Important experimental modes

The framework should eventually support three clean modes.

## Mode A — ordinary quantized inference

No oracle.

Purpose:

- measure real end-to-end quantization behaviour.

## Mode B — immediate oracle

Correct after every selected layer.

Purpose:

- isolate error creation.

## Mode C — delayed / sparse oracle

Correct only at selected later layers.

Purpose:

- study propagation, recoverability, and cross-layer correction.

These should share the same instrumentation API.

---

# 16. Long-term research questions

Once the explanatory layer works:

## 16.1 Is harmful error low-dimensional?

Does quantization create many error directions but only a few that matter downstream?

## 16.2 Can a quantizer shape its error?

Can weight rounding be chosen so that unavoidable error lands in directions that are:

- linearly correctable,
- downstream-null,
- weakly amplified,
- long-horizon recoverable?

## 16.3 Can corrections be sparse across depth?

Instead of correcting every layer, can a few bridges repair many preceding layers?

## 16.4 Can cross-layer correction be folded into weights?

If a correction is linear:

\[
h_lR_lW_{l+1}
\]

may be absorbed into

\[
W'_{l+1}=R_lW_{l+1}.
\]

This could turn oracle-discovered structure into practical layerwise compensation.

## 16.5 Can quantization optimize recoverability rather than local reconstruction?

Replace local objectives such as

\[
\|W-W_q\|^2
\]

with objectives informed by downstream geometry and correction cost.

---

# 17. Non-goals for the first phase

Do **not** initially spend time on:

- activation quantization,
- clipping optimization,
- mixed precision,
- transformer-specific machinery,
- CUDA kernels,
- hardware benchmarking,
- large models,
- elaborate quantization search,
- end-to-end training of correction modules,
- Shapley attribution,
- productionization.

The first phase succeeds if it gives a rigorous and reusable **explanation + intervention apparatus**.

---

# 18. Immediate coding order

Implement in this order:

1. `WeightQuantizer`
2. `MapDecomposition`
3. `ErrorLedger`
4. exact unit tests for the decomposition and ledger
5. tiny one-layer notebook
6. `ActivationTrace`
7. instrumented MLP wrapper
8. `ResidualOracle`
9. immediate `CorrectionBridge`
10. multi-layer causal test
11. delayed correction
12. constrained correction families
13. propagation/recoverability analysis
14. richer visualization

The first checkpoint is not “good quantization performance.”

It is:

> **For any chosen layer and datapoint, can we say exactly what error was created, which mechanism created it, and what exact correction would restore the FP representation?**

Once that works robustly, the project has a foundation.
