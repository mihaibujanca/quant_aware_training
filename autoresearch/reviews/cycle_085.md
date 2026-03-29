verdict: APPROVE
round: 1
issues: 0

## R1: Premise validity — PASS

Item 97 asks how many distinct linear behaviors a GPT-2 FFN block has on data. GELU'(z) is the correct elementwise Jacobian of the GELU nonlinearity — clustering these gives activation-pattern diversity. This is the natural follow-up to cycle 083's per-channel census (item 94), now asking the per-token pattern question. Well motivated.

## R2: Measurement soundness — PASS

The critical flaw from cycle 084 (d_c = mean(GELU'(z_i)) != GELU'(z_c)) is explicitly corrected. Step 5 states: "d_c = GELU'(z_c) via autograd (NOT k-means centroid of d)." The task line repeats: "Uses consistent affine linearization: d_c = GELU'(z_c), not k-means centroid of GELU'." The Jensen gap diagnostic (||d_c_recomputed - d_c_kmeans_centroid||) is retained for validation. The reconstruction formula in step 6 is algebraically correct: y_approx = W_out @ [GELU(z_c) + diag(d_c) @ (z - z_c)] + b_proj, which is a first-order Taylor expansion of W_out @ GELU(z) around z_c with consistent d_c = GELU'(z_c).

Null baseline (P5) is well specified: random 3072-dim patterns give all-distinct binary patterns and curse-of-dimensionality k-means. k=1 reconstruction error serves as the null floor.

## R3: Statistical adequacy — PASS

2048 tokens with 1536/512 train/holdout split. P7 correctly flags that k=256 gives only 6 tokens/cluster on the train split and adds the degenerate cluster diagnostic (<3 members). Holdout reconstruction avoids overfitting in the k-means fit. Five corpus sources provide distribution diversity.

Minor note: at k=256, n_init=5 with 1536 samples is fine computationally, and the MiniBatchKMeans fallback handles the runtime concern.

## R4: Method-architecture alignment — PASS

P6 explicitly requires `approximate='tanh'` for all GELU/GELU' computations, matching GPT-2's tanh-approximate GELU. Hook target `model.transformer.h[i].mlp.c_fc` captures pre-GELU activations (post linear projection, pre nonlinearity). Step 6 uses W_out (c_proj weights) and b_proj correctly for the FFN output reconstruction. Validation step checks GELU'(0)=0.5, GELU'(3)~1.0, GELU'(-3)~-0.012 — correct for tanh-approx GELU.

## R5: Known pitfall scan — PASS

- MP1 (threshold above trivial): 5% reconstruction error threshold is well below null (k=1 error will be much larger). Checked.
- MP2 (activation regime mismatch): Using GELU' directly — no regime mismatch.
- MP3 (underdetermined fitting): k-means is unsupervised clustering, not parameter fitting. Reconstruction is analytic (no fitting), evaluated on holdout. N/A.
- MP7 (post-LayerNorm measurements): Hooks are on mlp.c_fc (pre-GELU), not post-LayerNorm. The pre-activations z carry the full geometric signature. Checked.
- MP8 (representation vs functional rank): This measures functional diversity (Jacobian patterns), not representation geometry. Correct level of analysis.

No pitfalls apply.
