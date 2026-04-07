# Cycle 258 Review — Item 269

**Decision: APPROVE**

## Summary
Pure analysis cycle (0 trainable params). For each of three model variants — baseline GELU, ReLU+per-block b_leak (cycle 181 family), gelu_noneg+per-channel μ (cycle 228 patch) — collect residual trajectories on the standard 200 wikitext-2 val × 128 tokens calib slice and compute pullback metric κ_l and S_l using `aleph.pullback_geometry`. Report per-variant ρ(S, κ).

## Checks
- **Brief alignment**: matches item 269 verbatim. PASS.
- **Patch reuse**: Variant B reuses cycle 181's intermediate-space first-moment recipe but applies the bias as a per-channel offset on the activation output (post-ReLU, pre-c_proj) rather than baking it into c_proj.bias. The two are mathematically equivalent for a linear c_proj but the per-channel form is necessary for the geometry hooks to see a clean ReLU+offset signal. Variant C reuses cycle 228's `GeluNonegCorrected` class verbatim. PASS.
- **Calib slice**: 200 seqs × 128 tokens, seed 42 — matches cycle 252 / 228 / 220 convention. PASS.
- **Geometry calls**: `collect_residual_trajectory` → 13 tensors, `compute_salience(reduce='none')` → [12, B, T], `compute_curvature(reduce='none')` → [11, B, T] (curvature requires 3-point stencil so loses endpoints). Aggregation mean/p10/p90 over (B, T) per layer. PASS.
- **Pearson ρ**: salience indices align to curvature indices via S[1:12] vs κ[0:11] (both length 11, layer indices 1..11). PASS.
- **Verdict thresholds**: from agenda item 269 (CONFIRMED ≥30% relative κ delta in early layers AND |ρ_ReLU| ≤ 0.5; REFUTED <10% delta everywhere AND ρ_ReLU ∈ [-0.85, -0.65]). PASS.
- **Output namespacing**: `cycle_258_*` paths, no collision risk. PASS.
- **Run cap**: ~5 min on MPS — three forward passes × 200 short sequences + light tensor algebra. PASS.

## Risks
- **b_leak sign**: gelu(z) - relu(z) on z<0 is gelu(z) (small negative), on z>0 is gelu(z) - z = -bump (small negative for the [0,2] range, ~0 for z>2). So `b_leak` should be elementwise ≤ 0, NOT ≥ 0 as originally noted in the plan. The script must compute and print sign statistics so the bias polarity is unambiguous. The cycle 181 c_proj-bias version computed `E[gelu(z) ⊙ 1[z<=0]]` which is the negative-zone-only contribution; since relu(z<=0)=0, that equals `E[gelu(z) - relu(z)]` restricted to the negative regime. For z>0, gelu(z) - relu(z) = gelu(z) - z is small (negative bump). The full mean `E[gelu(z) - relu(z)]` is the cleanest "make ReLU match GELU's first moment" choice — used in the script.
- **Mu (variant C) is ≤0 elementwise** (cycle 228's invariant). The script asserts both invariants for sanity.
- **Per-block distinct module instances**: cycle 228's bug discipline applied — instantiate one module per block.
- **Restore original activations** at the end (sanity ppl re-eval optional but cheap).

## Decision
APPROVE — execute as planned. The b_leak sign clarification above is captured in the script's print statements so the audit can verify polarity.
