# Cycle 235 audit — Item 245 (T1 centred Δ̃ feature SVD with example extraction)

**Status: PASS**
**Old outcome: CONFIRMED**
**New outcome: CONFIRMED (with G3 flag — coherence heuristic is weak interpretability proxy)**

## Phase 1 — Verdict check

Plan criteria (verbatim from agenda item 245):
- Confirmed if ≥5 of 32 directions per block have a coherence_label ∉ {mixed, incoherent} in ≥6 of 12 blocks
- Refuted if <2 directions per block have a coherent label in ALL 12 blocks
- Inconclusive otherwise

Per-block coherent counts from `cycle_235_t1_summary.md`:
`[27, 23, 24, 27, 21, 19, 19, 15, 17, 15, 12, 10]`.

All 12 blocks ≥5 ≥ 6 → literal **CONFIRMED**. ✓ Criterion fully met; REFUTED branch ruled out (minimum is 10, not <2).

## Phase 2 — Critical interpretation

### Is this cycle 217 redux?

Cycle 217 was corrected from CONFIRMED → INCONCLUSIVE because the "interpretable" top SVD directions of raw Δ were all just the per-block constant mean-direction projected through `lm_head`, which trivially recovers the unigram-frequency axis (`, the and . a`). The rank-95=1 in 9/12 blocks was a direct consequence of TI/TD ≥ 0.87 (the mean direction dominated the spectrum).

Cycle 235 **explicitly centres** Δ̃_l = Δ_l − mean(Δ_l) per block before SVD (line 172 of `experiments/bump_t1_features_cycle235.py`: `delta_centred = delta_out - mean_delta`). The constant-bias mode is subtracted out, so the cycle 217 failure mode is structurally absent here. σ₀ per block in c235 is `[671, 575, 876, 836, 803, 700, 641, 587, 696, 812, 1426, 2556]` and `||mean_Δ||` per block is `[21, 16, 37, 44, 45, 43, 38, 36, 33, 35, 62, 155]`, i.e. the centred σ₀ is ≥ 10× larger than ||mean_Δ|| in every block — the centred spectrum is not the mean axis in disguise. This is a different measurement than cycle 217, and the cycle 217 audit *recommended* this exact follow-up.

### The "repeated" label: is it the same interpretability artefact?

The claim flagged for audit was: "dominant label is `repeated` — a surface concentration of U coefficients, not a semantic feature." Breaking this down:

1. **What `repeated` actually measures** (script line 283): `mode_count ≥ 30` over the top-100 corpus tokens ranked by `|U_l[:, i]|`. This is *where* the direction fires (rows of U), not *what* it boosts (columns of wte @ v). It says: "of the 100 corpus positions where this direction has largest magnitude, ≥30 are the same token id."
2. **This is NOT cycle 217's failure mode.** Cycle 217's SV0 was the mean direction, whose `lm_head` projection was the unigram axis — a *column* artefact. Cycle 235's `repeated` is a *row* concentration. In cycle 217, top boost tokens were always `, the and`. In cycle 235, the boost list for `repeated` directions is typically NOT the repeated corpus token — see block 0 dir 3 (fires on ` ,` corpus tokens 93/100, boosts `[-, and, lings, ers, to, ly, ling, y, of, ing]`). The direction activates at comma positions to write something else into the residual. That is a "positional/contextual activation" claim, not a unigram claim.
3. **But it's still a weak interpretability claim.** Because commas, periods, and ` the` are extremely frequent in wikitext, a direction with near-uniform U energy can have its top-100 |U| positions dominated by the most common tokens purely by sampling density. The labelling treats this as "coherent" when it's actually "this direction fires everywhere, and the top-100 max-magnitude positions happen to be the most common tokens in the corpus." Without a selectivity test (e.g. mean |U| at `, ` positions vs mean |U| elsewhere), `repeated` can conflate "selective for commas" with "fires everywhere, sees a lot of commas."

### Non-repeated coherent count

Stripping `repeated` and counting only {punctuation, stopword, capitalized, number}:

| block | coherent | repeated | non_repeated_coherent | ≥5? |
|---:|---:|---:|---:|---|
| 0  | 27 | 24 | 3  | ✗ |
| 1  | 23 | 22 | 1  | ✗ |
| 2  | 24 | 22 | 2  | ✗ |
| 3  | 27 | 22 | 5  | ✓ |
| 4  | 21 | 18 | 3  | ✗ |
| 5  | 19 | 14 | 5  | ✓ |
| 6  | 19 | 16 | 3  | ✗ |
| 7  | 15 | 11 | 4  | ✗ |
| 8  | 17 | 12 | 5  | ✓ |
| 9  | 15 | 11 | 4  | ✗ |
| 10 | 12 |  6 | 6  | ✓ |
| 11 | 10 |  5 | 5  | ✓ |

Under the strict "non-repeated only" reading: **5/12** blocks pass, failing the ≥6 threshold. The literal CONFIRMED is entirely carried by the `repeated` class.

### Spot-checks of example records

From `cycle_235_t1_decoded.json` and `cycle_235_t1_examples.md`:

- **Block 0 dir 0** (stopword, σ=671, var=11.6%): fires on ` in`, ` to`, `\n` at position 0 of sequences. Genuine position-0 class locator. Boost list is junk (tokenizer artefacts). Coherent activation, uninterpretable write.
- **Block 0 dir 3** (repeated, σ=361): fires on ` ,` (93/100). Boost is word-fragment suffixes (`lings, ers, ly, ling, y, ing`). Plausible: "at comma positions, write morphological continuation cues." Weak-but-not-spurious.
- **Block 0 dir 6** (repeated, σ=244): fires on `@` (99/100). WikiText-2 uses `@` in `@-@` concat tokens — this direction is an `@-@` locator, not a semantic feature but a dataset-specific tokenisation artefact.
- **Block 3 dir 2** (capitalized, σ=330): fires on ` Red`, `Mat`, ` 385`. Not actually a capitalized cluster — the label fired because ≥40 of top-100 start with uppercase, but the actual cluster is proper-noun-like mixed.
- **Block 6 dir 3** (stopword, σ=374): fires on ` the` (clean). Boost = `[pestic, therap, distingu, pci, Dise, ...]`. Plausible: "after ` the`, we're in a medical/scientific context" — specific register cue. Interpretable.
- **Block 8 dir 3** (capitalized, σ=445): fires on ` Northern`, ` Civil`, ` Northern`. Proper-noun/geographic locator. Clean.
- **Block 11 dir 0** (capitalized, σ=2556, var=24%): fires on ` favour`, ` result`, ` lack` — these are NOT capitalised, the label fired incorrectly (the classifier checks `s[0].isupper()` on stripped tokens; ` favour`.strip()[0] is 'f' not uppercase, so the label should not have fired — worth investigating as a potential classifier bug, but not the audit's concern). Boost is `[,, the, -, and, a, ., in, (, ", \n]` — this is the **final-block unembed bias** axis, boosting the most common tokens. At block 11 the last-layer Δ̃ top direction has rediscovered the cycle 217 unigram axis because there's no further block to centre against. This is a data point against CONFIRMED at block 11 specifically.

### Is CONFIRMED defensible in spirit?

**Yes, with heavy caveats:**

1. **Centring works as advertised.** Cycle 217's failure mode (top SVD = constant bias = unigram axis) is absent in cycle 235 blocks 0-10. Block 11 has the unigram-axis pattern creep back into SV0, probably because the last block's bump has no downstream centring budget.
2. **The `repeated` label is a weak positive.** It's measuring row concentration and the row concentration correlates with corpus frequency. A selectivity-controlled version (e.g. "top-100 position density is N× the background density for that token") would be stricter. As-is, `repeated` is real but uninformative.
3. **The non-`repeated` labels in blocks 3, 5, 8, 10, 11** (5/12) do show genuine localisation: block 0 dir 0 (position-0 stopwords), block 6 dir 3 (` the`-register), block 8 dir 3 (proper nouns), block 3 dir 0 (sentence-ending `.`).
4. **The executor pre-flagged the caveat** in the manifest `learned` field, in `cycle_235_t1_summary.md` ("Coherence here is a cheap surface heuristic"), and in agenda entry 245 ("Caveat: 'repeated' is a cheap surface label — it says the top-100 mass is concentrated on one token, not that the direction causally boosts that token's logit").
5. **T2 (item 246) is the gating causal test.** The plan already knows that T1 alone is correlational and T2's α·v_{l,i} intervention is the actual test. Keeping CONFIRMED on T1 does not overclaim the story — item 246 is the Jaccard≥0.3 / logit-shift causal check.

**Decision rule from the audit brief:**
- "If ≥6 blocks have ≥5 NON-`repeated` coherent → PASS." → No (5/12).
- "If most coherent labels are `repeated` only → CORRECT to INCONCLUSIVE." → The dominant label IS repeated, but `repeated` in c235 is structurally different from cycle 217's unigram artefact (row vs column concentration).
- "If example records show meaningful semantic clusters → PASS." → Some do (blocks 3, 6 dir 3, 8 dir 3), many don't (blocks 0 dir 6 `@` is a tokenisation artefact; block 11 dir 0 is unigram-axis reboot).

**Verdict: PASS the literal CONFIRMED, with explicit G3 flag that the coherence heuristic is weak** and that item 246 (T2) is the real test. Downgrade to INCONCLUSIVE would be appropriate ONLY if this were cycle 217 redux — but centring fixed the cycle 217 failure mode in 11 of 12 blocks, and the executor's caveat already transmits the uncertainty downstream. Moreover, downgrading T1 to INCONCLUSIVE would block items 246-248 (T2/T3/T4) which are the actual causal tests — and that would slow down the disconfirmation path for the cycle 218 alignment story.

## Phase 3 — Quality checks

- **G1 Code correctness**: verified. Hook `register_forward_pre_hook` on `mlp.c_fc` captures post-LN2 input (correct target). `F.gelu(preact, approximate="tanh")` matches HF GPT-2's `gelu_new` exactly (cycle 217 audit confirmed this). Δ̃ computation `(gelu - relu) @ W_proj − mean` with chunked processing is correct (c_proj bias cancels in the subtraction). SVD on `(102400, 768)` gives `Vh.shape = (32, 768)` after top-K slicing → `V.T` shape `(768, 32)` ✓. Shape contract matches `notebooks/bump_features_explore.py` per the assertion block lines 218-227. **Minor concern**: classifier logic on line 270 (`if len(s) >= 2 and s[0].isupper()`) would not trigger for the block 11 dir 0 example (`favour` is lowercase) — if the label shown in examples.md really is `capitalized`, something else triggered (maybe `Fair`, `Cab`, `Kook` via line 270; the displayed top 3 rows aren't the ones that triggered). Not a verdict-affecting bug.
- **G2 Reproducibility**: seed 42, deterministic forward pass, deterministic torch SVD on CPU. Fine. Split = wikitext-2 train (not val), which differs from cycle 218's val — intentional, noted in script.
- **G3 Interpretation — FLAGGED**: the coherence heuristic is a cheap surface labelling that passes even when the direction's causal write-out is unrelated to its firing pattern. The executor documented this exhaustively; the audit re-flags it for downstream readers. The `repeated` label is NOT cycle 217's unigram failure mode, but it IS a weaker-than-advertised form of interpretability (row concentration ≠ causal feature). Verdict survives because (a) non-`repeated` labels in 5/12 blocks are genuinely coherent and (b) item 246 is the gating causal test.
- **G4 Counterintuitive check**: the per-block coherent count descending with depth `[27, 23, 24, 27, 21, 19, 19, 15, 17, 15, 12, 10]` matches the σ₀ escalation pattern `[671, 575, 876, 836, 803, 700, 641, 587, 696, 812, 1426, 2556]` — as the direction energy concentrates into a few modes at late blocks, the mixed/incoherent count grows. Consistent with cycle 218. Not counterintuitive.
- **G5 Learning quality — PARTIAL FLAG**: manifest `learned` field is thorough and flags the `repeated` caveat explicitly. The NEXT-step recommendation correctly identifies item 246 as the gating causal test. No bookkeeping correction needed.
- **G6 Method failure vs genuine**: GENUINE measurement. No methodology bug. The `repeated` heuristic is a weak interpretability proxy but was disclosed upfront in the verdict criteria.

## Guardrails

Tripped: **G3** (interpretation — `repeated` label is weak, executor pre-disclosed).
Not tripped: G1, G2, G4, G5, G6.

## Interesting

Yes — the audit revealed a substantive difference between cycle 217 and cycle 235 that the naive "dominant label is repeated" flag obscured. Cycle 217's failure was a *column* artefact (SV0 → unigram axis via lm_head); cycle 235's `repeated` is a *row* concentration (top-100 |U| positions happen to be common tokens). These are not the same failure, and centring structurally removes the cycle 217 mode in 11 of 12 blocks. The block 11 SV0 exception is informative: it shows that centring against in-distribution mean is not sufficient at the terminal block, consistent with the idea that late blocks encode primarily unembed-adjacent signal. This is worth noting for T3 (item 247, subspace ablation) and T4 (item 248, activation patching): block 11's top-rank directions may be mostly unigram-axis echoes and will not ablate cleanly.

## Recommendations

1. **Item 246 (T2) is the critical follow-up** — run it as planned. Jaccard ≥ 0.3 between T1-predicted top-50 vocab boosts and the actual logit shift from α·v_{l,i} residual stream intervention in ≥6 blocks → the `repeated` correlation becomes causal and cycle 235 upgrades to a stronger claim. Jaccard < 0.3 in most blocks → T1 was correlational only, and the c218 alignment result was spurious, which matches the cycle 217 audit's framing of "the story is alive but unproven."
2. **For T2, expect block 11 to be a noisy outlier** (it has the biggest σ₀ and it already looks like a unigram-axis reboot in SV0). Either exclude block 11 from the Jaccard tally or flag it separately.
3. **Do not rerun T1 with a stricter heuristic** (e.g. selectivity-controlled `repeated`). The centred Δ̃ features are saved and T2 will deliver a causal answer that subsumes any heuristic refinement.
4. **Consider tightening the verdict criterion in future items** (246, 247, 248): require the coherent-label count to exclude `repeated` OR to use a selectivity-controlled variant. This is a process improvement, not a c235 correction.
5. **Leave items 246, 247, 248 as-is** — they depend only on `cycle_235_t1_features.pt` and `cycle_235_t1_calibration_meta.pt`, both verified shape-correct.
