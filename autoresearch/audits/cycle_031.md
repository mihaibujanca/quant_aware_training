# Cycle 031 Audit — Item 46: Grokking as Compression Catalyst (Modular Arithmetic v2)

## Audit Status: PASS

## Evidence Review

### Grokking detection
- Memorization epoch: 1000 (train=0.994, test=0.000) ✓
- Grokking epoch: 17500 (train=1.000, test=0.814) ✓
- Delay: 16500 epochs — genuine memorization→generalization phase transition ✓
- This satisfies the item criteria: train acc >99% then delayed test acc >80%

### Compression results
| Checkpoint | Teacher test acc | Student test acc |
|------------|-----------------|-----------------|
| pre_grokking (epoch 1000) | 0.000 | 0.000 |
| post_grokking (epoch 17500) | 0.814 | 0.002 |
| best_test (epoch 47500) | 0.956 | 0.002 |

- Compression gain = post - pre student acc = 0.002 - 0.000 = +0.001
- Well below 2% threshold → **REFUTED**

### Geometric signals
- Pattern count: 9409 throughout (constant = P² — every (a,b) pair gets unique activation pattern)
- Region-label purity: 1.000 throughout (trivially — each region has exactly one example)
- No geometric change visible through grokking (patterns and purity are fixed from epoch 500)

## Methodological Notes

**Why does distillation fail despite grokking?**
The modular arithmetic function requires the student to generalize to unseen (a,b) pairs. The student is trained only on the 40% training set. Pre-grokking teacher: 0% generalization, so student trivially gets 0%. Post-grokking teacher: the teacher has learned the algebraic structure (DFT-like embeddings), but the student, trained on only 3763 examples, cannot recover this structure at hidden_dim=32. The task requires O(p) embedding dimensions to generalize (Nanda et al.), which 32 hidden units may not have.

**Why are all patterns unique?**
With 128 neurons on 194-dimensional input for 9409 examples, near-maximum linear region density is expected. Every training example touches a unique activation pattern. This makes pattern count a vacuous signal for this architecture/dataset size.

**Is the experiment clean?**
- Task is canonical (Power et al. 2022 grokking setup)
- Seeds fixed, train/test split stable
- Pre-commit hooks fixed trailing whitespace in JSON
- Output files correctly named cycle_031 (bug fix applied)

## Verdict Confirmation
**REFUTED** — grokking criterion satisfied, but compression gain 0.001 << 2% threshold. This is consistent with prior refutation of item 39 (different task, same conclusion): grokking does not improve distillability from training data alone.

**Insight for agenda**: The distillation failure suggests the post-grokking algebraic structure is spread across all 128 hidden units in a way that cannot be recovered with 32 units. Future work: test whether the post-grokking teacher's internal representation can be compressed via SVD/low-rank decomposition (not distillation), since the DFT-based Fourier features used for grokking are known to be low-rank in the frequency domain.
