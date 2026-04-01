# Cycle 187: Layer-drop baseline — GPT-2 FFN skip blocks [1,4,7,11]

**Hypothesis:** Some transformer blocks contribute negligibly to task performance. Dropping their FFN output entirely (residual passthrough) should reveal redundancy. Blocks ranked by single-block-drop PPL damage will correlate with compression difficulty from cycle 005.

**Method:**
1. Load GPT-2 (pretrained, 12 blocks, 768 hidden, 3072 FFN hidden)
2. Evaluate teacher baseline PPL on 200 sequences, length 128, from openwebtext sample
3. For each of 12 blocks: zero the MLP output (residual passthrough), evaluate PPL, restore
4. Drop all 4 target blocks [1,4,7,11] simultaneously, evaluate PPL
5. Report single-block and multi-block results

**Criteria:**
- **Confirmed**: ≥3 blocks have single-drop PPL <40 (task remains viable after removing that block's computation)
- **Refuted**: all single-drop PPL >60 (no easy drops; all blocks essential)
- **Inconclusive**: 1-2 blocks droppable (weak signal)

**Expected outcome:**
Identify which FFN blocks are redundant or low-impact. Lower-ranked blocks (by drop damage) are candidates for more aggressive compression; high-damage blocks warrant full-rank preservation.

**Runtime:** ~5 min (12 single-block evals + 1 baseline + 1 multi-block = 14 passes, ~200 seqs each)

**Success**: Clear ranking of block importance; enables prioritization for cycle 188+ compression designs.
