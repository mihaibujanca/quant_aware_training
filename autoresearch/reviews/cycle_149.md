# Review — Cycle 149 (Item 159)

## Checklist
- [x] Dependencies: none required, builds on established findings (cycles 138, 148)
- [x] Blocks [1,4,7,11] at 6x match all prior stacking experiments
- [x] SVD init verification step included (Frobenius check + step-0 PPL sanity)
- [x] Topk baseline should reproduce cycle 138 numbers (~70-71 PPL)
- [x] Intermediate eval checkpoints at [500,1000,1500,2000] give learning dynamics
- [x] Seed 42 only for screening — appropriate for a 45-min run

## Risks
- **Param count mismatch**: topk baseline trains full 3072x768 c_proj (9.4M params) while factorization trains rank-64 (986K). This is inherent to the comparison — the whole point of factorization is fewer params. Plan acknowledges this and mitigates via SVD init. The fair comparison is: can rank-64 joint training beat rank-512 topk+polish? If yes, the training protocol matters more than param count. OK.
- **Conv1D transposition**: GPT-2 uses Conv1D where weight shape is (in, out), not (out, in). SVD init MUST account for this. Plan mentions checking but should be explicit: `c_proj.weight` is shape (3072, 768), so SVD operates on (3072, 768) directly. The implementation must verify this.
- **Cosine schedule for (a)/(b) vs no schedule for (c)**: condition (c) uses flat lr to match historical protocol. This is correct — changing the baseline would invalidate comparison. But it means any (a)/(b) advantage could partly come from the schedule. Acceptable for screening; if confirmed, ablate schedule in follow-up.

## Verdict
**PASS** — well-motivated experiment attacking the core barrier. Clean design with appropriate controls and sanity checks. The param count asymmetry is a known limitation, not a flaw. Ready to run.
