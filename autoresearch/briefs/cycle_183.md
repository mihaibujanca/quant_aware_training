# Item
**193. Matched-budget distillation control baseline — GPT-2 FFN k=4** [EMPIRICAL]

# Context
Current best recipes:
- topk+lstsq+corrections: PPL ~59 (1.6M + 49K correction params)
- topk+lstsq+polish: PPL ~70.7 (1.58M params)

Item 193 is the missing control baseline. Program.md states: "distillation only as a control baseline" — we've never run matched-budget KL distillation to verify our geometric recipe actually beats naive student SGD at same compute/param budget.

Cycle 182 (inconclusive): iterative ReLU swap yielded 37.0 PPL vs float 22.7 (+63%); constant-bias insufficient.

# Dependencies
None. ~10 min compute (500 Adam steps × 4 blocks).

# Key Files
- **Experiment**: `experiments/distillation_baseline_cycle183.py` (pre-written)
- **Baseline recipes**: cycle 170/171 topk+lstsq results (PPL ~70)
- **Status**: `autoresearch/STATUS.md` (cycle 182 summary)
- **Program**: `autoresearch/program.md` (control baseline rationale)

# Constraints
- **Budget parity**: 500 Adam steps = polish budget (not SGD full training)
- **Architecture**: 768→512→768 GELU (~787K params per student)
- **Blocks**: [1,4,7,11] (standard 4-block set)
- **Data**: 2048 OWT tokens, 4 epochs (same as topk+polish)
- **Eval metric**: stacked PPL (4-block swap)

# Handoff
Run distillation_baseline_cycle183.py. Report:
- Single-block swap PPL (4 blocks separately)
- Stacked 4-block PPL
- Total param count vs topk+lstsq
- Verdict: beat topk+polish by ≥3 PPL (confirm), match/lose (refute), or within ±3 (inconclusive)

Compare directly to cycle 170/171 topk+lstsq+polish baseline (PPL ~70.7).
