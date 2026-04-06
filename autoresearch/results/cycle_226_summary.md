# Cycle 226 — Item 236: KD distillation (CORRECTED loss normalization)

**Verdict: INCONCLUSIVE**

_NOTE: best α=0 (pure CE wins) — KD provides no benefit; this is essentially cycle 213 reproduction. Marking INCONCLUSIVE because hypothesis 'KD beats CE' is not supported._

- Criteria (item 236): Confirmed ≤44.0, Refuted ≥47.0, else Inconclusive
- Fix: reshape `(B,T,V)` → `(B*T, V)` before `kl_div(reduction='batchmean')` so KD divides by per-token count (B·T) and matches CE scale.
- Config: BLOCKS=[1, 4, 7], K=512, LR=0.0001, τ=2.0
- Float teacher PPL: 53.895

## Phase 1 — α-sweep (SEED=42, 2000 steps, eval_every=200)

| α | init | best | best_step |
|---|---|---|---|
| 0.00 | 61.746 | 46.430 | 1200 |
| 0.50 | 61.746 | 47.096 | 2000 |
| 1.00 | 61.746 | 61.746 | 0 |

**Best α = 0.0** (best_ppl=46.430)

## Phase 2 — full run (SEEDS=[42, 137, 2024], 4000 steps, α=0.0)

| seed | init | best | best_step |
|---|---|---|---|
| 42 | 61.746 | 46.430 | 1200 |
| 137 | 61.746 | 46.321 | 1200 |
| 2024 | 61.746 | 46.342 | 1400 |
| **mean** | 61.746 | **46.364** | 1267 |
| **std**  | 0.000 | 0.047 | 94 |

## Comparison vs baselines

| training | loss | best PPL |
|---|---|---|
| W_out+W_fc | CE only (cycle 213) | 46.424 |
| W_out+W_fc | KD+CE BROKEN norm (cycle 224) | 60.938 |
| W_out+W_fc | KD+CE CORRECTED α=0.0 (this) | **46.364 ± 0.047** |

- Δ vs cycle 213 (CE same surface): **-0.060 PPL**
- Δ vs cycle 224 (broken KD):       **-14.574 PPL**
- Float ceiling: 53.895

## Per-seed trajectories

### Seed 42
- init=61.75, step1100=46.58, step2100=47.73, best=step1200@46.43, step3100=52.71, step4000=58.81
### Seed 137
- init=61.75, step1100=46.86, step2100=47.49, best=step1200@46.32, step3100=52.04, step4000=58.94
### Seed 2024
- init=61.75, step1100=46.81, step2100=47.75, best=step1400@46.34, step3100=52.61, step4000=58.93
