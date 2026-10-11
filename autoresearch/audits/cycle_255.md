# Cycle 255 Audit — Item 265

**Status: PASS** (interesting=true)

## Item (verbatim)
≤51 confirmed; ≥53 refuted; else inconclusive (based on min over subsets).

## Claimed vs verified
- Verdict claim: **CONFIRMED** — min best_mean=49.609 on [1,2,4,7].
- Phase 1 rule check: `min_best_mean_49.609 <= 51.0` → CONFIRMED. PASS.
- Refutation branch (`all best_mean >= 53.0`) correctly not triggered (max=50.712).

## Per-subset verification (exec_result ↔ manifest ↔ code)
| subset      | n | init   | best ± std       | best_step   | final@4000 |
|-------------|---|--------|------------------|-------------|------------|
| [1,2,4,7]   | 3 | 66.826 | 49.609 ± 0.080   | 2367 ± 205  | 51.259     |
| [1,2,7,8]   | 3 | 67.896 | 50.209 ± 0.011   | 1767 ± 47   | 54.332     |
| [1,4,7,8]   | 3 | 68.634 | 50.712 ± 0.053   | 1733 ± 94   | 54.409     |

Float held-out = 53.895. All 3 subsets beat float by 3.18–4.29 PPL (all ≤ 51.0). Per-seed values in manifest match exec_result line (e.g. [1,2,4,7]=[49.4973, 49.6783, 49.6515]).

## Phase 2 — is the signal real?
- 3 seeds per subset, 9 fits total. Wall 22.8 min < 30 min cap → no fallback.
- Within-subset std 0.011–0.080 PPL; across-subset spread 1.10 PPL (49.609→50.712). Subset ordering is statistically distinguishable — at 4-block scale block identity matters, unlike cycle 250's finding at 6-block.
- Init-PPL ordering (66.8 < 67.9 < 68.6) matches best-PPL ordering (49.6 < 50.2 < 50.7); Adam does not equalize differently-initialized subsets.
- Best-state restore round-trip clean (assert |restored-best|<1e-3 holds for all 9 seeds per code L192).
- Harder subsets overshoot more past best (final-best: +1.65 / +4.12 / +3.70), consistent with plateau-then-climb.

## Phase 3 — count-driven extrapolation
Monotonic blocks-dropped vs held-out PPL at K=512 topk+joint Adam:
- 6b ~59.5 → 5b 53.826 (c245) → 4b 49.609 (c255) → 3b 46.556 (c210 [1,4,7])
- Each block dropped buys ~3–6 PPL. 4-block recovers ~97% of the 5b→3b gap with 4/3× wider coverage than 3b.

## Bookkeeping
- commit=3fa68c7; bookkeeping=d60c995
- Artifacts present: json, summary.md, log, manifest, plan, experiment script.
- Verdict logic in code (L278-283) matches item 265 criteria verbatim.

**Interesting**: yes — first clean demonstration that every dropped block buys monotonic held-out PPL under a single recipe, and first evidence of block-identity sensitivity at 4-block scale.
