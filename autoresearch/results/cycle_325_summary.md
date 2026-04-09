# Cycle 325 — Item 338: Canonical Checkpoint Was Already lrelu2-Trained

## Finding

The canonical checkpoint `13L_640d_322_7k_wd70_batch20k_best.npz` is **lrelu2-trained**.

Verification:
```
python3 -c "import json; d=json.load(open('.../13L_640d_322_7k_wd70_batch20k_summary.json')); print(d['config']['mlp_act'])"
# → lrelu2
```

Field extracted: `config.mlp_act = "lrelu2"` in `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_summary.json`.

## Verdict: MOOT-already-lrelu2-trained

Item 338's premise — that cycles 318/319 measured a ReLU²-trained checkpoint and therefore we need to run again on a true lrelu2-trained checkpoint — **is false**. The checkpoint was always lrelu2-trained. Cycles 318 and 319 mislabeled it as "ReLU²-trained" in their summaries and plan docs, but they measured the correct model.

**Consequence**: the LeakyReLU²-trained checkpoint census requested by item 338 was already performed in cycles 318 and 319. No new training or measurement is needed.

## Errata

- `autoresearch/results/cycle_318_summary.md` refers to the checkpoint as "ReLU² checkpoint" — should read "lrelu2-trained checkpoint".
- `autoresearch/results/cycle_319_summary.md` header line reads "Curvature residual decomposition on parameter-golf 13L_640d ReLU² checkpoint" — should read "lrelu2-trained checkpoint".
- `autoresearch/cross_project/parameter_golf.md` §1 table describes the canonical checkpoint as "ReLU² activation (no `mlp.gate.weight`)" — this refers to architectural absence of a gating weight (correct), but the activation function is lrelu2, not relu2.

## Cycle 318 eff_rank profile (IS the lrelu2-trained census)

Checkpoint: `13L_640d_322_7k_wd70_batch20k_best.npz` (lrelu2-trained, 13 blocks, 640 dim)
n_tokens: 25600

| block | eff_rank | rank_95 | rank_99 |
|---|---|---|---|
| 0 | 187.6 | 378 | 542 |
| 1 | 160.1 | 367 | 534 |
| 2 | 86.0 | 314 | 499 |
| 3 | 4.9 | 94 | 309 |
| 4 | 38.3 | 198 | 396 |
| 5 | 40.0 | 201 | 401 |
| 6 | 28.2 | 171 | 365 |
| 7 | 23.2 | 144 | 323 |
| 8 | 30.8 | 164 | 346 |
| 9 | 28.8 | 168 | 358 |
| 10 | 21.6 | 150 | 326 |
| 11 | 13.4 | 114 | 280 |
| 12 | 6.9 | 38 | 92 |

Block 3 collapse (eff_rank=4.9) and block 12 late-stack collapse (eff_rank=6.9, rank_95=38) are features of the **lrelu2-trained** model.

## Cycle 319 curvature residual baselines (IS the lrelu2-trained curvature census)

(lrelu², relu²) profile — blocks with rank_95 ≤ 50: **b4=18, b5=33, b12=21** (localized).
(relu², relu) and (gelu, relu) panels also measured on the same checkpoint.
See `autoresearch/results/cycle_319_summary.md` for full tables.

## Items depending on 338

Items 340, 341, 342, 343 that list "Depends on: 338" should note that the lrelu2-trained checkpoint IS the canonical checkpoint already. No new checkpoint path is needed.
