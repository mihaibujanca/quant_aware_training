# Cycle 217 — Item 231: Per-token Δ = GELU_FFN − ReLU_FFN

**Verdict:** CONFIRMED

## Headline
- Median rank_95 across 12 blocks: **1** (max 132)
- Blocks with rank_95 ≤ 64: **9 / 12** (low_rank criterion: ≥ 6)
- Blocks with interpretable top SVs (word-token heuristic): **12 / 12**

## Per-block

| Block | rank_95 | rank_99 | ||Δ||_mean | TI/TD | SV0 top boost |
|------:|--------:|--------:|-----------:|------:|:--------------|
| 0 | 10 | 322 | 21.696 | 0.9224 | ' challeng' 'names' ' the' |
| 1 | 132 | 492 | 17.504 | 0.8680 | ',' ' the' ' and' |
| 2 | 1 | 173 | 37.876 | 0.9650 | ' guiActiveUn' '���' 'cloneembedreportprint' |
| 3 | 1 | 100 | 44.643 | 0.9751 | '���' ' guiActiveUn' 'inventoryQuantity' |
| 4 | 1 | 107 | 45.475 | 0.9737 | ',' ' the' ' and' |
| 5 | 1 | 173 | 43.502 | 0.9679 | ' the' ',' ' a' |
| 6 | 1 | 278 | 39.185 | 0.9542 | ' the' ',' ' and' |
| 7 | 8 | 353 | 37.391 | 0.9413 | ' the' ',' ' a' |
| 8 | 73 | 430 | 34.707 | 0.9109 | 'inventoryQuantity' 'DragonMagazine' 'quickShipAvailable' |
| 9 | 110 | 461 | 37.077 | 0.8978 | ',' ' the' '-' |
| 10 | 1 | 323 | 61.937 | 0.9496 | ',' ' the' '-' |
| 11 | 1 | 1 | 153.284 | 0.9883 | 'DragonMagazine' 'natureconservancy' 'cloneembedreportprint' |

## Cross-block alignment — Δ_l top SVs vs c_fc_{l+1} left SVs (top-20)

| l→l+1 | cos_top3_mean | cos_top20_mean | random_baseline_top3_mean |
|:------|-------------:|--------------:|--------------------------:|
| 0→1 | 0.4122 | 0.2402 | 0.1580 |
| 1→2 | 0.2718 | 0.2270 | 0.1549 |
| 2→3 | 0.3640 | 0.2487 | 0.1685 |
| 3→4 | 0.2830 | 0.2633 | 0.1722 |
| 4→5 | 0.3414 | 0.3495 | 0.1760 |
| 5→6 | 0.3716 | 0.3327 | 0.1759 |
| 6→7 | 0.3757 | 0.3422 | 0.1785 |
| 7→8 | 0.4417 | 0.3620 | 0.1674 |
| 8→9 | 0.5564 | 0.3287 | 0.1686 |
| 9→10 | 0.3740 | 0.2644 | 0.1682 |
| 10→11 | 0.2671 | 0.2389 | 0.1637 |
| **mean** | **0.3690** | **0.2907** | **0.1683** |

Cycle 216 W_down→c_fc top-5 principal cos mean baseline: **0.649** (mid-blocks 4-7 hit 0.70-0.78).

## Striking blocks
### Block 2 — rank_95=1, TI/TD=0.9650, ||Δ||=37.876
- **SV0** (σ=5978.79)
  - boost: [' guiActiveUn', '���', 'cloneembedreportprint', '\x06', '\x1a', ' RandomRedditor', '�', '\x05', '\x0e', '\x1e']
  - supp:  [',', ' the', ' and', '.', '-', ' in', ' a', '\n', ' to', ' (']
- **SV1** (σ=230.64)
  - boost: ['pace', ' oriented', ' challeng', ' afloat', 'oppable', ' Royale', ' Solitaire', 'etheus', ' seiz', 'prep']
  - supp:  ['ighter', 'uries', 'inger', ' veto', 'ere', ' Guns', 'sonian', 'old', ' lectures', ' wildfires']
- **SV2** (σ=190.38)
  - boost: ['channelAvailability', ' encour', 'wordpress', ' Demonic', 'demon', 'mosp', 'recomm', 'ixel', 'onge', 'BN']
  - supp:  ['itiveness', 'ifiers', 'iture', 'ase', '>,', ' landmarks', 'ifier', ' Board', 'iary', ' liberties']

### Block 3 — rank_95=1, TI/TD=0.9751, ||Δ||=44.643
- **SV0** (σ=7075.79)
  - boost: ['���', ' guiActiveUn', 'inventoryQuantity', 'quickShipAvailable', 'cloneembedreportprint', '�醒', '\x06', '\x1a', '�', ' RandomRedditor']
  - supp:  [',', ' the', ' and', '-', ' a', '.', ' in', ' to', ' "', '\n']
- **SV1** (σ=209.55)
  - boost: [' Sov', ' [&', ' Mov', ' nodd', 'pring', ' Eag', ' Fiat', 'Sov', ' promot', 'itud']
  - supp:  [' scapego', 'icken', ' groom', 'hal', ' QUEST', ' shooters', '=-=-=-=-=-=-=-=-', ' quest', '":{"', 'displayText']
- **SV2** (σ=175.38)
  - boost: [' aisle', ' habitual', ' profession', ' disob', ' Moz', ' voice', ' economy', ' circulation', ' mouth', ' subscribe']
  - supp:  ['hai', 'abouts', 'nesses', 'Leaks', 'inctions', 'ocation', 'uld', 'sels', 'oche', 'warts']

### Block 4 — rank_95=1, TI/TD=0.9737, ||Δ||=45.475
- **SV0** (σ=7207.23)
  - boost: [',', ' the', ' and', '.', ' in', ' a', '-', ' to', ' (', '\n']
  - supp:  ['inventoryQuantity', 'quickShipAvailable', '���', ' guiActiveUn', ' Flavoring', 'cloneembedreportprint', '\x06', '�', '\x1a', ' RandomRedditor']
- **SV1** (σ=245.16)
  - boost: [' at', ' elect', ' Ore', '—', 'gr', ' and', 'st', 'ind', ',', '.']
  - supp:  ['76561', '��', 'ocrates', 'quickShipAvailable', 'ewitness', 'umenthal', 'ゴン', 'ikhail', ' Metatron', ' detrim']
- **SV2** (σ=192.48)
  - boost: ['inous', 'ussion', 'ttes', 'ifest', 'aign', 'opsis', 'anche', 'yond', 'ipt', 'lege']
  - supp:  [' Braz', 'Monitor', ' weekends', ' archived', ' mimic', ' monitors', 'Experts', ' Notre', ' Indeed', ' CTRL']

### Block 9 — rank_95=110, TI/TD=0.8978, ||Δ||=37.077
- **SV0** (σ=5642.17)
  - boost: [',', ' the', '-', ' and', ' in', ' a', '.', ' to', ' (', ' "']
  - supp:  ['isSpecialOrderable', 'quickShipAvailable', 'inventoryQuantity', 'DragonMagazine', 'soDeliveryDate', 'ゴン', 'renheit', '��', 'osponsors', 'ciating']
- **SV1** (σ=341.19)
  - boost: ['.」', '!.', '.</', '.�', '.}', '!,', ".''.", ".''", ';', ',—']
  - supp:  [' provisional', 'assing', 'ptive', ' CBS', ' PT', ' Paul', ' Cull', ' UNC', 'aband', ' BART']
- **SV2** (σ=262.51)
  - boost: [' sugg', 'Cause', ' confir', 'ouver', ' princip', ' incent', ' streng', ' neigh', 'Finish', ' Deposit']
  - supp:  [' the', ' his', 'the', ' another', ' it', ' both', ' its', ' these', ' their', 'his']

### Block 1 — rank_95=132, TI/TD=0.8680, ||Δ||=17.504
- **SV0** (σ=2625.42)
  - boost: [',', ' the', ' and', '-', '.', ' a', '\n', ' in', ' "', ' to']
  - supp:  ['���', ' guiActiveUn', 'cloneembedreportprint', 'DeliveryDate', '\x06', '\x1a', '�', ' RandomRedditor', '\x05', '�']
- **SV1** (σ=247.84)
  - boost: ['surv', 'published', 'nesota', ' Jill', 'gotten', 'micro', 'appa', 'ammy', 'EVA', 'doi']
  - supp:  ['iors', ' istg', 'rodu', 'ibilities', 'isine', ' Skies', ' une', ' guiActiveUn', 'iations', 'fires']
- **SV2** (σ=244.65)
  - boost: ['theless', 'ラン', '*=-', 'izoph', ' Cosponsors', 'SPONSORED', 'govtrack', 'BuyableInstoreAndOnline', ' Canaver', 'taboola']
  - supp:  [',', '.', ' the', ' and', '-', ' (', ' a', ' in', ' "', '\n']

## Plain-language summary
The GELU−ReLU difference Δ(x) is extremely low-rank: median rank_95 = 1 across the 12 blocks (9/12 have rank_95 ≤ 64; the only exceptions are blocks 1, 8, 9 at rank_95 = 73-132). **Important caveat**: the very high TI/TD ratio (0.87-0.99) shows Δ is dominated by a *token-independent mean direction*, so rank_95 = 1 mostly reflects that one average vector. SV0 in most blocks boosts common punctuation / stopwords (`,`, ` the`, ` and`, `.`) and suppresses tokenizer-artefact tokens (`guiActiveUn`, `inventoryQuantity`, `cloneembedreportprint`) — consistent with a constant bias rather than per-token routing. The token-dependent structure that *does* vary per-token lives in SV1+ (σ ≈ 200-350, much smaller than σ0 ≈ 3k-7k) and shows weaker but still interpretable clusters (e.g. block 9 SV1: sentence-ending punctuation cluster `.」 !. .</ .� .}`). Cross-block principal cosines between Δ_l's top SVs and c_fc_{l+1}'s top-20 left SVs average **0.369** (top-3) / **0.291** (top-20), clearly above the random baseline of **0.168** — Δ does write into directions the next block's c_fc reads, though the alignment is weaker than cycle 216's W_down→c_fc baseline of 0.649. **Verdict CONFIRMED** by the plan's criteria, but the interpretation is "Δ is mostly a constant bias per block plus a small token-dependent residual", not "Δ is a rich token-dependent low-rank message". Follow-ups 232-234 should focus on Δ − mean(Δ) to isolate the actually-per-token component.
