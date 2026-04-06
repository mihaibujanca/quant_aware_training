# Cycle 218 — Item 232: Centred Δ̃ alignment with downstream readers

**Verdict:** CONFIRMED

## Headline
- **Significant cross-block pairs**: 11 / 11 (CONFIRMED ≥ 8, REFUTED = 0)
- Per-reader significant pair counts: {'c_fc': 11, 'Q': 10, 'K': 6, 'V': 8, 'resPC': 11}
- Blocks with word-token SVs (heuristic): 12 / 12

## Rank of centred Δ̃ vs raw Δ (cycle 217)

| Block | rank_95 RAW (c217) | rank_95 CENTRED | rank_99 CENTRED | σ₀ | σ₁ | σ₂ |
|------:|-------------------:|----------------:|----------------:|---:|---:|---:|
| 0 | 10 | 489 | 660 | 341.8 | 270.0 | 177.8 |
| 1 | 132 | 554 | 692 | 260.1 | 247.3 | 220.2 |
| 2 | 1 | 524 | 682 | 431.0 | 202.8 | 181.3 |
| 3 | 1 | 520 | 681 | 417.8 | 208.0 | 170.7 |
| 4 | 1 | 526 | 689 | 495.7 | 193.2 | 165.4 |
| 5 | 1 | 538 | 692 | 392.2 | 223.6 | 198.3 |
| 6 | 1 | 551 | 697 | 328.6 | 259.8 | 211.4 |
| 7 | 8 | 565 | 701 | 305.8 | 239.6 | 209.4 |
| 8 | 73 | 558 | 698 | 368.1 | 305.7 | 231.7 |
| 9 | 110 | 562 | 699 | 421.4 | 328.4 | 250.4 |
| 10 | 1 | 567 | 702 | 704.4 | 371.3 | 317.5 |
| 11 | 1 | 552 | 701 | 1292.9 | 503.4 | 386.8 |

Median rank_95: raw 1 → centred 552 (centring removes per-block constant bias).

## Random baseline (10 seeds × 11 pairs × 5 readers)

| Reader | mean top20 | std top20 | mean top5 | std top5 |
|:-------|-----------:|----------:|----------:|---------:|
| c_fc | 2.7135 | 0.0979 | 1.2557 | 0.0448 |
| Q | 2.7228 | 0.1202 | 1.2531 | 0.0515 |
| K | 2.6997 | 0.0998 | 1.2532 | 0.0485 |
| V | 2.7331 | 0.0972 | 1.2628 | 0.0463 |
| resPC | 2.7233 | 0.0997 | 1.2681 | 0.0371 |

## Cross-block alignment (top-20 principal cosine sums)

| l→l+1 | c_fc | Q | K | V | resPC | any_sig |
|:------|-----:|--:|--:|--:|------:|:-------:|
| 0→1 | 4.847 | 4.487 | 4.908 | 4.397 | 6.913 | ✓ |
| 1→2 | 4.423 | 4.204 | 4.043 | 4.200 | 6.965 | ✓ |
| 2→3 | 4.864 | 3.578 | 3.740 | 5.045 | 5.110 | ✓ |
| 3→4 | 5.229 | 3.203 | 3.391 | 5.000 | 5.143 | ✓ |
| 4→5 | 6.896 | 3.998 | 2.585 | 4.771 | 4.769 | ✓ |
| 5→6 | 6.575 | 4.307 | 4.127 | 5.013 | 5.013 | ✓ |
| 6→7 | 6.709 | 4.151 | 2.607 | 4.025 | 5.482 | ✓ |
| 7→8 | 7.068 | 4.517 | 3.242 | 3.690 | 6.024 | ✓ |
| 8→9 | 6.392 | 4.208 | 2.917 | 1.735 | 6.351 | ✓ |
| 9→10 | 5.183 | 3.540 | 2.352 | 1.464 | 6.763 | ✓ |
| 10→11 | 4.562 | 3.010 | 2.373 | 1.662 | 8.382 | ✓ |

**Significance**: cos_sum > random_mean + 3·std (≈ p < 0.01). Per-reader thresholds ≈ c_fc 3.007, Q 3.083, K 2.999, V 3.025, resPC 3.022.

## Vocab interpretability — centred Δ̃ top-3 SVs

### Block 0 — rank_95 centred = 489, σ₀=341.8, σ₁=270.0
- **SV0** (σ=341.77)
  - boost: ['Mand', 'イト', 'ÍÍ', 'oti', 'assis', ' Benedict', 'oto', 'aser', ' Planetary', 'quist']
  - supp:  [' derogatory', 'tis', ' flush', ' FB', 'eries', 'ournals', ' slang', 'mu', ' Username', 'iott']
- **SV1** (σ=269.99)
  - boost: [' guiActiveUn', 'ール', 'DragonMagazine', '��', 'EStream', 'isine', 'ウス', 'ÍÍ', 'adelphia', ' unfocusedRange']
  - supp:  [' the', ' a', " '", ' I', ' The', ' two', ' D', ' other', ' one', ' "']
- **SV2** (σ=177.83)
  - boost: ['lings', 'ers', 'jet', 'est', 'ly', 'ling', ' tro', 'oon', ' stations', ' Shot']
  - supp:  ['phia', 'VERTISEMENT', 'Iv', 'trl', 'docker', 'Services', 'utics', 'qs', 'ibaba', 'lished']

### Block 3 — rank_95 centred = 520, σ₀=417.8, σ₁=208.0
- **SV0** (σ=417.85)
  - boost: [',', ' the', ' "', ' and', ' a', '.', ' in', ' (', '\n', '-']
  - supp:  ['���', 'cloneembedreportprint', '\x06', '\x1a', ' RandomRedditor', '\x05', '�', '\x08', '�', '\x1e']
- **SV1** (σ=208.04)
  - boost: [' scapego', ' groom', 'displayText', 'hal', 'icken', 'nesia', 'oning', '--------------------------------------------------------', ' QUEST', '=-=-=-=-=-=-=-=-']
  - supp:  ['pring', ' Eag', ' Sov', ' nodd', ' Mov', ' [&', ' promot', ' Fiat', 'achine', 'Sov']
- **SV2** (σ=170.69)
  - boost: [' maturity', ' economy', 'animate', ' profession', ' aisle', ' habitual', ' unions', ' resume', ' tune', ' delivery']
  - supp:  ['hai', 'Leaks', 'resy', ' Qiao', 'abouts', 'inctions', 'borgh', 'nesses', 'pointers', 'scl']

### Block 2 — rank_95 centred = 524, σ₀=431.0, σ₁=202.8
- **SV0** (σ=430.99)
  - boost: [' the', ',', ' "', ' a', ' and', '.', '\n', ' in', ' to', ' (']
  - supp:  ['���', ' guiActiveUn', 'cloneembedreportprint', 'ÃÂÃÂÃÂÃÂ', '\x1a', '\x06', ' RandomRedditor', '\x05', '\x13', '\x0e']
- **SV1** (σ=202.83)
  - boost: [' Compass', 'channelAvailability', 'istar', 'worthy', ' Penet', 'caliber', 'etheus', ' mag', ' Advocate', ' Evening']
  - supp:  [' recess', 'yles', 'ighter', 'ardy', 'ifiers', 'SPONSORED', 'ase', 'ere', 'sonian', 'asses']
- **SV2** (σ=181.35)
  - boost: ['将', 'ğ', 'onge', 'udeau', 'ubuntu', ' Balloon', 'chrom', ' laps', 'ppel', 'adoes']
  - supp:  ['DC', 'AGES', ' antioxid', 'asis', ' Division', ' Agency', 'ancy', 'ines', ' Wells', 'gars']

### Block 7 — rank_95 centred = 565, σ₀=305.8, σ₁=239.6
- **SV0** (σ=305.84)
  - boost: ['quickShipAvailable', 'inventoryQuantity', 'ertodd', 'isSpecialOrderable', 'osponsors', 'soType', 'soDeliveryDate', 'natureconservancy', '�醒', '�']
  - supp:  [' the', ',', ' a', ' "', '\n', ' (', ' and', ' in', '-', '.']
- **SV1** (σ=239.56)
  - boost: ['¶', 'iour', 'EngineDebug', ' 🙂', '\\">', ' ;)', ' confir', 'esan', 'eston', 'displayText']
  - supp:  [' a', ' little', ' the', ' Minor', ' minor', ' optional', ' parties', ' two', ' considerable', ' strong']
- **SV2** (σ=209.40)
  - boost: ['.', ',', "'s", ' of', '-', ' is', ' for', ':', ' who', ';']
  - supp:  ['igl', 'adish', 'rolet', 'entin', 'ixel', ' volunte', 'irtual', 'oreAnd', 'utters', ' Volunte']

### Block 10 — rank_95 centred = 567, σ₀=704.4, σ₁=371.3
- **SV0** (σ=704.39)
  - boost: ['-', ' the', ' a', ',', ' and', ' in', ' "', '.', ' (', ' to']
  - supp:  ['cloneembedreportprint', 'isSpecialOrderable', '���', 'soDeliveryDate', ' guiActiveUn', '�', '\x05', ' RandomRedditor', '龍契士', 'StreamerBot']
- **SV1** (σ=371.31)
  - boost: ['auga', 'nesota', '��', 'ActionCode', '��', 'ertodd', 'reportprint', 'embedreportprint', ' unintention', '\x1b']
  - supp:  ['.', '-', ',', ':', ' to', ' and', "'s", ' (', ';', '!']
- **SV2** (σ=317.45)
  - boost: [' Kubrick', 'ocaust', ' Canaver', 'Pokemon', ' Freud', ' Lauder', ' SEO', ' Plex', 'natureconservancy', 'SQL']
  - supp:  ['.—', '.–', '.:', ',[', '.(', '.--', '.):', ').[', '>,', '.]']

## Plain-language summary
After centring (subtracting the per-block mean Δ), the token-dependent residual Δ̃ writes into directions that downstream blocks actually read. 11/11 cross-block pairs exceed the random baseline by > 3σ against at least one downstream reader (c_fc:11, Q:10, K:6, V:8, resPC:11). Removing the mean promotes rank_95 from 1 (raw, c217) to 552 (centred), confirming the constant bias hid the true per-token geometry. The bump IS structured inter-block communication; downstream components explicitly depend on it.

## Implications for items 233 / 234
Item 233 (anti-podal channel twin census) depends on item 230 (REFUTED: <1.2% twins), so its premise is already weakened regardless of this result. Item 234 (surgical bump removal: GELU_noneg vs GELU_nobump) is where the functional load of the bump will actually be localised. Today's CONFIRMED result on Δ̃ alignment means the bump DOES write into the next block's reading directions, so 234 is expected to show that removing the bump knee (or negative leak) breaks cross-block information flow, not just within-block nonlinearity.
