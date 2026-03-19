# Cycle 005 Live Summary

- Input dir: `runs/nanochat_single_block_localization_overnight_main`
- Records: `5976` total, `5976` successful
- Teacher: `gpt2` (gpt2)
- Proxy dataset label: `proxy_shakespeare`
- Teacher perplexity: `122.350`
- Selected layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`
- Budgets: `[0.125, 0.0625]` with `max_hidden=512`

## Heartbeat
- Updated: `2026-03-16T17:55:52+00:00`
- Repeat: `0`
- Running: layer `4`, fraction `0.0625`, method `distilled`
- Status: `running`

## Item 10 Progress
- `>=4x` geometric swaps observed: `24`
- Current count meeting the agenda's success condition: `22`

## Best Geometric Swaps So Far
- layer `7` frac `0.0625` ppl `121.581` rel_ppl `-0.629%` top1 `0.817`
- layer `5` frac `0.0625` ppl `122.500` rel_ppl `+0.123%` top1 `0.809`
- layer `4` frac `0.0625` ppl `123.132` rel_ppl `+0.639%` top1 `0.812`
- layer `1` frac `0.0625` ppl `123.256` rel_ppl `+0.741%` top1 `0.844`
- layer `3` frac `0.0625` ppl `123.570` rel_ppl `+0.997%` top1 `0.811`
- layer `6` frac `0.0625` ppl `123.573` rel_ppl `+1.000%` top1 `0.812`
- layer `2` frac `0.0625` ppl `123.749` rel_ppl `+1.143%` top1 `0.817`
- layer `9` frac `0.0625` ppl `124.350` rel_ppl `+1.635%` top1 `0.798`

## Geometric vs Distilled Pairs
- layer `7` frac `0.0625` geo_rel `-0.629%` dist_rel `+2.205%` geo_ppl `121.581` dist_ppl `125.048`
- layer `5` frac `0.0625` geo_rel `+0.123%` dist_rel `+0.733%` geo_ppl `122.500` dist_ppl `123.246`
- layer `4` frac `0.0625` geo_rel `+0.639%` dist_rel `+1.807%` geo_ppl `123.132` dist_ppl `124.560`
- layer `1` frac `0.0625` geo_rel `+0.741%` dist_rel `+3.038%` geo_ppl `123.256` dist_ppl `126.067`
- layer `3` frac `0.0625` geo_rel `+0.997%` dist_rel `+3.656%` geo_ppl `123.570` dist_ppl `126.823`
- layer `6` frac `0.0625` geo_rel `+1.000%` dist_rel `+5.453%` geo_ppl `123.573` dist_ppl `129.021`
- layer `2` frac `0.0625` geo_rel `+1.143%` dist_rel `+3.811%` geo_ppl `123.749` dist_ppl `127.013`
- layer `9` frac `0.0625` geo_rel `+1.635%` dist_rel `+3.967%` geo_ppl `124.350` dist_ppl `127.204`
- layer `10` frac `0.0625` geo_rel `+1.810%` dist_rel `+3.352%` geo_ppl `124.564` dist_ppl `126.451`
- layer `8` frac `0.0625` geo_rel `+1.953%` dist_rel `+3.463%` geo_ppl `124.739` dist_ppl `126.587`
