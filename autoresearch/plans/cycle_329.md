# Cycle 329 Plan: LASER b11 spectrum preflight

## Step 0 (REQUIRED)
Read `autoresearch/cross_project/parameter_golf.md` for subprocess invocation pattern.

## Step 1: Spectrum check
Reuse `/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_laser_b12_spectrum.py` with `--target-block 11 --rank 13`.
The script is fully parameterized and works for any block.

Command:
```
/Users/mihai/Projects/parameter-golf/.venv/bin/python \
  /Users/mihai/Projects/parameter-golf/local_scripts/pgolf_laser_b12_spectrum.py \
  --checkpoint /Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz \
  --target-block 11 --rank 13
```

## Step 2: Gate decision
- top-13 energy < 10%: REFUTED-flat-spectrum, skip retrain, write manifest
- top-13 energy >= 10%: proceed to retrain (unlikely given b3 and b12 pattern)

## Verdict mapping
- REFUTED-flat-spectrum: top-13 energy < 10%
- CONFIRMED-LASER-transfers: retrain delta <= 0.001
- REFUTED-LASER-fails: retrain delta >= 0.01
