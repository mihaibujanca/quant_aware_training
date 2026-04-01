# Cycle 187 Implementation Plan: Layer-drop baseline — GPT-2 FFN skip blocks [1,4,7,11]

## Config & Constants
```
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
SEED = 42
DEVICE = "cpu" (or "cuda" if available)
TARGET_BLOCKS = [1, 4, 7, 11]
ROOT = /Users/mihai/Projects/quant_aware_training (project root)
DATA_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_PATH = ROOT / "autoresearch" / "results" / "cycle_187_layer_drop.json"
EXP_PATH = ROOT / "experiments" / "layer_drop_cycle187.py"
```

## Implementation Steps

### Step 1: Load Data
- Read `openwebtext_sample.txt`
- Tokenize using `transformers.GPT2Tokenizer.from_pretrained("gpt2")`
- Create 200 sequences of length 128 (pad/truncate as needed)
- Random seed: 42 for reproducibility

### Step 2: Load GPT-2
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")
```

### Step 3: Baseline PPL
- Forward pass all 200 sequences through unmodified GPT-2
- Compute cross-entropy loss (averaged over tokens)
- Compute PPL = exp(loss)
- Store as `baseline_ppl`

### Step 4: Single-Block Drop Evals
For each block in range [0, 11]:
1. Save reference to `model.transformer.h[block_id].mlp`
2. Create wrapper that replaces `.mlp.forward()` with a zero-pass:
   ```python
   def zero_mlp(self, x):
       return torch.zeros_like(x)

   original_forward = model.transformer.h[block_id].mlp.forward
   model.transformer.h[block_id].mlp.forward = lambda x: torch.zeros_like(x)
   ```
3. Forward pass all 200 sequences
4. Compute PPL
5. Restore: `model.transformer.h[block_id].mlp.forward = original_forward`
6. Store result as `{block_id: ppl}`

### Step 5: Multi-Block Drop Eval
- Apply zero-pass to all blocks in TARGET_BLOCKS = [1, 4, 7, 11] simultaneously
- Forward pass all 200 sequences
- Compute PPL
- Restore all blocks
- Store result as `multi_block_ppl`

### Step 6: Compute Metrics
- For each block: `damage[block_id] = ppl[block_id] - baseline_ppl`
- Rank blocks by damage (ascending = least harmful first)

## Output JSON Schema
```json
{
  "cycle": 187,
  "model": "gpt2",
  "eval_config": {
    "num_sequences": 200,
    "sequence_length": 128,
    "batch_size": 4,
    "seed": 42,
    "device": "cuda" | "cpu"
  },
  "baseline_ppl": 45.32,
  "single_block_drops": {
    "block_0": {"ppl": 48.12, "damage": 2.80},
    "block_1": {"ppl": 52.01, "damage": 6.69},
    ...
    "block_11": {"ppl": 46.78, "damage": 1.46}
  },
  "multi_block_drop": {
    "blocks": [1, 4, 7, 11],
    "ppl": 75.23,
    "damage": 29.91
  },
  "block_ranking": [
    {"block": 11, "damage": 1.46},
    {"block": 0, "damage": 2.80},
    ...
  ],
  "droppable_blocks": [11, 0, 3],
  "verdict": "confirmed" | "refuted" | "inconclusive"
}
```

## Verdict Logic
```python
def compute_verdict(single_block_ppl_values):
    """
    Confirmed: ≥3 blocks have PPL < 40 above baseline
    Refuted: all blocks have PPL > baseline + 60
    Inconclusive: 1-2 blocks droppable
    """
    baseline = ppl_values["baseline"]
    damage_values = [ppl - baseline for ppl in single_block_ppl_values.values()]
    droppable = sum(1 for d in damage_values if d < 40)

    if droppable >= 3:
        return "confirmed"
    elif all(d > 60 for d in damage_values):
        return "refuted"
    else:
        return "inconclusive"
```

## Execution

### Run Command
```bash
cd /Users/mihai/Projects/quant_aware_training
python experiments/layer_drop_cycle187.py
```

### Commit Command
After results are generated and validated:
```bash
cd /Users/mihai/Projects/quant_aware_training
git add autoresearch/briefs/cycle_187.md autoresearch/plans/cycle_187.md experiments/layer_drop_cycle187.py autoresearch/results/cycle_187_layer_drop.json
git commit -m "[autoresearch] cycle 187: layer-drop baseline for GPT-2 FFN blocks [1,4,7,11]"
```

## Validation
After running:
1. Check JSON is valid: `python3 -c "import json; json.load(open('autoresearch/results/cycle_187_layer_drop.json'))"`
2. Check verdict is in {confirmed, refuted, inconclusive}
3. Verify baseline_ppl is reasonable for GPT-2 on openwebtext (typically 30-50)
4. Verify multi_block_drop.ppl > baseline_ppl (dropping blocks should increase loss)
5. No single-block PPL should be NaN or infinite

## Notes
- Use `.eval()` and `torch.no_grad()` for all forward passes
- Batch processing with BATCH_SIZE=4 to manage memory
- No gradient computation needed
- Hook-based approach (monkey-patching `mlp.forward`) is simpler than subclassing and safer than context managers for temporary changes
