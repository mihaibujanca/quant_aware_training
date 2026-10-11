# Cycle 122 Plan: Train MoE Testbed — Tiny Mixtral (Dropout Fix)

Item 138. Retry cycle 121 with FFN dropout + wider lr sweep.

## Pre-flight

- P1 file: `experiments/moe_train_cycle122.py` (new, based on cycle 121)
- P2 no multi-seed needed (implementation type)
- P3 success: val PPL < 50 on WikiText-2
- P4 refuted: PPL > 50 at 30M tokens with dropout + lr sweep
- P5 wall time: ~65 min CPU (20 min sweep + 45 min full)
- P6 depends on: nothing
- P7 CPU run

## Changes from cycle 121

Copy `experiments/moe_train_cycle121.py` to `experiments/moe_train_cycle122.py`.

### 1. Inject FFN dropout via monkey-patch

`MixtralConfig` only has `attention_dropout`; there is no hidden/FFN dropout field.
Setting arbitrary kwargs is silently swallowed. Inject dropout explicitly:

```python
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

def inject_ffn_dropout(model, p=0.1):
    """Wrap each MoE block's forward to apply dropout after expert output."""
    for module in model.modules():
        if isinstance(module, MixtralSparseMoeBlock):
            drop = nn.Dropout(p)
            orig_forward = module.forward
            def make_wrapper(orig, d):
                def wrapper(*args, **kwargs):
                    out, router_logits = orig(*args, **kwargs)
                    return d(out), router_logits
                return wrapper
            module.forward = make_wrapper(orig_forward, drop)
```

Call `inject_ffn_dropout(model)` right after `make_model()`. Also set
`attention_dropout=0.1` in `MixtralConfig(...)`.

### 2. Dropout verification gate (MUST pass before training)

```python
def verify_dropout_active(model):
    model.train()
    x = torch.randint(0, VOCAB_SIZE, (1, 32), device=DEVICE)
    out1 = model(x).logits
    out2 = model(x).logits
    assert not torch.equal(out1, out2), "Dropout NOT active — outputs identical"
    print("PASS: FFN dropout verified active", flush=True)
```

Run this immediately after `inject_ffn_dropout`. If it fails, abort the script.

### 3. Gradient accumulation

`BATCH_SIZE = 2`, `GRAD_ACCUM = 8` (effective 16 seqs = 4096 tok/step).
Accumulate micro-steps before `optimizer.step()`. Adjust `steps_target` for
`tokens_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM`.

### 4. Wider lr sweep

`SWEEP_LRS = [3e-4, 5e-4, 1e-3, 2e-3]`, `SWEEP_TOKENS = 2_000_000` (1 full epoch).

### 5. Full training

`FULL_TOKENS = 30_000_000`, `EVAL_EVERY = 2_000_000`. No 60M extension.

### 6. Paths and verdict

```python
CKPT_PATH = EXPR_RESULTS_DIR / "cycle_122_tiny_mixtral.pt"
JSON_PATH = RESULTS_DIR / "cycle_122_moe_training.json"
outcome = "CONFIRMED" if effective_ppl < 50.0 else "REFUTED"
```

Save checkpoint only if CONFIRMED.

## Verification

- JSON: sweep_results (4 lrs), best_lr, effective_ppl, val_ppl_curve, outcome
- Dropout gate must print PASS before any training begins
- If CONFIRMED: checkpoint loads successfully
