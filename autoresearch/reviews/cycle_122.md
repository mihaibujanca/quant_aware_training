# Cycle 122 Review (Round 2) — Train MoE Testbed (Dropout Fix)

Item 138. Reviewer: design-review agent.

## Round 1 issue status

**hidden_dropout config field**: FIXED. The plan now correctly identifies that
`MixtralConfig` has no FFN dropout field and injects dropout via monkey-patch on
`MixtralSparseMoeBlock.forward`. The verification gate (stochastic forward check)
ensures dropout is active before training proceeds. This was the sole blocking issue
from round 1.

## R1: Premise validity — OK

Monkey-patch approach is correct in principle: wrap the MoE block forward to apply
`nn.Dropout(p)` on the expert output. The `make_wrapper` closure factory avoids the
late-binding pitfall. `attention_dropout=0.1` in config is valid (confirmed field
exists). Gradient accumulation (effective batch 4096 tokens) and wider lr sweep are
reasonable additions.

## R2: Measurement soundness — OK

Unchanged from round 1. Val PPL on WikiText-2, eval every 2M tokens, sweep at 1 epoch.

## R3: Capacity — OK

Unchanged. ~65 min CPU, implementation type.

## R4: Method-architecture alignment — MINOR FIX NEEDED

The monkey-patch wrapper unpacks a tuple:

```python
out, router_logits = orig(*args, **kwargs)
return d(out), router_logits
```

In transformers 5.2.0, `MixtralSparseMoeBlock.forward` returns a **single tensor**,
not a tuple. The `gate()` call returns `(_, top_k_weights, top_k_index)` internally
but the block's `forward` only returns `hidden_states`. The caller
(`MixtralDecoderLayer.forward`) does `hidden_states = self.mlp(hidden_states)` —
single tensor in, single tensor out.

The wrapper must be:

```python
def make_wrapper(orig, d):
    def wrapper(*args, **kwargs):
        out = orig(*args, **kwargs)
        return d(out)
    return wrapper
```

This would crash at the verification gate (not silently), so it cannot cause a
silent failure. But it will abort the script before training, wasting the setup.

**Severity**: minor (caught by existing verification gate, 1-line fix).

## R5: Known pitfall scan — OK

No applicable pitfalls. The verification gate eliminates the MP6 risk flagged in
round 1.

## Summary

| Check | Status |
|-------|--------|
| R1 Premise | OK (monkey-patch approach is sound) |
| R2 Measurement | OK |
| R3 Capacity | OK |
| R4 Alignment | MINOR — wrapper unpacks tuple but forward returns single tensor |
| R5 Pitfalls | OK (verification gate covers silent-failure risk) |

## Verdict: ACCEPT with minor fix

Fix the wrapper return signature (single tensor, not tuple). One-line change, no
structural revision needed. The verification gate would catch this at runtime anyway.
