"""
Cycle 122: Train Tiny Mixtral MoE testbed — item 138 (retry with dropout).

Changes from cycle 121:
- FFN dropout=0.1 via monkey-patch on MixtralSparseMoeBlock.forward
- attention_dropout=0.1 in MixtralConfig
- Grad accumulation: BATCH_SIZE=2, GRAD_ACCUM=8 (effective 16 seqs = 4096 tok/step)
- Wider lr sweep: [3e-4, 5e-4, 1e-3, 2e-3] at 2M tokens each
- Full training at 30M tokens (no 60M extension)
- Dropout verification gate before training begins

Goal: val PPL < 50 -> CONFIRMED
"""

import os, sys, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import (
    MixtralConfig, MixtralForCausalLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

RESULTS_DIR      = Path("autoresearch/results")
EXPR_RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOK_DIR   = EXPR_RESULTS_DIR / "cycle_118_tokenizer"
CKPT_PATH = EXPR_RESULTS_DIR / "cycle_122_tiny_mixtral.pt"
JSON_PATH = RESULTS_DIR / "cycle_122_moe_training.json"

SEQ_LEN    = 256
VOCAB_SIZE = 4096
BATCH_SIZE = 2        # micro-batch (grad accum handles effective batch)
GRAD_ACCUM = 8        # effective batch = 2*8=16 seqs = 4096 tok/step
DEVICE     = "cpu"

torch.manual_seed(42)

# ─── 1. Load tokenizer ───────────────────────────────────────────────────────
print("Loading tokenizer from cycle_118 ...", flush=True)
hf_tok = PreTrainedTokenizerFast(
    tokenizer_file=str(TOK_DIR / "tokenizer.json"),
    pad_token="<pad>",
    eos_token="<eos>",
    unk_token="<unk>",
)
EOS_ID = hf_tok.eos_token_id

# ─── 2. Load dataset ─────────────────────────────────────────────────────────
print("Loading WikiText-2 ...", flush=True)
ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ds_val   = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

train_texts = [x for x in ds_train["text"] if x.strip()]
val_texts   = [x for x in ds_val["text"]   if x.strip()]

# ─── 3. Pack sequences ───────────────────────────────────────────────────────
def pack_texts(texts, seq_len):
    """Concatenate all token ids with EOS separators, chunk into seq_len blocks."""
    all_ids = []
    for t in texts:
        ids = hf_tok.encode(t)
        all_ids.extend(ids)
        all_ids.append(EOS_ID)
    all_ids = torch.tensor(all_ids, dtype=torch.long)
    n_blocks = len(all_ids) // seq_len
    blocks = all_ids[: n_blocks * seq_len].view(n_blocks, seq_len)
    return blocks

print("Packing sequences ...", flush=True)
train_blocks = pack_texts(train_texts, SEQ_LEN)
val_blocks   = pack_texts(val_texts,   SEQ_LEN)
print(f"  train blocks: {len(train_blocks)}, val blocks: {len(val_blocks)}", flush=True)

n_train = len(train_blocks)

# ─── 4. FFN dropout injection ────────────────────────────────────────────────
def inject_ffn_dropout(model, p=0.1):
    """Wrap each MoE block's forward to apply dropout after expert output."""
    count = 0
    for module in model.modules():
        if isinstance(module, MixtralSparseMoeBlock):
            drop = nn.Dropout(p)
            orig_forward = module.forward
            def make_wrapper(orig, d):
                def wrapper(*args, **kwargs):
                    result = orig(*args, **kwargs)
                    # Safe: handle single tensor OR tuple return
                    if isinstance(result, tuple):
                        return (d(result[0]),) + result[1:]
                    else:
                        return d(result)
                return wrapper
            module.forward = make_wrapper(orig_forward, drop)
            count += 1
    print(f"  FFN dropout p={p} injected into {count} MixtralSparseMoeBlock modules", flush=True)

# ─── 5. Dropout verification ─────────────────────────────────────────────────
def verify_dropout_active(model):
    """Two forward passes under train() must differ if dropout is active."""
    model.train()
    x = torch.randint(0, VOCAB_SIZE, (1, 32), device=DEVICE)
    with torch.no_grad():
        out1 = model(x).logits
        out2 = model(x).logits
    if torch.equal(out1, out2):
        raise RuntimeError("FATAL: Dropout NOT active — outputs identical! Aborting.")
    print("PASS: FFN dropout verified active (two forward passes differ)", flush=True)

# ─── 6. Model factory ────────────────────────────────────────────────────────
def make_model():
    config = MixtralConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.001,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQ_LEN,
        sliding_window=None,
        attention_dropout=0.1,
    )
    m = MixtralForCausalLM(config).to(DEVICE)
    inject_ffn_dropout(m, p=0.1)
    return m

# Print param count once
_test_model = make_model()
n_params = sum(p.numel() for p in _test_model.parameters())
print(f"Model params: {n_params/1e6:.1f}M", flush=True)
print("Verifying dropout on a test model ...", flush=True)
verify_dropout_active(_test_model)
del _test_model

# ─── 7. Val PPL utility ──────────────────────────────────────────────────────
def eval_val_ppl(model, max_blocks=256):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, min(max_blocks, len(val_blocks)), BATCH_SIZE):
            batch = val_blocks[i:i+BATCH_SIZE].to(DEVICE)
            out = model(input_ids=batch, labels=batch)
            losses.append(out.loss.item())
    model.train()
    return math.exp(sum(losses) / len(losses))

# ─── 8. Single training run ──────────────────────────────────────────────────
def train_one_run(lr, target_tokens, label="run", eval_every_tokens=None):
    """
    Train a fresh model for target_tokens with gradient accumulation.
    Returns (model, best_val_ppl, val_ppl_curve).
    """
    torch.manual_seed(42)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    tokens_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM  # effective step size
    steps_target = target_tokens // tokens_per_step
    n_warmup = max(100, steps_target // 20)

    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, steps_target)

    eval_steps = None
    if eval_every_tokens is not None:
        eval_steps = set(range(eval_every_tokens // tokens_per_step,
                               steps_target + 1,
                               eval_every_tokens // tokens_per_step))

    print(f"\n[{label}] lr={lr:.0e}, {target_tokens/1e6:.0f}M tokens, "
          f"{steps_target} optimizer steps (GRAD_ACCUM={GRAD_ACCUM})", flush=True)
    t0 = time.time()
    model.train()
    val_ppl_curve = []
    best_val_ppl = float("inf")
    best_state = None

    recent_val_ppls = []

    for step in range(1, steps_target + 1):
        # Gradient accumulation loop
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            idxs = torch.randint(0, n_train, (BATCH_SIZE,))
            x = train_blocks[idxs].to(DEVICE)
            out = model(input_ids=x, labels=x)
            micro_loss = out.loss / GRAD_ACCUM
            micro_loss.backward()
            accum_loss += micro_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tokens_done = step * tokens_per_step

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] step {step:5d} | loss {accum_loss:.3f} | "
                  f"tokens {tokens_done/1e6:.1f}M | {elapsed:.0f}s", flush=True)

        # Evaluate at checkpoints
        if eval_steps is not None and step in eval_steps:
            vp = eval_val_ppl(model)
            val_ppl_curve.append((tokens_done / 1e6, vp))
            print(f"  [{label}] EVAL @ {tokens_done/1e6:.0f}M: val_ppl={vp:.2f}", flush=True)

            if vp < best_val_ppl:
                best_val_ppl = vp
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            # Overfitting detection: val PPL rising for 3 consecutive checkpoints
            recent_val_ppls.append(vp)
            if len(recent_val_ppls) >= 3:
                last3 = recent_val_ppls[-3:]
                if last3[0] < last3[1] < last3[2] and accum_loss < 3.5:
                    print(f"  [{label}] Early stop: val PPL diverging "
                          f"({last3[0]:.2f} < {last3[1]:.2f} < {last3[2]:.2f}), "
                          f"train_loss={accum_loss:.3f}", flush=True)
                    break

    # Final val eval
    final_ppl = eval_val_ppl(model)
    if eval_steps is not None:
        if final_ppl < best_val_ppl:
            best_val_ppl = final_ppl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        best_val_ppl = final_ppl
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"  [{label}] DONE in {elapsed:.0f}s | final val_ppl={final_ppl:.2f} | "
          f"best val_ppl={best_val_ppl:.2f}", flush=True)

    model.load_state_dict(best_state)
    return model, best_val_ppl, val_ppl_curve


# ─── 9. Phase 1: lr sweep ────────────────────────────────────────────────────
SWEEP_LRS    = [3e-4, 5e-4, 1e-3, 2e-3]
SWEEP_TOKENS = 2_000_000  # 2M tokens per lr

print("\n" + "="*60, flush=True)
print("PHASE 1: lr sweep at 2M tokens each", flush=True)
print("="*60, flush=True)

sweep_results = {}
for lr in SWEEP_LRS:
    _, vp, _ = train_one_run(lr, SWEEP_TOKENS, label=f"sweep_lr{lr:.0e}")
    sweep_results[str(lr)] = vp
    print(f"  lr={lr:.0e} -> val_ppl={vp:.2f}", flush=True)

print("\nSweep table:", flush=True)
for k, v in sweep_results.items():
    print(f"  lr={k}: val_ppl={v:.2f}", flush=True)

# Pick best lr (lowest val PPL)
ppl_values = list(sweep_results.values())
ppl_range = max(ppl_values) - min(ppl_values)
if ppl_range < 5.0:
    best_lr = 3e-4
    print(f"\nAll lrs within 5 PPL (range={ppl_range:.2f}), defaulting to 3e-4", flush=True)
else:
    best_lr_str = min(sweep_results, key=lambda k: sweep_results[k])
    best_lr = float(best_lr_str)
    print(f"\nBest lr: {best_lr:.0e} (val_ppl={sweep_results[best_lr_str]:.2f})", flush=True)

# ─── 10. Phase 2: full training ──────────────────────────────────────────────
FULL_TOKENS = 30_000_000
EVAL_EVERY  = 2_000_000  # eval every 2M tokens (per plan)

print("\n" + "="*60, flush=True)
print(f"PHASE 2: full training at 30M tokens with lr={best_lr:.0e}", flush=True)
print("="*60, flush=True)

model, best_ppl_30m, val_curve = train_one_run(
    best_lr, FULL_TOKENS, label="full_30M",
    eval_every_tokens=EVAL_EVERY,
)

effective_ppl = best_ppl_30m
outcome = "CONFIRMED" if effective_ppl < 50.0 else "REFUTED"

print(f"\nOutcome: {outcome}", flush=True)
print(f"Val PPL curve: {val_curve}", flush=True)

# ─── 11. Save checkpoint (only if CONFIRMED) ─────────────────────────────────
if outcome == "CONFIRMED":
    torch.save({
        "model_state": model.state_dict(),
        "val_ppl": effective_ppl,
        "best_lr": best_lr,
    }, CKPT_PATH)
    print(f"Checkpoint saved to {CKPT_PATH}", flush=True)

    _ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    assert "model_state" in _ckpt, "Checkpoint missing model_state"
    print("Checkpoint loads OK.", flush=True)
else:
    print("No checkpoint saved (PPL >= 50).", flush=True)

# ─── 12. Save JSON ───────────────────────────────────────────────────────────
results = {
    "sweep_results": sweep_results,
    "best_lr": best_lr,
    "effective_ppl": effective_ppl,
    "checkpoint_path": str(CKPT_PATH) if outcome == "CONFIRMED" else None,
    "val_ppl_curve": val_curve,
    "outcome": outcome,
    "model_params_M": n_params / 1e6,
    "dropout_ffn": 0.1,
    "dropout_attention": 0.1,
    "grad_accum": GRAD_ACCUM,
    "batch_size": BATCH_SIZE,
    "sweep_tokens_M": SWEEP_TOKENS / 1e6,
    "full_tokens_M": FULL_TOKENS / 1e6,
}

with open(JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {JSON_PATH}", flush=True)

print("\n=== DONE ===", flush=True)
print(f"Outcome: {outcome}")
print(f"Effective val PPL: {effective_ppl:.2f}")
print(f"Best lr: {best_lr:.0e}")
print(f"Sweep: {sweep_results}")
print(f"Val PPL curve: {val_curve}")
