"""
Cycle 121: Train Tiny Mixtral MoE testbed — item 138.

Phase 1: lr sweep at 1M tokens each for lr={1e-4, 3e-4, 5e-4}
Phase 2: full training at 30M tokens with best lr
  - If PPL > 50 at 30M: extend to 60M tokens
  - Early stopping if val PPL starts diverging from train loss
Goal: val PPL < 50 → CONFIRMED

Design notes:
- WikiText-2 is ~2M unique tokens; 30M = ~12 epochs. Monitor overfitting.
- Cosine schedule with 5% warmup, weight_decay=0.01, AdamW
- Val PPL evaluated every 5M tokens during full train
- Best checkpoint saved based on lowest val PPL
"""

import os, sys, json, time, math
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import (
    MixtralConfig, MixtralForCausalLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

RESULTS_DIR      = Path("autoresearch/results")
EXPR_RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOK_DIR   = EXPR_RESULTS_DIR / "cycle_118_tokenizer"  # reuse cycle 118 tokenizer
CKPT_PATH = EXPR_RESULTS_DIR / "cycle_121_tiny_mixtral.pt"
JSON_PATH = RESULTS_DIR / "cycle_121_moe_training.json"

SEQ_LEN    = 256
VOCAB_SIZE = 4096
BATCH_SIZE = 4
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

# ─── 4. Model factory ────────────────────────────────────────────────────────
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
    )
    return MixtralForCausalLM(config).to(DEVICE)

# Print param count once
_test_model = make_model()
n_params = sum(p.numel() for p in _test_model.parameters())
print(f"Model params: {n_params/1e6:.1f}M", flush=True)
del _test_model

# ─── 5. Val PPL utility ──────────────────────────────────────────────────────
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

# ─── 6. Single training run ──────────────────────────────────────────────────
def train_one_run(lr, target_tokens, label="run", eval_every_tokens=None):
    """
    Train a fresh model for target_tokens. Returns (model, final_val_ppl, val_ppl_curve).
    val_ppl_curve: list of (tokens_M, ppl).
    If eval_every_tokens is set, evaluates val PPL at those checkpoints.
    """
    torch.manual_seed(42)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    tokens_per_step = BATCH_SIZE * SEQ_LEN
    steps_target = target_tokens // tokens_per_step
    n_warmup = max(100, steps_target // 20)

    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, steps_target)

    eval_steps = None
    if eval_every_tokens is not None:
        eval_steps = set(range(eval_every_tokens // tokens_per_step,
                               steps_target + 1,
                               eval_every_tokens // tokens_per_step))

    print(f"\n[{label}] lr={lr:.0e}, {target_tokens/1e6:.0f}M tokens, {steps_target} steps", flush=True)
    t0 = time.time()
    model.train()
    val_ppl_curve = []
    best_val_ppl = float("inf")
    best_state = None

    # Track recent val PPLs for overfitting detection
    recent_val_ppls = []

    for step in range(1, steps_target + 1):
        idxs = torch.randint(0, n_train, (BATCH_SIZE,))
        x = train_blocks[idxs].to(DEVICE)
        out = model(input_ids=x, labels=x)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 200 == 0:
            elapsed = time.time() - t0
            tok_done = step * tokens_per_step
            print(f"  [{label}] step {step:5d} | loss {loss.item():.3f} | "
                  f"tokens {tok_done/1e6:.1f}M | {elapsed:.0f}s", flush=True)

        # Evaluate at checkpoints
        if eval_steps is not None and step in eval_steps:
            tok_done = step * tokens_per_step
            vp = eval_val_ppl(model)
            val_ppl_curve.append((tok_done / 1e6, vp))
            print(f"  [{label}] EVAL @ {tok_done/1e6:.0f}M: val_ppl={vp:.2f}", flush=True)

            if vp < best_val_ppl:
                best_val_ppl = vp
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            # Overfitting detection: if val PPL has been rising for last 2 checkpoints
            # and train loss is below 3.0 (well-trained), stop early
            recent_val_ppls.append(vp)
            if len(recent_val_ppls) >= 3:
                last3 = recent_val_ppls[-3:]
                if last3[0] < last3[1] < last3[2] and loss.item() < 3.5:
                    print(f"  [{label}] Early stop: val PPL diverging "
                          f"({last3[0]:.2f} < {last3[1]:.2f} < {last3[2]:.2f}), "
                          f"train_loss={loss.item():.3f}", flush=True)
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

    # Restore best weights
    model.load_state_dict(best_state)
    return model, best_val_ppl, val_ppl_curve


# ─── 7. Phase 1: lr sweep ────────────────────────────────────────────────────
SWEEP_LRS     = [1e-4, 3e-4, 5e-4]
SWEEP_TOKENS  = 1_000_000

print("\n" + "="*60, flush=True)
print("PHASE 1: lr sweep at 1M tokens each", flush=True)
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
# If all within 5 PPL, default to 3e-4
ppl_values = list(sweep_results.values())
ppl_range = max(ppl_values) - min(ppl_values)
if ppl_range < 5.0:
    best_lr = 3e-4
    print(f"\nAll lrs within 5 PPL (range={ppl_range:.2f}), defaulting to 3e-4", flush=True)
else:
    best_lr_str = min(sweep_results, key=lambda k: sweep_results[k])
    best_lr = float(best_lr_str)
    print(f"\nBest lr: {best_lr:.0e} (val_ppl={sweep_results[best_lr_str]:.2f})", flush=True)

# ─── 8. Phase 2: full training ───────────────────────────────────────────────
FULL_TOKENS = 30_000_000
EVAL_EVERY  = 5_000_000  # eval every 5M tokens

print("\n" + "="*60, flush=True)
print(f"PHASE 2: full training at 30M tokens with lr={best_lr:.0e}", flush=True)
print("="*60, flush=True)

model, best_ppl_30m, val_curve = train_one_run(
    best_lr, FULL_TOKENS, label="full_30M",
    eval_every_tokens=EVAL_EVERY,
)

final_ppl_30m = best_ppl_30m
final_ppl_60m = None

if final_ppl_30m > 50.0:
    print(f"\nPPL={final_ppl_30m:.2f} > 50 at 30M tokens. Extending to 60M ...", flush=True)
    FULL_TOKENS_60 = 60_000_000
    model, best_ppl_60m, val_curve_60 = train_one_run(
        best_lr, FULL_TOKENS_60, label="full_60M",
        eval_every_tokens=EVAL_EVERY,
    )
    final_ppl_60m = best_ppl_60m
    val_curve = val_curve_60  # use the 60M curve as canonical

# ─── 9. Save checkpoint ──────────────────────────────────────────────────────
effective_ppl = final_ppl_60m if final_ppl_60m is not None else final_ppl_30m
torch.save({
    "model_state": model.state_dict(),
    "val_ppl": effective_ppl,
    "best_lr": best_lr,
}, CKPT_PATH)
print(f"Checkpoint saved to {CKPT_PATH}", flush=True)

# Quick load check
_ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
assert "model_state" in _ckpt, "Checkpoint missing model_state"
print("Checkpoint loads OK.", flush=True)

# ─── 10. Outcome verdict ─────────────────────────────────────────────────────
if effective_ppl < 50.0:
    outcome = "CONFIRMED"
elif final_ppl_60m is not None and final_ppl_60m > 50.0:
    outcome = "REFUTED"
else:
    outcome = "INCONCLUSIVE"

print(f"\nOutcome: {outcome}", flush=True)
print(f"Val PPL curve: {val_curve}", flush=True)

# ─── 11. Save JSON ───────────────────────────────────────────────────────────
results = {
    "sweep_results": sweep_results,
    "best_lr": best_lr,
    "final_ppl_30m": final_ppl_30m,
    "final_ppl_60m": final_ppl_60m,
    "effective_ppl": effective_ppl,
    "checkpoint_path": str(CKPT_PATH),
    "val_ppl_curve": val_curve,
    "outcome": outcome,
    "model_params_M": n_params / 1e6,
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
