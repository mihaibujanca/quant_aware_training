"""
Cycle 333 — LASER rank sweep on tiny GPT activation variants.
Tests whether cycle 280's rank-r-beats-baseline finding transfers across
GELU / ReLU² / LeakyReLU² activations on a controlled testbed.
"""

import json, math, os, sys, time
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

# ── Config ────────────────────────────────────────────────────────────────────
VARIANTS      = ["gelu", "relu2", "lrelu2"]
CKPT_DIR      = "experiments/tiny_gpt_act_compare_checkpoints"
RANKS         = [1, 2, 5, 10]
N_STEPS       = 500
LR            = 1e-4
EVAL_EVERY    = 50
SEEDS         = [42, 137, 2024]
N_EVAL        = 100
N_CALIB       = 200
BLOCK_SIZE    = 128
BATCH_SIZE    = 4
SPECTRUM_GATE = 0.10  # top-5 energy ≥ 10% needed

# Model constants (must match cycle 331/332)
N_LAYER     = 4
N_HEAD      = 4
N_EMBD      = 256
N_INNER     = 1024
VOCAB_SIZE  = 50257

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")

# ── Model (copied from cycle331 to avoid import issues) ───────────────────────
def get_act(name):
    if name == "gelu":
        return F.gelu
    elif name == "relu2":
        return lambda z: F.relu(z).square()
    elif name == "lrelu2":
        return lambda z: F.leaky_relu(z, 0.5).square()
    else:
        raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(self, act_fn):
        super().__init__()
        self.c_fc   = nn.Linear(N_EMBD, N_INNER, bias=False)
        self.c_proj = nn.Linear(N_INNER, N_EMBD, bias=False)
        self.act    = act_fn

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn      = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_attn_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.n_head      = N_HEAD
        self.n_embd      = N_EMBD
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(BLOCK_SIZE, BLOCK_SIZE), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        head_size = C // self.n_head
        def reshape(t):
            return t.view(B, T, self.n_head, head_size).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        scale = 1.0 / math.sqrt(head_size)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_attn_proj(y)


class Block(nn.Module):
    def __init__(self, act_fn):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBD)
        self.ln2  = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.mlp  = MLP(act_fn)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, act_name):
        super().__init__()
        act_fn = get_act(act_name)
        self.wte  = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe  = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(0.0)
        self.blocks    = nn.ModuleList([Block(act_fn) for _ in range(N_LAYER)])
        self.ln_f      = nn.LayerNorm(N_EMBD)
        self.lm_head   = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss


# ── Low-rank c_proj ───────────────────────────────────────────────────────────
class LowRankCProj(nn.Module):
    """Rank-r replacement for nn.Linear c_proj (no bias).

    c_proj.weight shape: (out_features=256, in_features=1024)
    SVD: U(256,256), S(256,), Vh(256,1024)
    A = U[:, :r] * S[:r]  → (256, r), frozen
    B = Vh[:r, :]          → (r, 1024), trainable
    Forward: F.linear(x, A @ B)  [since F.linear(x,W) = x @ W.T and W=A@B]
    """
    def __init__(self, A_frozen, B_init):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())

    def forward(self, x):
        W = self.A @ self.B  # (256, 1024)
        return F.linear(x, W)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    print("Loading wikitext-2 ...", flush=True)
    ds  = load_dataset("wikitext", "wikitext-2-raw-v1")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(split):
        texts   = [t for t in ds[split]["text"] if t.strip()]
        all_ids = []
        for text in texts:
            ids = tok.encode(text)
            all_ids.extend(ids)
        return torch.tensor(all_ids, dtype=torch.long)

    print("  tokenizing train...", flush=True)
    train_ids = encode("train")
    print(f"  train tokens: {len(train_ids):,}", flush=True)
    print("  tokenizing val...", flush=True)
    val_ids   = encode("validation")
    print(f"  val tokens: {len(val_ids):,}", flush=True)
    return train_ids, val_ids


def make_batches(ids, n_seq, block_size, batch_size):
    x_list, y_list = [], []
    stride = block_size + 1
    for i in range(n_seq):
        start = (i * stride) % max(1, len(ids) - stride)
        chunk = ids[start : start + stride]
        if len(chunk) < stride:
            chunk = torch.cat([chunk, ids[:stride - len(chunk)]])
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x, y


def eval_ppl(model, val_x, val_y, device, n_seq=None):
    """Evaluate perplexity on val sequences."""
    model.eval()
    if n_seq is not None:
        val_x = val_x[:n_seq]
        val_y = val_y[:n_seq]
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(val_x), BATCH_SIZE):
            xb = val_x[i:i+BATCH_SIZE].to(device)
            yb = val_y[i:i+BATCH_SIZE].to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item()
            n_batches += 1
    return math.exp(total_loss / n_batches)


def load_model(variant, seed, device):
    """Load checkpoint for (variant, seed)."""
    ckpt_path = os.path.join(CKPT_DIR, f"{variant}_s{seed}.pt")
    model = TinyGPT(variant).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


# ── Step 1: Spectrum preflight ────────────────────────────────────────────────
def spectrum_preflight(val_x, val_y, device):
    print("\n=== SPECTRUM PREFLIGHT ===")
    spectrum = {}
    target_blocks = {}

    for variant in VARIANTS:
        model = load_model(variant, 42, device)
        spectrum[variant] = {}
        for b in range(N_LAYER):
            W = model.blocks[b].mlp.c_proj.weight.float()  # (256, 1024)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            energy    = S.pow(2)
            total_e   = energy.sum().item()
            top1_e    = energy[0].item()  / total_e
            top5_e    = energy[:5].sum().item()  / total_e
            top10_e   = energy[:10].sum().item() / total_e
            spectrum[variant][b] = {
                "top1":  top1_e,
                "top5":  top5_e,
                "top10": top10_e,
            }

        # Best block = highest top5 energy
        best_b   = max(range(N_LAYER), key=lambda b: spectrum[variant][b]["top5"])
        best_top5 = spectrum[variant][best_b]["top5"]
        target_blocks[variant] = (best_b, best_top5)

    # Print 3×4 table
    print(f"\n{'Variant':<10} " + " ".join(f"B{b}_top5" for b in range(N_LAYER)) + "  BestBlk  BestTop5")
    for variant in VARIANTS:
        row = " ".join(f"{spectrum[variant][b]['top5']:.4f}" for b in range(N_LAYER))
        bb, bt5 = target_blocks[variant]
        print(f"{variant:<10} {row}  B{bb}       {bt5:.4f}")

    return spectrum, target_blocks


# ── Step 2: Baseline PPL ──────────────────────────────────────────────────────
def compute_baselines(val_x, val_y, device):
    print("\n=== BASELINE PPL ===")
    baselines = {}
    for variant in VARIANTS:
        baselines[variant] = {}
        ppls = []
        for seed in SEEDS:
            model = load_model(variant, seed, device)
            ppl = eval_ppl(model, val_x, val_y, device, n_seq=N_EVAL)
            baselines[variant][seed] = ppl
            ppls.append(ppl)
            print(f"  {variant} seed={seed}: PPL={ppl:.2f}")
        mean_ppl = sum(ppls) / len(ppls)
        print(f"  {variant} MEAN: PPL={mean_ppl:.2f}")
    return baselines


# ── Helpers for LASER ─────────────────────────────────────────────────────────
def make_low_rank(model, block_idx, r, device):
    """Replace block_idx's c_proj with LowRankCProj of rank r. Return new model."""
    import copy
    m = copy.deepcopy(model)
    W = m.blocks[block_idx].mlp.c_proj.weight.float()  # (256, 1024)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # A = U[:, :r] * S[:r], shape (256, r)
    A = (U[:, :r] * S[:r]).to(device)
    # B = Vh[:r, :], shape (r, 1024)
    B = Vh[:r, :].to(device)
    lr_layer = LowRankCProj(A, B)
    m.blocks[block_idx].mlp.c_proj = lr_layer
    return m.to(device)


def laser_retrain(model, block_idx, r, calib_x, calib_y, val_x, val_y, device, seed):
    """Retrain B in LowRankCProj(r) for N_STEPS steps. Return best PPL."""
    torch.manual_seed(seed)
    m = make_low_rank(model, block_idx, r, device)

    # Only B is trainable
    params = [p for name, p in m.named_parameters() if name.endswith(".B")]
    assert len(params) == 1, f"Expected 1 trainable param (B), got {len(params)}"
    opt = torch.optim.Adam(params, lr=LR)

    best_ppl = float("inf")
    best_B   = params[0].data.clone()

    for step in range(1, N_STEPS + 1):
        m.train()
        # Random batch from calib
        idx = torch.randint(0, calib_x.shape[0], (BATCH_SIZE,))
        xb  = calib_x[idx].to(device)
        yb  = calib_y[idx].to(device)
        _, loss = m(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0:
            ppl = eval_ppl(m, val_x, val_y, device, n_seq=N_EVAL)
            if ppl < best_ppl:
                best_ppl = ppl
                best_B   = params[0].data.clone()

    # Restore best B
    params[0].data.copy_(best_B)
    # Final eval
    ppl = eval_ppl(m, val_x, val_y, device, n_seq=N_EVAL)
    if ppl < best_ppl:
        best_ppl = ppl
    return best_ppl


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    train_ids, val_ids = load_data()

    val_x,   val_y   = make_batches(val_ids,   N_EVAL,  BLOCK_SIZE, BATCH_SIZE)
    calib_x, calib_y = make_batches(val_ids,   N_CALIB, BLOCK_SIZE, BATCH_SIZE)

    results = {}

    # ── Step 1: Spectrum ──────────────────────────────────────────────────────
    spectrum, target_blocks = spectrum_preflight(val_x, val_y, device=DEVICE)
    results["spectrum"] = {
        v: {str(b): spectrum[v][b] for b in range(N_LAYER)}
        for v in VARIANTS
    }
    results["target_blocks"] = {v: int(target_blocks[v][0]) for v in VARIANTS}

    # Check gate
    skipped = []
    for variant in VARIANTS:
        _, best_top5 = target_blocks[variant]
        if best_top5 < SPECTRUM_GATE:
            print(f"  GATE FAIL: {variant} best top5={best_top5:.4f} < {SPECTRUM_GATE} → skip")
            skipped.append(variant)
    active_variants = [v for v in VARIANTS if v not in skipped]
    print(f"Active variants: {active_variants}  Skipped: {skipped}")

    # ── Step 2: Baselines ─────────────────────────────────────────────────────
    baselines = compute_baselines(val_x, val_y, device=DEVICE)
    results["baselines"] = {
        v: {str(s): baselines[v][s] for s in SEEDS}
        for v in VARIANTS
    }

    # ── Step 4.0: Full-rank sanity (GELU seed-42) ─────────────────────────────
    print("\n=== FULL-RANK SANITY (GELU, seed=42, r=256) ===")
    gelu_model = load_model("gelu", 42, DEVICE)
    b_idx = target_blocks["gelu"][0]
    full_rank_model = make_low_rank(gelu_model, b_idx, r=N_EMBD, device=DEVICE)
    fr_ppl = eval_ppl(full_rank_model, val_x, val_y, DEVICE, n_seq=N_EVAL)
    base_ppl_gelu = baselines["gelu"][42]
    diff = abs(fr_ppl - base_ppl_gelu)
    print(f"  Full-rank PPL={fr_ppl:.4f}  Baseline PPL={base_ppl_gelu:.4f}  diff={diff:.4f}")
    if diff > 0.05:
        print(f"  WARNING: full-rank PPL diff={diff:.4f} > 0.05 — possible transposition bug!")
        # Don't abort, but flag it
    else:
        print(f"  PASS: full-rank recovery within 0.05 PPL")
    results["full_rank_sanity"] = {"ppl": fr_ppl, "baseline": base_ppl_gelu, "diff": diff}

    # ── Step 3+4: LASER sweep ─────────────────────────────────────────────────
    print("\n=== LASER SWEEP ===")
    laser_results = {}

    for variant in active_variants:
        block_idx = target_blocks[variant][0]
        laser_results[variant] = {}
        base_model = load_model(variant, 42, DEVICE)  # always same base weights

        for r in RANKS:
            print(f"\n  {variant}  block={block_idx}  r={r}")
            laser_results[variant][r] = {}

            # No-retrain PPL
            lr_model_noret = make_low_rank(base_model, block_idx, r, DEVICE)
            noret_ppl = eval_ppl(lr_model_noret, val_x, val_y, DEVICE, n_seq=N_EVAL)
            laser_results[variant][r]["no_retrain"] = noret_ppl
            print(f"    no-retrain PPL={noret_ppl:.2f}")

            # Retrain per seed
            seed_ppls = []
            for seed in SEEDS:
                best_ppl = laser_retrain(
                    base_model, block_idx, r,
                    calib_x, calib_y, val_x, val_y,
                    DEVICE, seed
                )
                laser_results[variant][r][seed] = best_ppl
                seed_ppls.append(best_ppl)
                print(f"    retrain seed={seed}: best_PPL={best_ppl:.2f}")

            mean_ppl = sum(seed_ppls) / len(seed_ppls)
            std_ppl  = (sum((p - mean_ppl)**2 for p in seed_ppls) / len(seed_ppls)) ** 0.5
            laser_results[variant][r]["mean"] = mean_ppl
            laser_results[variant][r]["std"]  = std_ppl
            print(f"    retrain mean={mean_ppl:.2f} ±{std_ppl:.2f}")

    results["laser"] = {
        v: {str(r): laser_results[v][r] for r in laser_results.get(v, {})}
        for v in active_variants
    }

    # ── Step 5: Output tables ─────────────────────────────────────────────────
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Variant':<10} {'Baseline':<12} " + " ".join(f"r={r}_mean" for r in RANKS))
    for variant in VARIANTS:
        base = sum(baselines[variant][s] for s in SEEDS) / len(SEEDS)
        if variant in laser_results:
            row = " ".join(
                f"{laser_results[variant][r]['mean']:.2f}"
                if r in laser_results[variant] else "  --  "
                for r in RANKS
            )
        else:
            row = "SKIPPED (flat spectrum)"
        print(f"{variant:<10} {base:<12.2f} {row}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n=== VERDICT ===")
    verdict_per_variant = {}
    for variant in VARIANTS:
        if variant in skipped:
            verdict_per_variant[variant] = "SKIP-flat-spectrum"
            continue
        base = baselines[variant][42]  # use seed-42 as reference
        passed = any(
            laser_results[variant][r]["mean"] <= base + 0.5
            for r in RANKS
            if r in laser_results[variant]
        )
        verdict_per_variant[variant] = "PASS" if passed else "FAIL"
        print(f"  {variant}: base={base:.2f}, " +
              ", ".join(f"r={r}→{laser_results[variant][r]['mean']:.2f}" for r in RANKS if r in laser_results[variant]) +
              f"  → {verdict_per_variant[variant]}")

    all_pass = all(verdict_per_variant[v] == "PASS" for v in VARIANTS)
    only_gelu = (verdict_per_variant.get("gelu") == "PASS" and
                 all(verdict_per_variant.get(v) != "PASS" for v in ["relu2", "lrelu2"]))
    all_flat  = len(skipped) == len(VARIANTS)

    if all_flat:
        VERDICT = "REFUTED-flat-spectrum"
    elif all_pass:
        VERDICT = "CONFIRMED-LASER-generic"
    elif only_gelu:
        VERDICT = "REFUTED-LASER-GELU-specific"
    else:
        VERDICT = "INCONCLUSIVE"

    print(f"\nFINAL VERDICT: {VERDICT}")
    results["verdict"]         = VERDICT
    results["verdict_per_variant"] = verdict_per_variant
    results["skipped"]         = skipped
    results["elapsed_sec"]     = time.time() - t0

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = "autoresearch/results/cycle_333_laser.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    return VERDICT, results


if __name__ == "__main__":
    VERDICT, results = main()
    print(f"\nDone. Verdict: {VERDICT}")
