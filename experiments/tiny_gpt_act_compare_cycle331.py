"""
Cycle 332 / Item 331 — Tiny GPT activation comparison
3 activations {GELU, ReLU², LeakyReLU²} × 3 seeds × 3000 steps on wikitext-2
Reports val PPL and eff_rank census per block.
"""

import json, math, os, sys, time
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

# ── Config ────────────────────────────────────────────────────────────────────
N_LAYER      = 4
N_HEAD       = 4
N_EMBD       = 256
N_INNER      = 1024
VOCAB_SIZE   = 50257
BLOCK_SIZE   = 128
BATCH_SIZE   = 32
N_STEPS      = 3000
WARMUP       = 200
LR_MAIN      = 3e-4
LR_FALLBACK  = 1e-4
WD           = 0.01
BETAS        = (0.9, 0.95)
SEEDS        = [42, 137, 2024]
ACTIVATIONS  = ["gelu", "relu2", "lrelu2"]
N_VAL_SEQ    = 100
N_CALIB_SEQ  = 200
CKPT_DIR     = "experiments/tiny_gpt_act_compare_checkpoints"
RESULTS_JSON = "autoresearch/results/cycle_332_eff_rank.json"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs("autoresearch/results", exist_ok=True)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")

# ── Activation map ─────────────────────────────────────────────────────────────
def get_act(name):
    if name == "gelu":
        return F.gelu
    elif name == "relu2":
        return lambda z: F.relu(z).square()
    elif name == "lrelu2":
        return lambda z: F.leaky_relu(z, 0.5).square()
    else:
        raise ValueError(f"Unknown activation: {name}")

# ── Model ─────────────────────────────────────────────────────────────────────
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
        self.c_attn_proj = nn.Linear(N_EMBD, N_EMBD,     bias=False)
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
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
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
        self.blocks = nn.ModuleList([Block(act_fn) for _ in range(N_LAYER)])
        self.ln_f   = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight  # tie weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

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


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data():
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(split):
        # Single-threaded encoding to avoid forking issues with tokenizers
        texts = [t for t in ds[split]["text"] if t.strip()]
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
    """Return (n_seq, block_size) input and target tensors."""
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


def get_random_batch(train_ids, device):
    ix = torch.randint(len(train_ids) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x  = torch.stack([train_ids[i : i + BLOCK_SIZE]     for i in ix])
    y  = torch.stack([train_ids[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


# ── LR schedule ───────────────────────────────────────────────────────────────
def make_schedule(optimizer, n_steps, warmup):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ── Eval ───────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_ppl(model, val_ids, device):
    model.eval()
    x_val, y_val = make_batches(val_ids, N_VAL_SEQ, BLOCK_SIZE, BATCH_SIZE)
    losses = []
    for i in range(0, N_VAL_SEQ, BATCH_SIZE):
        xb = x_val[i:i+BATCH_SIZE].to(device)
        yb = y_val[i:i+BATCH_SIZE].to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    ppl = math.exp(sum(losses) / len(losses))
    model.train()
    return ppl


# ── Single training run ────────────────────────────────────────────────────────
def train_run(act_name, seed, train_ids, val_ids, lr, tag=""):
    torch.manual_seed(seed)
    model = TinyGPT(act_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD, betas=BETAS)
    sched = make_schedule(opt, N_STEPS, WARMUP)

    best_ppl = float('inf')
    best_state = None
    t0 = time.time()

    for step in range(N_STEPS):
        x, y = get_random_batch(train_ids, DEVICE)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % 500 == 0:
            ppl = eval_ppl(model, val_ids, DEVICE)
            elapsed = time.time() - t0
            print(f"  [{act_name} s{seed}{tag}] step {step+1:4d}  loss={loss.item():.4f}  val_ppl={ppl:.2f}  {elapsed:.0f}s")
            if ppl < best_ppl:
                best_ppl = ppl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final PPL at last step even if not 500-aligned
    final_ppl = eval_ppl(model, val_ids, DEVICE)
    if final_ppl < best_ppl:
        best_ppl = final_ppl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"  [{act_name} s{seed}{tag}] FINAL val_ppl={final_ppl:.2f}  best={best_ppl:.2f}  n_params={n_params:,}")

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"{act_name}_s{seed}.pt")
    torch.save({"state_dict": best_state, "act_name": act_name, "seed": seed, "val_ppl": best_ppl}, ckpt_path)
    return best_ppl, ckpt_path


# ── Training with LR fallback ──────────────────────────────────────────────────
def train_with_fallback(act_name, seed, train_ids, val_ids):
    """Train for 3000 steps. No fallback — just record final PPL."""
    torch.manual_seed(seed)
    model = TinyGPT(act_name).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAIN, weight_decay=WD, betas=BETAS)
    sched = make_schedule(opt, N_STEPS, WARMUP)

    best_ppl = float('inf')
    best_state = None
    t0 = time.time()

    for step in range(N_STEPS):
        x, y = get_random_batch(train_ids, DEVICE)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % 500 == 0:
            ppl = eval_ppl(model, val_ids, DEVICE)
            elapsed = time.time() - t0
            print(f"  [{act_name} s{seed}] step {step+1:4d}  loss={loss.item():.4f}  val_ppl={ppl:.2f}  {elapsed:.0f}s", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    final_ppl = eval_ppl(model, val_ids, DEVICE)
    if final_ppl < best_ppl:
        best_ppl = final_ppl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"  [{act_name} s{seed}] FINAL val_ppl={final_ppl:.2f}  best={best_ppl:.2f}  n_params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    ckpt_path = os.path.join(CKPT_DIR, f"{act_name}_s{seed}.pt")
    torch.save({"state_dict": best_state, "act_name": act_name, "seed": seed, "val_ppl": best_ppl}, ckpt_path)
    return best_ppl, ckpt_path, False


# ── Eff_rank census ────────────────────────────────────────────────────────────
@torch.no_grad()
def eff_rank_census(ckpt_path, val_ids):
    """Return list of 4 eff_rank values (one per block) from c_proj outputs."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    act_name = ckpt["act_name"]
    model = TinyGPT(act_name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(DEVICE)

    activations = {i: [] for i in range(N_LAYER)}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # out shape: (B, T, N_EMBD) — flatten to (B*T, N_EMBD)
            activations[layer_idx].append(out.detach().cpu().reshape(-1, N_EMBD))
        return hook

    for i, block in enumerate(model.blocks):
        h = block.mlp.c_proj.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Forward N_CALIB_SEQ sequences
    x_calib, _ = make_batches(val_ids, N_CALIB_SEQ, BLOCK_SIZE, BATCH_SIZE)
    for i in range(0, N_CALIB_SEQ, BATCH_SIZE):
        xb = x_calib[i:i+BATCH_SIZE].to(DEVICE)
        model(xb)

    for h in hooks:
        h.remove()

    ranks = []
    for i in range(N_LAYER):
        A = torch.cat(activations[i], dim=0).float()  # (N, 256)
        A = A - A.mean(dim=0, keepdim=True)
        C = A.T @ A / A.shape[0]
        sv = torch.linalg.svdvals(C)
        sv = sv.clamp(min=0)
        sv_sqrt = sv.sqrt()  # singular values of A (up to scale)
        er = sv_sqrt.sum().square() / sv_sqrt.square().sum()
        ranks.append(er.item())

    model.cpu()
    return ranks


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    train_ids, val_ids = load_data()

    ppl_results = {a: {} for a in ACTIVATIONS}
    fallback_flags = {a: {} for a in ACTIVATIONS}

    # ── Training ──────────────────────────────────────────────────────────────
    for act in ACTIVATIONS:
        print(f"\n{'='*60}")
        print(f"  Activation: {act}")
        print(f"{'='*60}")
        for seed in SEEDS:
            ppl, _, fb = train_with_fallback(act, seed, train_ids, val_ids)
            ppl_results[act][seed] = ppl
            fallback_flags[act][seed] = fb

    # ── PPL summary ───────────────────────────────────────────────────────────
    import statistics
    print("\n\n=== Val PPL Summary ===")
    print(f"{'Act':<12}  {'s42':>8}  {'s137':>8}  {'s2024':>8}  {'mean':>8}  {'std':>8}")
    print("-" * 70)
    for act in ACTIVATIONS:
        ppls = [ppl_results[act][s] for s in SEEDS]
        mu   = statistics.mean(ppls)
        sd   = statistics.stdev(ppls)
        fbs  = [f"*" if fallback_flags[act][s] else "" for s in SEEDS]
        vals = "  ".join(f"{p:8.2f}{fb}" for p, fb in zip(ppls, fbs))
        print(f"{act:<12}  {ppls[0]:8.2f}  {ppls[1]:8.2f}  {ppls[2]:8.2f}  {mu:8.2f}  {sd:8.3f}")

    # ── Eff_rank census ───────────────────────────────────────────────────────
    print("\n=== Eff_rank Census ===")
    eff_rank_data = {}
    for act in ACTIVATIONS:
        seed_ranks = []
        for seed in SEEDS:
            ckpt_path = os.path.join(CKPT_DIR, f"{act}_s{seed}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  Missing checkpoint: {ckpt_path}")
                continue
            ranks = eff_rank_census(ckpt_path, val_ids)
            seed_ranks.append(ranks)
            print(f"  {act} s{seed}: {[f'{r:.2f}' for r in ranks]}")

        if not seed_ranks:
            continue

        n_seeds = len(seed_ranks)
        block_stats = {}
        for b in range(N_LAYER):
            vals_b = [seed_ranks[s][b] for s in range(n_seeds)]
            mu_b   = statistics.mean(vals_b)
            sd_b   = statistics.stdev(vals_b) if n_seeds > 1 else 0.0
            block_stats[str(b)] = {
                "mean": mu_b,
                "std":  sd_b,
                "per_seed": vals_b
            }
        eff_rank_data[act] = block_stats

    # ── Eff_rank table ────────────────────────────────────────────────────────
    print("\n=== Eff_rank Table (mean ± std across seeds) ===")
    print(f"{'Act':<12}  {'Block0':>12}  {'Block1':>12}  {'Block2':>12}  {'Block3':>12}")
    print("-" * 70)
    for act in ACTIVATIONS:
        if act not in eff_rank_data:
            continue
        cells = []
        for b in range(N_LAYER):
            s = eff_rank_data[act][str(b)]
            cells.append(f"{s['mean']:5.2f}±{s['std']:.2f}")
        print(f"{act:<12}  {'  '.join(cells)}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    # Use 250 as convergence threshold (3000 steps, tiny model, wikitext-2 is hard)
    all_converged = all(
        ppl_results[a][s] <= 250 for a in ACTIVATIONS for s in SEEDS
    )

    # Check if all in eff_rank_data
    if len(eff_rank_data) == 3:
        # Max pairwise diff per block
        import itertools
        pairs = list(itertools.combinations(ACTIVATIONS, 2))
        blocks_with_big_diff = 0
        for b in range(N_LAYER):
            means = {a: eff_rank_data[a][str(b)]["mean"] for a in ACTIVATIONS if a in eff_rank_data}
            stds  = {a: eff_rank_data[a][str(b)]["std"]  for a in ACTIVATIONS if a in eff_rank_data}
            for a1, a2 in pairs:
                diff = abs(means[a1] - means[a2])
                max_std = max(stds[a1], stds[a2])
                if diff >= 0.5 and max_std < 0.3:
                    blocks_with_big_diff += 1
                    break  # count each block once

        # Check if all within ±0.3
        all_close = all(
            abs(eff_rank_data[a1][str(b)]["mean"] - eff_rank_data[a2][str(b)]["mean"]) < 0.3
            for a1, a2 in pairs
            for b in range(N_LAYER)
        )

        if all_converged and blocks_with_big_diff >= 3:
            verdict = "CONFIRMED-rank-profile-shape"
        elif all_close:
            verdict = "REFUTED-rank-profile-converges"
        else:
            verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  all_converged={all_converged}  n_blocks_with_big_diff={blocks_with_big_diff if len(eff_rank_data)==3 else 'N/A'}")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "verdict": verdict,
        "ppl": {a: {str(s): ppl_results[a][s] for s in SEEDS} for a in ACTIVATIONS},
        "fallback": {a: {str(s): fallback_flags[a][s] for s in SEEDS} for a in ACTIVATIONS},
        "eff_rank": eff_rank_data
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
