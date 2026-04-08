"""
Cycle 308 — Item 318: Synthetic rank-2 unigram projector for block 1 c_proj.

Hand-designed rank-2 c_proj replacement at GPT-2 small block 1, using
empirical unigram log-frequencies and a stopword-indicator direction
(both projected through W_E = wte.weight) as the OUTPUT-side rank-2
basis. Cycle 280's frozen A = U[:, :2] @ diag(S[:2]) is reused as the
LEFT factor (3072, 2) — that is the only piece reused from cycle 280
(no Adam, no SVD beyond A's extraction, no data fit beyond the unigram
histogram and a 25-point grid search over two scalar magnitudes).

Verdict per item 318:
  CONFIRMED iff best synth-projector PPL <= 60
  REFUTED   iff best synth-projector PPL >= 100
  INCONCLUSIVE otherwise.
"""
import json
import math
import time

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_BLOCK = 1
RANK = 2
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42

GRID = [0.5, 1.0, 2.0, 5.0, 10.0]

FLOAT_BASELINE_REF = 53.895
CYCLE_280_INIT_REF = 57.664  # no-retrain rank-2 SVD init PPL at block 1
CYCLE_280_BEST_REF = 49.032  # cycle 280's trained rank-2 PPL at block 1

OUTPUT_JSON = "autoresearch/results/cycle_308_synthetic_unigram.json"
OUTPUT_SUMMARY = "autoresearch/results/cycle_308_summary.md"

# ~300 GPT-2 BPE stopwords + punctuation. Each gets encoded with and without
# leading space; encode() returns a list of token ids; we accept all returned
# ids that decode back exactly to the requested string (single-token only).
STOPWORD_BASE = [
    # Punctuation
    ".", ",", ";", ":", "?", "!", "'", '"', "(", ")", "[", "]", "{", "}",
    "-", "--", "/", "\\", "|", "*", "&", "%", "$", "#", "@", "+", "=",
    "<", ">", "~", "`", "_",
    # Common contractions / suffixes
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "n't",
    # Articles, pronouns
    "the", "a", "an", "this", "that", "these", "those",
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "what", "which", "where", "when", "why", "how",
    # Be / aux
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "ought",
    # Conjunctions / prepositions
    "and", "or", "but", "nor", "so", "yet", "for", "as", "if", "then",
    "else", "because", "since", "while", "until", "though", "although",
    "unless", "however", "therefore", "moreover", "thus",
    "in", "on", "at", "by", "from", "of", "to", "with", "without",
    "into", "onto", "upon", "over", "under", "above", "below",
    "between", "among", "through", "across", "around", "behind",
    "before", "after", "during", "about", "against", "near", "off",
    "out", "up", "down", "back", "away", "here", "there",
    # Quantifiers / determiners
    "all", "any", "some", "no", "none", "every", "each", "few", "many",
    "much", "more", "most", "less", "least", "other", "another", "such",
    "same", "different", "various",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten",
    # Negation / agreement
    "not", "yes", "also", "too", "very", "just", "only", "even", "still",
    "always", "never", "often", "sometimes", "usually", "rarely",
    "perhaps", "maybe",
    # Common verbs (extremely high frequency)
    "said", "say", "says", "saying",
    "make", "made", "makes", "making",
    "go", "goes", "went", "gone", "going",
    "get", "got", "gets", "getting",
    "see", "saw", "seen", "seeing",
    "know", "knew", "known", "knowing",
    "take", "took", "taken", "taking",
    "come", "came", "coming",
    "give", "gave", "given", "giving",
    "find", "found", "finding",
    "think", "thought", "thinking",
    "use", "used", "using", "uses",
    "want", "wanted", "wants",
    "look", "looked", "looking", "looks",
    "work", "worked", "working", "works",
    "call", "called", "calling", "calls",
    "try", "tried", "trying", "tries",
    "ask", "asked", "asking", "asks",
    "need", "needed", "needs",
    "feel", "felt", "feeling",
    "become", "became",
    "leave", "left", "leaving",
    "put", "puts", "putting",
    "mean", "means", "meant", "meaning",
    "keep", "kept", "keeping",
    "let", "lets", "letting",
    "begin", "began", "begun",
    "show", "showed", "shown",
    "hear", "heard",
    "play", "played",
    "run", "ran",
    "move", "moved",
    "live", "lived",
    "believe", "believed",
    "hold", "held",
    "bring", "brought",
    "happen", "happened",
    "write", "wrote", "written",
    "provide", "provided",
    "sit", "sat",
    "stand", "stood",
    "lose", "lost",
    "pay", "paid",
    "meet", "met",
    "include", "included",
    "continue", "continued",
    "set", "sets",
    "learn", "learned",
    "change", "changed",
    "lead", "led",
    "understand", "understood",
    "watch", "watched",
    "follow", "followed",
    "stop", "stopped",
    "create", "created",
    "speak", "spoke",
    "read", "reading",
    "spend", "spent",
    "grow", "grew",
    "open", "opened",
    "walk", "walked",
    "win", "won",
    "offer", "offered",
    "remember", "remembered",
    "consider", "considered",
    "appear", "appeared",
    "buy", "bought",
    "wait", "waited",
    "serve", "served",
    "die", "died",
    "send", "sent",
    "build", "built",
    "stay", "stayed",
    "fall", "fell", "fallen",
    "cut", "cuts",
    "reach", "reached",
    "kill", "killed",
    "remain", "remained",
]

torch.manual_seed(SEED)


def main():
    print(f"Device: {DEVICE}", flush=True)
    t0_total = time.time()

    # ── Load model + tokenizer + data ────────────────────────────────────────
    print("Loading GPT-2 and wikitext-2 val...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
    all_ids = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        if enc["input_ids"].shape[1] >= SEQ_LEN:
            all_ids.append(enc["input_ids"][:, :SEQ_LEN])
        if len(all_ids) >= N_CALIB + N_EVAL:
            break
    assert len(all_ids) >= N_CALIB + N_EVAL, f"only got {len(all_ids)} seqs"
    calib_ids = [t for t in all_ids[:N_CALIB]]  # CPU; used for unigram histogram
    eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
    print(f"Calib: {len(calib_ids)} seqs; Eval: {len(eval_ids)} seqs", flush=True)

    # ── Float baseline ───────────────────────────────────────────────────────
    def eval_ppl(m):
        m.eval()
        total_loss = 0.0
        total_tok = 0
        with torch.no_grad():
            for ids in eval_ids:
                out = m(ids, labels=ids)
                total_loss += out.loss.item() * (ids.shape[1] - 1)
                total_tok += ids.shape[1] - 1
        return math.exp(total_loss / total_tok)

    t0 = time.time()
    float_ppl = eval_ppl(model)
    print(f"Float PPL: {float_ppl:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
        f"Float PPL {float_ppl:.4f} differs from ref {FLOAT_BASELINE_REF} by > 0.5"
    )

    # ── Snapshot block 1 c_proj weight + bias + SVD ──────────────────────────
    c_proj = model.transformer.h[TARGET_BLOCK].mlp.c_proj
    W_orig = c_proj.weight.data.detach().clone()  # (3072, 768)
    b_orig = c_proj.bias.data.detach().clone()    # (768,)
    assert W_orig.shape == (3072, 768)
    assert b_orig.shape == (768,)
    print(f"\nSVD of block-{TARGET_BLOCK} c_proj.weight ({tuple(W_orig.shape)})...", flush=True)
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W_orig.float().cpu(), full_matrices=False)
    print(f"  SVD done in {time.time() - t0:.1f}s; S[:5]={S[:5].tolist()}", flush=True)
    A_frozen = (U[:, :RANK] * S[:RANK]).contiguous()  # (3072, 2) — cycle 280 A factor
    Vh_init = Vh[:RANK, :].contiguous()  # (2, 768) — cycle 280 SVD-init B factor
    assert A_frozen.shape == (3072, RANK)
    assert Vh_init.shape == (RANK, 768)
    # Magnitudes used for synthetic-grid scaling: the cycle 280 SVD init has
    # B = Vh[:2, :] with unit-norm rows. We rescale the synthetic d1, d2 directions
    # to the SAME row-norm = 1.0 (so s1=s2=1 corresponds to "same scale as the
    # SVD-init B rows" — directly comparable to the no-retrain init point).
    mag1_base = 1.0
    mag2_base = 1.0

    # ── Sanity: SVD-init replacement reproduces cycle 280 init PPL ───────────
    print("\nSanity: install A @ Vh[:2, :] (cycle 280 no-retrain init)...", flush=True)
    W_init_repl = (A_frozen @ Vh_init).to(W_orig.dtype).to(DEVICE)
    c_proj.weight.data.copy_(W_init_repl)
    init_ppl = eval_ppl(model)
    print(f"  init PPL (rank-{RANK} no-retrain): {init_ppl:.4f}  (cycle 280 ref {CYCLE_280_INIT_REF})", flush=True)
    assert abs(init_ppl - CYCLE_280_INIT_REF) < 0.05, (
        f"init PPL {init_ppl:.4f} != cycle 280 ref {CYCLE_280_INIT_REF}"
    )
    # Restore original weight before continuing.
    c_proj.weight.data.copy_(W_orig)

    # ── Unigram histogram on calib slice (25,600 tokens) ─────────────────────
    print(f"\nComputing unigram histogram on {N_CALIB}x{SEQ_LEN}={N_CALIB*SEQ_LEN} tokens...", flush=True)
    vocab_size = model.config.vocab_size
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    n_tokens_total = 0
    for ids in calib_ids:
        flat = ids.view(-1).long()
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
        n_tokens_total += flat.numel()
    assert n_tokens_total == N_CALIB * SEQ_LEN
    p = counts / n_tokens_total
    eps = 1.0 / (n_tokens_total * 10)  # << min observed freq, prevents log(0)
    log_p = torch.log(p + eps).float()  # (50257,)
    n_observed_tokens = int((counts > 0).sum().item())
    print(f"  observed unique tokens: {n_observed_tokens} / {vocab_size}", flush=True)
    print(f"  log_p stats: min={float(log_p.min()):.3f}  max={float(log_p.max()):.3f}  "
          f"mean={float(log_p.mean()):.3f}", flush=True)

    # ── Build d1 = wte.T @ log_p, normalised ─────────────────────────────────
    wte = model.transformer.wte.weight.detach().float().cpu()  # (50257, 768)
    assert wte.shape == (vocab_size, 768)
    d1_raw = wte.T @ log_p  # (768,)
    d1_norm = float(torch.linalg.norm(d1_raw).item())
    d1 = d1_raw / d1_norm
    print(f"\nd1 = wte^T @ log_p:  ||d1_raw|| = {d1_norm:.4f}; "
          f"d1 unit norm = {float(torch.linalg.norm(d1)):.6f}", flush=True)

    # ── Build stopword indicator and d2_raw ──────────────────────────────────
    stopword_ids = set()
    for word in STOPWORD_BASE:
        for variant in (word, " " + word):
            try:
                ids = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            if len(ids) == 1:
                stopword_ids.add(int(ids[0]))
    print(f"\nStopword indicator: {len(stopword_ids)} unique single-token ids "
          f"(target ~300)", flush=True)
    assert len(stopword_ids) >= 200, (
        f"got only {len(stopword_ids)} stopword ids (need >= 200)"
    )
    s_vec = torch.zeros(vocab_size, dtype=torch.float32)
    for tid in stopword_ids:
        s_vec[tid] = 1.0
    d2_raw = wte.T @ s_vec  # (768,)
    d2_raw_norm = float(torch.linalg.norm(d2_raw).item())
    # Gram-Schmidt: project off d1
    proj = float((d2_raw @ d1).item())
    d2_orth = d2_raw - proj * d1
    d2_norm = float(torch.linalg.norm(d2_orth).item())
    d2 = d2_orth / d2_norm
    cos_d1_d2 = float((d1 @ d2).item())
    print(f"d2 = wte^T @ s_vec - <d2,d1>d1: ||d2_raw||={d2_raw_norm:.4f}  "
          f"<d2_raw,d1>={proj:.4f}  ||d2_orth||={d2_norm:.4f}  "
          f"|d1.d2|={abs(cos_d1_d2):.2e}", flush=True)
    assert abs(cos_d1_d2) < 1e-5, f"Gram-Schmidt failed: |cos|={abs(cos_d1_d2):.2e}"

    # ── Grid search over (s1, s2) ────────────────────────────────────────────
    print(f"\n=== Grid search over (s1, s2) in {GRID} x {GRID} ({len(GRID)**2} points) ===",
          flush=True)
    A_dev = A_frozen.to(W_orig.dtype).to(DEVICE)
    grid_results = []
    best = None
    for s1 in GRID:
        for s2 in GRID:
            B_synth = torch.stack([
                (s1 * mag1_base) * d1,
                (s2 * mag2_base) * d2,
            ], dim=0).to(W_orig.dtype).to(DEVICE)  # (2, 768)
            W_repl = A_dev @ B_synth  # (3072, 768)
            c_proj.weight.data.copy_(W_repl)
            t_e = time.time()
            ppl = eval_ppl(model)
            dt = time.time() - t_e
            print(f"  s1={s1:4.1f}  s2={s2:4.1f}  PPL={ppl:9.4f}  ({dt:.1f}s)",
                  flush=True)
            grid_results.append({
                "s1": float(s1),
                "s2": float(s2),
                "ppl": float(ppl),
                "wall_s": float(dt),
            })
            if best is None or ppl < best["ppl"]:
                best = {"s1": float(s1), "s2": float(s2), "ppl": float(ppl)}
    # Restore original
    c_proj.weight.data.copy_(W_orig)

    print(f"\nBest synth: s1*={best['s1']}, s2*={best['s2']}, "
          f"PPL*={best['ppl']:.4f}", flush=True)

    # ── Verdict per item 318 ─────────────────────────────────────────────────
    if best["ppl"] <= 60.0:
        verdict = "CONFIRMED"
    elif best["ppl"] >= 100.0:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n=== Verdict (item 318): {verdict} ===", flush=True)
    print(f"  best_synth_ppl = {best['ppl']:.4f}", flush=True)
    print(f"  CONFIRMED iff <= 60   ({best['ppl'] <= 60.0})", flush=True)
    print(f"  REFUTED   iff >= 100  ({best['ppl'] >= 100.0})", flush=True)

    # ── Save JSON ────────────────────────────────────────────────────────────
    result = {
        "cycle": 308,
        "item": 318,
        "verdict": verdict,
        "best_synth": best,
        "float_ppl": float(float_ppl),
        "cycle_280_init_ppl_reproduced": float(init_ppl),
        "cycle_280_best_ref": CYCLE_280_BEST_REF,
        "n_observed_tokens": n_observed_tokens,
        "vocab_size": vocab_size,
        "n_calib_tokens": N_CALIB * SEQ_LEN,
        "n_stopword_ids": len(stopword_ids),
        "d1_raw_norm": d1_norm,
        "d2_raw_norm": d2_raw_norm,
        "d2_orth_norm": d2_norm,
        "d1_d2_cos": cos_d1_d2,
        "S_top5": [float(x) for x in S[:5].tolist()],
        "grid": GRID,
        "grid_results": grid_results,
        "config": {
            "TARGET_BLOCK": TARGET_BLOCK,
            "RANK": RANK,
            "N_CALIB": N_CALIB,
            "N_EVAL": N_EVAL,
            "SEQ_LEN": SEQ_LEN,
            "SEED": SEED,
            "DEVICE": DEVICE,
            "GRID": GRID,
            "STOPWORD_BASE_count": len(STOPWORD_BASE),
        },
        "wall_s_total": float(time.time() - t0_total),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}", flush=True)

    # ── Markdown summary ─────────────────────────────────────────────────────
    lines = []
    lines.append(f"# Cycle 308 Summary — Item 318: Synthetic rank-2 unigram projector at block 1\n")
    lines.append(f"**Verdict: {verdict}**\n")
    lines.append(f"- best_synth_ppl = **{best['ppl']:.4f}** at (s1, s2) = ({best['s1']}, {best['s2']})")
    lines.append(f"- float_ppl = {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})")
    lines.append(f"- cycle_280 no-retrain init PPL reproduced = {init_ppl:.4f} (ref {CYCLE_280_INIT_REF})")
    lines.append(f"- cycle_280 trained best (reference, NOT this cycle) = {CYCLE_280_BEST_REF}")
    lines.append("")
    lines.append("## Verdict logic (item 318 verbatim)\n")
    lines.append("- CONFIRMED iff `best_synth_ppl <= 60`")
    lines.append("- REFUTED   iff `best_synth_ppl >= 100`")
    lines.append("- INCONCLUSIVE otherwise.\n")
    lines.append("## Synthetic construction\n")
    lines.append(f"- A_frozen = U[:, :2] @ diag(S[:2]) (cycle 280 LEFT factor, frozen, no training)")
    lines.append(f"- d1 = wte^T @ log_p, normalised (||d1_raw|| = {d1_norm:.4f})")
    lines.append(f"- d2 = wte^T @ s_vec, Gram-Schmidt vs d1, normalised (||d2_raw|| = {d2_raw_norm:.4f}, |d1.d2| = {abs(cos_d1_d2):.2e})")
    lines.append(f"- Stopword indicator vocabulary: {len(stopword_ids)} unique single-token ids "
                 f"from {len(STOPWORD_BASE)} base strings (each encoded with and without leading space)")
    lines.append(f"- B_synth(s1, s2) = stack([s1 * d1, s2 * d2])  shape (2, 768)")
    lines.append(f"- replacement: c_proj_1.weight <- A_frozen @ B_synth")
    lines.append(f"- Trainable params = 0 (grid search over 2 scalars is not training)\n")
    lines.append("## Grid (PPL per (s1, s2))\n")
    lines.append("| s1\\s2 | " + " | ".join(f"{s:>8.1f}" for s in GRID) + " |")
    lines.append("|---:|" + "|".join(["---:" for _ in GRID]) + "|")
    grid_by_pair = {(g["s1"], g["s2"]): g["ppl"] for g in grid_results}
    for s1 in GRID:
        row = [f"{s1:>4.1f}"]
        for s2 in GRID:
            row.append(f"{grid_by_pair[(s1, s2)]:8.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Config\n")
    lines.append(f"`TARGET_BLOCK={TARGET_BLOCK}, RANK={RANK}, N_CALIB={N_CALIB}, N_EVAL={N_EVAL}, "
                 f"SEQ_LEN={SEQ_LEN}, SEED={SEED}, DEVICE={DEVICE}, GRID={GRID}`\n")
    lines.append(f"Wall: {result['wall_s_total']:.1f} s\n")
    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT_SUMMARY}", flush=True)

    print(f"\nTotal wall: {time.time() - t0_total:.1f} s", flush=True)
    print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
