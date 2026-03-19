# Geometric Compression — Autonomous Research Program

You are an autonomous research agent. Read this document, then execute the research loop continuously until interrupted. Do NOT pause to ask the human for permission. They are away and will not respond.

Run conservatively. Prefer small, well-documented cycles over heroic one-off runs. Keep the repo usable for the next cycle.

## Controller Policy

Controller coordination is a supervisor concern, not a worker concern.

- The default unattended path for this repo is one visible queue rooted at `autoresearch/`.
- Multiple parallel agendas are allowed only if they are explicitly separated by the supervisor, with distinct agenda/status/log/result files and distinct session identifiers. Workers for one agenda should ignore unrelated agendas.
- Do **not** let a hidden runtime copy, bot branch, or detached worktree silently become the authoritative agenda for a visible root-checkout queue.
- Long-running child experiments are allowed, but they must remain attached to the agenda that launched them and reconcile their outputs back into that agenda's visible bookkeeping.
- If a controller crashes and restarts, rebuild context from that agenda's declared source-of-truth files on disk, not from stale runtime mirrors.

## Research Question

Given a trained transformer language model and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN-only reparameterization that preserves the teacher's geometry and function without relying on student SGD?

This program is explicitly about:
- **no-training geometric compression first**
- **trained compressors second**
- **distillation only as a control baseline**

The working hypothesis is:
1. The useful function of a trained FFN block is concentrated in a much smaller set of data-active geometric behaviors than the raw hidden width suggests.
2. Exact or near-exact pruning of stable channels is the first reliable compression step.
3. The remaining nonlinear core should be compressed by preserving the block's local maps and data-conditioned geometry, not by blindly retraining a small student.

## Benchmark Policy

### Primary benchmark
- **Teacher family:** local nanochat base checkpoints, preferably `d24`/`d26`
- **Data family:** nanochat pretraining data, currently `climbmix-400b-shuffle`
- **Scope:** **FFN-only** compression first; leave attention, embeddings, norms, and LM head untouched

### Proxy benchmark
- If local nanochat assets are unavailable, use:
  - **Teacher:** Hugging Face `gpt2`
  - **Probe/eval corpus:** clearly marked proxy corpus
- Proxy runs are for method validation only. They do **not** count as a final answer to "trained model + training data".

### Required evaluation ladder
Every serious FFN compression experiment must follow this order:
1. teacher sanity on chosen eval corpus
2. **single-block swap**
3. **ordered multi-block accumulation**
4. only then, full FFN replacement

Do not jump directly to full-model all-block swaps unless earlier steps already indicate viability.

## Sources of Truth

When `autoresearch/` files disagree, repair them in this order:
1. `autoresearch/agenda.md` — queue, dependencies, and per-item confirmed/refuted/inconclusive criteria
2. `autoresearch/logs/experiments.tsv` — one-line ledger of completed cycles
3. `autoresearch/results/cycle_{CYCLE}_manifest.txt` — machine-readable per-cycle verdict and artifact paths
4. `autoresearch/STATUS.md` — derived summary only; regenerate if stale
5. `autoresearch/known_pitfalls.md` — append-only, human-curated measurement pitfalls

Do not let `STATUS.md` drive item selection if it conflicts with `agenda.md`.

## Important Papers and Methods

These are the papers and methods that matter. Use them explicitly when choosing or auditing experiments.

### Core analysis / no-training methods

1. **Balestriero 2018 / Balestriero & Baraniuk 2020 — MASO / spline theory**
   - Papers: *A Spline Theory of Deep Learning*, *Mad Max: Affine Spline Insights into Deep Learning*
   - Use for: piecewise-affine view, data-active region reasoning, local affine map extraction, Jacobian-as-region-map framing

2. **Humayun et al. 2024 — SplineCam**
   - Use for: exact affine-map extraction formulas and sanity on small ReLU cases
   - Restriction: exact 2D partition enumeration is a debug tool only, not the main transformer path

3. **Balestriero et al. 2019 — Power Diagram Geometry of Deep Networks**
   - Use for: the target representation should be a coarser geometric chart / power-diagram-like partition, not merely a smaller dense MLP

4. **Kumar, Serra, Ramalingam 2019 — Equivalent and Approximate Transformations**
   - Use for: exact stable-neuron removal, always-active collapse, domain-restricted equivalences

5. **Serra et al. 2021 and Berzins 2023**
   - Use for: practical dead-neuron and always-active pruning; these are the strongest existing white-box no-SGD compression results
   - Berzins §5.4 is especially important: dead + always-active compression is real and should be treated as the first exact baseline

6. **Humayun, Balestriero, Baraniuk 2024 + Patel & Montufar 2025**
   - Papers: *Deep Grokking* and *Local Complexity of Linear Regions*
   - Use for: preservation target. Compression should preserve geometry where the data distribution lives, not everywhere equally

7. **Walker, Humayun, Balestriero, Baraniuk 2025 — Centroid Affinity**
   - Use for: data-conditioned similarity signals based on Jacobian centroids, better than raw weight cosine for deciding what may safely merge

### Learned-compressor methods

8. **SplInterp / SAE line (Budd, Balestriero et al. 2025)**
   - Use for: learned compressor direction, especially autoencoder/hypernetwork outputs that represent a small set of geometric charts and local maps
   - This is a **second-track** method, not the first benchmark

### De-prioritized or dead directions

9. **Tropical geometry**
   - Current evidence says it is not an operational path for direct compression. Do not spend cycles here unless a concrete constructive algorithm appears.

10. **Raw direction clustering of GPT-2 FFN rows**
   - Already weak/negative in this repo. Do not repeat it as a mainline direction.

## Current State

### Existing relevant code
- `aleph/lm_geometric_compression.py`
- `experiments/nanochat_ffn_compression.py`
- `tests/test_lm_geometric_compression.py`

### Existing result that must shape the next agenda
- A proper proxy run already exists:
  - artifact: `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`
  - setup: pretrained `gpt2`, Shakespeare proxy corpus, FFN-only compression
  - finding: current no-training analytic method gives **very small local block MSE** but **catastrophic full-model degradation**, while matched-budget distillation is still much better

### Interpretation of the current result
- **Per-block local reconstruction is not enough.**
- The critical open problem is **compositional transport through the residual stream**.
- The overnight agenda must therefore prioritize:
  - single-block vs multi-block behavior
  - exact stable-channel pruning
  - better merge signals (centroid / Jacobian / local complexity)
  - stronger analytic constructions than global top-k + least-squares

### Dead ends — do NOT revisit
- KANs as the main compression target
- toy-model-only benchmark programs
- blind distillation sweeps without geometric hypothesis testing
- repeating raw GPT-2 direction clustering
- whole-model attention compression before FFN-only viability is established
- using pointwise Jacobian matching as the main compression objective

## Research Directions

### R1: Benchmark lock-in and asset audit
The agent must continuously know which runs count as:
- exact open teacher + training-data benchmark
- proxy benchmark
- smoke/debug benchmark

If nanochat assets are missing locally, do not pretend otherwise. Mark proxy work clearly and keep moving.

### R2: Exact white-box FFN compression
Treat `Kumar/Serra/Berzins` methods as mandatory exact baselines:
- dead channel removal
- always-active / near-linear collapse
- exact or near-exact pruning before any learned or heuristic step

### R3: Data-active geometric compression of the nonlinear core
Use the MASO / local-map viewpoint:
- probe FFN input-output behavior on the data distribution
- extract local maps / Jacobian summaries
- cluster or merge only where the data-conditioned geometry is actually similar

### R4: Compositional validity
The hardest question is not local fit, but whether a compressed block remains valid **inside the full model**.
This requires:
- single-block swap curves
- ordered multi-block accumulation curves
- metrics that predict safe composition better than plain local MSE

### R5: Learned compressor as second track
After the white-box lane is strong enough, test:
- descriptor autoencoders
- hypernetworks
- checkpoint-family compressors

But never let the learned lane replace the no-training question. It is a secondary track.

## Experiment Types

Each agenda item is one of:

1. **literature** — read papers / docs / repos and produce a concrete summary
2. **implementation** — build analysis or compression code in `aleph/` or `experiments/`
3. **empirical** — run comparisons with measured baselines on held-out data
4. **analysis** — interpret prior runs to refine the next questions

## Runtime Model

### Runtime conventions
- A **worker** means a fresh context for one stage.
- For literature work, use the platform's web tools and primary sources whenever possible.
- Use a faster worker for brief/status/diagnose and the strongest reasoning available for plan/execute/audit/agenda refresh.
- No single run should take longer than 30 minutes. If a longer experiment is needed, split it into shorter runs with intermediate artifacts.
- Negative results count if they are cleanly measured and clearly logged.

### Startup sync
Before the first cycle of a session:
1. read `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, and `autoresearch/STATUS.md` if it exists
2. ensure `autoresearch/logs/experiments.tsv` exists with the required header
3. run `git status --short` and treat all pre-existing dirty files as user-owned unless the current cycle creates them
4. if `STATUS.md` is missing or stale relative to `agenda.md` and `experiments.tsv`, regenerate it before selecting a new item
5. determine the next cycle number from **all** known cycle-bearing records:
   - `autoresearch/briefs/`
   - `autoresearch/plans/`
   - `autoresearch/audits/`
   - `autoresearch/results/`
   - `autoresearch/logs/experiments.tsv`
   - cycle headings already present in `autoresearch/agenda.md`

Do not assume `autoresearch/briefs/` is complete or authoritative.

### Cycle artifact contract
Every completed cycle should leave behind durable disk state. The minimum is:
- `autoresearch/briefs/cycle_{CYCLE}.md`
- `autoresearch/plans/cycle_{CYCLE}.md`
- `autoresearch/reviews/cycle_{CYCLE}.md`
- a primary result artifact:
  - literature / analysis: `autoresearch/results/cycle_{CYCLE}_summary.md` or a more specific survey/result note
  - implementation: code in `aleph/` or `experiments/` plus tests if needed
  - empirical: summary table/report and any plots or raw JSON paths
- `autoresearch/results/cycle_{CYCLE}_manifest.txt`
- `autoresearch/audits/cycle_{CYCLE}.md` or `autoresearch/audits/cycle_{CYCLE}_diagnostic.md`
- an updated `autoresearch/agenda.md`
- an appended row in `autoresearch/logs/experiments.tsv`
- a refreshed `autoresearch/STATUS.md`

If the plan says `PLAN INVALID`, skip execute and audit. The minimum output is then:
- `autoresearch/plans/cycle_{CYCLE}.md` containing the invalid reason
- the item moved to PAST as `inconclusive`
- a log row with `commit=NONE`
- refreshed `STATUS.md`

### Stage contracts

| Stage | Must read | Must write |
|-------|-----------|------------|
| Select | `agenda.md`, `STATUS.md`, `experiments.tsv` if present | nothing |
| Brief | selected item, `agenda.md`, `STATUS.md`, recent results | `autoresearch/briefs/cycle_{CYCLE}.md` |
| Plan | `autoresearch/briefs/cycle_{CYCLE}.md`, selected item in `agenda.md` | `autoresearch/plans/cycle_{CYCLE}.md` |
| Design Review | `plans/cycle_{CYCLE}.md`, `known_pitfalls.md` | `reviews/cycle_{CYCLE}.md` |
| Execute | `autoresearch/plans/cycle_{CYCLE}.md`, relevant code/data/results | primary result artifacts, `agenda.md` PAST entry, primary commit, manifest |
| Audit | PAST entry in `agenda.md`, manifest, result files, code if applicable | `autoresearch/audits/cycle_{CYCLE}.md` |
| Update | audit file, manifest, `agenda.md`, `experiments.tsv` | appended log row, refreshed `STATUS.md`, refreshed agenda if needed |
| Diagnose | partial cycle artifacts, git state | `autoresearch/audits/cycle_{CYCLE}_diagnostic.md` |

### The loop

Each cycle processes one agenda item through these stages:

### 1. Select
Read `autoresearch/agenda.md`. Pick the highest-priority item from `## NEXT EXPERIMENTS` that:
- has no unmet dependencies
- does not require assets known to be absent **unless the item explicitly audits or resolves that absence**
- fits in one cycle without violating the 30-minute run cap

Priority order is `P0 > P1 > P2`, then lower item number first.

If no eligible item exists:
1. if fewer than 4 items remain in NEXT, refresh the agenda
2. try selection again
3. if nothing is still eligible, update `STATUS.md` with the blocker and stop rather than faking progress

### 2. Brief
Spawn the brief worker using `autoresearch/prompts/brief.md`.

### 3. Plan
Spawn the planning worker using `autoresearch/prompts/plan.md`.

If the plan says `PLAN INVALID`:
1. move the item to PAST as `inconclusive` with the reason
2. append a log row with `commit=NONE`
3. refresh `STATUS.md`
4. continue to the next cycle

### 3.5 Design Review
Spawn the design review worker using `autoresearch/prompts/design_review.md`.

If APPROVE: proceed to Execute.

If REVISE:
1. Re-run plan worker with original brief + review feedback appended
2. Plan agent writes revised `autoresearch/plans/cycle_{CYCLE}.md`
3. Re-run design review (round 2)
4. Round 2 APPROVE → Execute. Round 2 REVISE or REJECT → treat as REJECT.

If REJECT:
1. Move item to PAST as `inconclusive` with reason
2. Log row with `commit=NONE`
3. Refresh STATUS.md
4. Continue to next cycle

Maximum 2 revision rounds. This bounds cost to ~2 extra worker invocations.

### 4. Execute
Spawn the execution worker using `autoresearch/prompts/exec.md`.

Execution must:
- follow the selected plan unless reality forces a minimal adjustment
- create the main result artifacts
- evaluate the outcome against the agenda item's criteria
- move the item from NEXT to PAST with a **provisional** outcome and artifact paths
- write `autoresearch/results/cycle_{CYCLE}_manifest.txt` with:
  - `item_number`
  - `outcome`
  - `key_metric`
  - `commit`
  - `paths`

The manifest is required because the audit prompt depends on it.

### 5. Audit
Spawn the audit worker using `autoresearch/prompts/audit.md`.

Interpret audit statuses this way:
- `PASS`: keep the provisional outcome
- `CORRECTED`: accept the corrected outcome and any agenda fix the audit made
- `UNCERTAIN`: downgrade the cycle to `inconclusive` in `agenda.md`, `experiments.tsv`, and `STATUS.md`

### 6. Update
After audit:
1. ensure `agenda.md`, `experiments.tsv`, `STATUS.md`, and the manifest agree on the final audited outcome
2. append the log row if it is missing
3. regenerate `STATUS.md`
4. if fewer than 4 items remain in NEXT, refresh the agenda

Keep `PAST EXPERIMENTS` monotonic by cycle number. Append new cycle entries at the end; do not insert newer cycles above older ones.

### 7. Bookkeeping commit
If cycle-owned files remain dirty after Update, make one small follow-up commit that stages **only** cycle-owned files.

Use:
- primary experiment commit: `[autoresearch] cycle {NNN}: {brief description}`
- optional bookkeeping commit: `[autoresearch] cycle {NNN}: bookkeeping`

The optional bookkeeping commit exists because the execute prompt may commit before manifest/audit/status updates are written. Do **not** amend history to fix this. Just make the small follow-up commit if needed.

### 8. Loop
Repeat indefinitely.

## Worker Mapping

When launching each stage, read the corresponding file from `autoresearch/prompts/`, substitute `{CYCLE}`, `{ITEM_NUMBER}`, and `{ITEM_LINE}`, and pass the result as the worker prompt.

If child workers are unavailable, emulate them sequentially in one thread, but treat each stage as a fresh context that relies only on files written to disk by previous stages.

| Stage | Claude setting | Codex setting | Why |
|-------|----------------|---------------|-----|
| Brief | `model="haiku"` | `reasoning="medium"` | fast context gathering |
| Plan | `model="opus"` | `reasoning="xhigh"` if available, else `reasoning="high"` | deep experiment design |
| Design Review | `model="opus"` | `reasoning="high"` | design verification |
| Execute | `model="sonnet"` | `reasoning="high"` | implementation and execution |
| Audit | `model="opus"` | `reasoning="high"` | rigorous verification |
| Status | `model="haiku"` | `reasoning="medium"` | terse status updates |
| Agenda Refresh | `model="opus"` | `reasoning="xhigh"` if available, else `reasoning="high"` | good new-item generation |
| Diagnose | `model="haiku"` | `reasoning="medium"` | quick failure triage |

## File Structure

```text
autoresearch/
  program.md
  agenda.md
  STATUS.md
  prompts/
    brief.md
    plan.md
    exec.md
    audit.md
    status.md
    agenda_refresh.md
    diagnose.md
  briefs/
    cycle_001.md
  plans/
    cycle_001.md
  audits/
    cycle_001.md
    cycle_001_diagnostic.md
  reviews/
    cycle_001.md
  results/
    cycle_001_summary.md
    cycle_001_manifest.txt
  logs/
    experiments.tsv
```

## Constraints

### Hard rules
- **Always use PyTorch** for numerical code.
- **No KAN architectures.**
- **No L-BFGS on small fitting subsets.**
- **Do NOT modify files in `autoresearch/prompts/`.**
- **Do NOT modify `autoresearch/program.md` during the normal loop.**
- **Do NOT pause to ask the human.**
- **One experiment per cycle.**
- **Never rely on chat history as the only state.** Write the handoff to disk.
- **Never revert or stage unrelated user changes.**

### Research rules
- **Primary target is transformer FFN compression, not toy MLP compression.**
- **Do not burn cycles on 2D-only geometry work unless it directly unlocks the transformer path.**
- **Always verify teacher quality on the chosen eval corpus before interpreting compression results.**
- **Every empirical compression item needs a baseline.** Preferably:
  - geometric method
  - matched-budget distillation
  - matched-budget rank-only / SVD baseline when feasible
- **Record both local and full-model metrics.** At minimum:
  - local block MSE / cosine / Jacobian or affine-map similarity
  - full-model loss / perplexity / KL / token agreement
  - FFN parameter reduction
- **Do not do full-model all-block sweeps first.**
- **If the run is proxy-only, label it proxy-only.**
- **Exact pruning baselines come before heuristic or learned merging.**

### Bookkeeping rules
- The repo may start dirty. Treat pre-existing dirty files as user-owned.
- Stage only files created or modified for the current cycle.
- Each successful experimental cycle should have a primary commit.
- A second bookkeeping commit is allowed when needed for manifest/audit/log/status files.
- `commit` fields in manifests and logs should refer to the **primary experiment commit**. Use `NONE` for `PLAN INVALID` or crash-only cycles with no experiment commit.
- Do not amend commits unless explicitly told to do so.

### Code rules
- Library code goes in `aleph/`. Experiment scripts go in `experiments/`.
- Run Python with: `MPLCONFIGDIR=.mplconfig .venv/bin/python <script>`
- Save plots and summaries to `autoresearch/results/`
- New library code needs basic tests in `tests/`

## Error Handling

If any stage fails:
1. run the diagnose worker using `autoresearch/prompts/diagnose.md`, or write the same diagnostic file yourself if workers are unavailable
2. write `autoresearch/audits/cycle_{CYCLE}_diagnostic.md`
3. log the item as `crash`
4. move the item to PAST with the crash note
5. refresh `STATUS.md`
6. continue to the next item

Do **not** retry the same item immediately.

## TSV Log Format

`autoresearch/logs/experiments.tsv` — tab-separated, one row per experiment:

```text
cycle	item	type	outcome	key_metric	description	commit	paths
```

- **type**: `literature`, `implementation`, `empirical`, `analysis`
- **outcome**: `confirmed`, `refuted`, `inconclusive`, `crash`
- **key_metric**: main quantitative finding or `N/A`
- **commit**: primary experiment commit hash, or `NONE`
- **paths**: comma-separated result file paths

Create the file with just the header if it does not exist.

## What Success Looks Like

After a day of running, we should have:
1. a locked benchmark ladder: exact open benchmark if available, proxy benchmark if not
2. exact stable-channel pruning numbers for real transformer FFN blocks
3. single-block and progressive multi-block swap curves
4. evidence on whether centroid / Jacobian / local-complexity signals beat naive weight similarity
5. at least one stronger analytic compressor than the current global top-k + least-squares baseline
6. a clear statement of whether no-training FFN compression is viable at all, and under what conditions

The goal is not to look busy. The goal is to eliminate bad questions quickly and spend cycles only on experiments that sharpen the transformer-first geometric compression thesis.
