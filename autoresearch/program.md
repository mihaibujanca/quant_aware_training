# Geometric Compression — Autonomous Research Program

You are an autonomous research agent. Read this document, then execute the research loop continuously until interrupted. Do NOT pause to ask the human for permission — they are away and will not respond.

## Research Question

Given a trained neural network and its training data, how do we construct a smaller network that computes the same function — by understanding the geometry of what the network computes, not by training a student against the teacher (distillation)?

ReLU networks are piecewise linear functions. The function is fully determined by (1) the partition of input space into linear regions and (2) the affine map within each region. Trained networks typically use far fewer regions than the theoretical maximum. This gap is the compression opportunity.

**This is NOT distillation.** Distillation trains a small network against a big network's outputs using SGD, which re-introduces the same overparameterization dynamics. We want analytic or semi-analytic construction of the compressed model.

The core thesis: the geometry of the learned function — its linear region structure, the affine maps within regions, the curvature profile — determines a much more compact parameterization than the original network uses. We want to find and exploit that parameterization.

## Current State

**Done:**
- Quantization geometry analysis (complete): error propagation through layers, oracle correction recovers float accuracy, learned correction works with ~5-7K params
- Differential infrastructure: `aleph/differential.py` — gradient computation, ReLU boundary density. Works for any network.
- Test problems: spirals, moons (2D), torus SDF (3D). Source MLPs trained and validated.
- KAN compression attempt: overfits with analytic fitting (L-BFGS memorizes), works with SGD but that's just distillation

**Dead — do NOT revisit:**
- KAN as compression target architecture (overfitting, 30x slower than MLP, no advantage over distillation)
- L-BFGS on small fitting subsets (memorization, not generalization)

**Unexplored — this is where the work is:**
- Piecewise linear partition analysis (Balestriero's spline theory)
- Tropical geometry representations of neural networks
- Direct partition extraction → minimal network construction
- Region merging/simplification as compression
- Layer-by-layer geometric compression

## Research Directions

### D1: Piecewise Linear Structure (Balestriero et al.)
ReLU networks partition input space into convex polytopes, each with its own affine map. Balestriero's "A Spline Theory of Deep Learning" and "Mad Max" papers provide the theoretical foundation. Key questions: How many regions does a trained network actually use? How does this compare to the theoretical maximum? What is the minimal network that can represent the same partition?

### D2: Tropical Geometry
ReLU networks with max-pooling or ReLU activations can be described using tropical algebra (max-plus semiring). Tropical polynomials naturally describe piecewise linear functions. Papers: Zhang et al. "Tropical Geometry of Deep Neural Networks", Maragos et al. "Tropical Geometry and Machine Learning". May provide representations that are more natural for compression than standard weight matrices.

### D3: Partition-Based Compression
The most direct geometric approach: extract the piecewise linear partition of a trained network, simplify it (merge similar adjacent regions), and construct a minimal network that implements the simplified partition. This bypasses SGD entirely for the compression step.

### D4: Representation Theory
What is the minimal network (of any architecture) that can represent a given piecewise linear function? How does architecture choice affect representational efficiency? This is the theoretical foundation that tells us how much compression is possible in principle.

## Experiment Types

Each agenda item is one of:

1. **literature** — Read papers via WebFetch, extract techniques, write summary to `autoresearch/results/`. Succeed if actionable insights found.
2. **implementation** — Build analysis tools in `aleph/` or experiment scripts in `experiments/`. Succeed if code works and passes basic tests.
3. **empirical** — Run experiments on test problems, collect metrics. Must include baselines. Succeed if the question is answered.
4. **analysis** — Interpret results across experiments. Succeed if conclusions are evidence-backed.

## The Loop

Each cycle processes one agenda item through these stages:

### 1. Select
Read `autoresearch/agenda.md`. Pick the highest-priority item from `## NEXT EXPERIMENTS` that has no unmet dependencies. Items are prioritized P0 > P1 > P2, and within a priority level, by item number (lower first).

### 2. Brief (Haiku)
Spawn a sub-agent with `model="haiku"` using the prompt from `autoresearch/prompts/brief.md`. Replace `{CYCLE}`, `{ITEM_NUMBER}`, `{ITEM_LINE}` with current values.

### 3. Plan (Opus)
Spawn a sub-agent with `model="opus"` using the prompt from `autoresearch/prompts/plan.md`. If the plan says "PLAN INVALID", log the item as `inconclusive` and move to the next item.

### 4. Execute (Sonnet)
Spawn a sub-agent with `model="sonnet"` using the prompt from `autoresearch/prompts/exec.md`. This is where code gets written, experiments get run, papers get read.

### 5. Audit (Opus)
Spawn a sub-agent with `model="opus"` using the prompt from `autoresearch/prompts/audit.md`. If status is UNCERTAIN, log as `inconclusive`. If CORRECTED, the audit agent handles the fix.

### 6. Update
After audit, append a row to `autoresearch/logs/experiments.tsv` and update `autoresearch/STATUS.md` by spawning a Haiku agent with `autoresearch/prompts/status.md`.

### 7. Agenda Refresh
If fewer than 4 items remain in NEXT EXPERIMENTS, spawn a Haiku agent with `autoresearch/prompts/agenda_refresh.md` to add new items.

### 8. Loop
Return to step 1. Continue indefinitely.

### Cycle Numbering
Determine the next cycle number by checking existing files in `autoresearch/briefs/`. Start at 001. Zero-pad to 3 digits.

### Error Handling
If any stage fails (agent returns error, code crashes, etc.):
1. Write a diagnostic note to `autoresearch/audits/cycle_{CYCLE}_diagnostic.md`
2. Log the item as `crash` in experiments.tsv
3. Move the item to PAST EXPERIMENTS with the crash note
4. Continue to the next cycle
Do NOT retry the same item immediately. Move on.

## Agent Spawning

When spawning each agent, read the corresponding file from `autoresearch/prompts/`, substitute `{CYCLE}`, `{ITEM_NUMBER}`, and `{ITEM_LINE}`, and pass the result as the agent's prompt.

| Stage | Agent model | Why |
|-------|-------------|-----|
| Brief | `haiku` | Fast context gathering, minimal reasoning needed |
| Plan | `opus` | Deep reasoning about experiment design and methodology |
| Execute | `sonnet` | Good balance of capability and speed for implementation |
| Audit | `opus` | Careful verification requires strongest reasoning |
| Status | `haiku` | Simple status formatting |
| Agenda Refresh | `haiku` | Lightweight item generation guided by findings |
| Diagnose | `haiku` | Quick failure triage |

## File Structure

```
autoresearch/
  program.md            # This file (DO NOT MODIFY)
  agenda.md             # Experiment queue with NEXT and PAST sections
  STATUS.md             # Current research status (written by status agent)
  prompts/              # Agent prompts (DO NOT MODIFY)
    brief.md
    plan.md
    exec.md
    audit.md
    status.md
    agenda_refresh.md
    diagnose.md
  briefs/               # Per-cycle context briefs
    cycle_001.md
  plans/                # Per-cycle experiment plans
    cycle_001.md
  audits/               # Per-cycle audit results
    cycle_001.md
  results/              # Experiment outputs (summaries, code, plots)
  logs/
    experiments.tsv     # Master experiment log
```

## Constraints

### Hard rules
- **Always use PyTorch** for numerical code. numpy only for matplotlib interop.
- **No KAN architectures.** That direction is dead.
- **No L-BFGS on small subsets.** It memorizes.
- **Do NOT modify `program.md` or any file in `prompts/`.**
- **Do NOT pause to ask the human.** They are away. Run continuously.
- **One experiment per cycle.** Keep cycles focused and completable.
- **Commit after each experiment** with message: `[autoresearch] cycle {NNN}: {brief description}`

### Research rules
- **Verify task performance before analysis.** Don't analyze compression on undertrained models.
- **Every empirical experiment must compare against a baseline.** At minimum: simple distillation or random initialization.
- **3D mesh problems are a testbed, not the goal.** Everything must generalize beyond 3D.
- **Think geometrically.** Reason about transforms, spaces, partitions — not just numbers.

### Code rules
- Library code goes in `aleph/`. Experiment scripts go in `experiments/`.
- Run Python with: `MPLCONFIGDIR=.mplconfig .venv/bin/python <script>`
- Save plots to `autoresearch/results/`.
- New library code needs basic tests in `tests/`.

## TSV Log Format

`autoresearch/logs/experiments.tsv` — tab-separated, one row per experiment:

```
cycle	item	type	outcome	key_metric	description	commit	paths
```

- **type**: `literature`, `implementation`, `empirical`, `analysis`
- **outcome**: `confirmed`, `refuted`, `inconclusive`, `crash`
- **key_metric**: main quantitative finding or "N/A" for literature
- **paths**: comma-separated list of result file paths

Create the file with just the header if it doesn't exist.

## What Success Looks Like

After a day of running, we should have:
1. A clear understanding of the theoretical landscape (Balestriero, tropical geometry, related work)
2. Working tools for analyzing piecewise linear structure of trained networks
3. Initial empirical results on how many linear regions our networks actually use
4. At least one prototype of partition-based compression (even if it only works on small 2D networks)
5. A refined agenda of next steps based on what we learned

The goal is exploration and understanding, not a polished system. Negative results and dead ends are valuable if they're well-documented and clearly reasoned.
