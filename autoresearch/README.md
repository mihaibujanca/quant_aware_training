# Running `autoresearch`

`autoresearch/` is a file-driven autonomous research loop. [`program.md`](/Users/mihai/Projects/quant_aware_training/autoresearch/program.md) is the controller. The prompt files in [`prompts/`](/Users/mihai/Projects/quant_aware_training/autoresearch/prompts) define each stage. The handoff state lives on disk in [`briefs/`](/Users/mihai/Projects/quant_aware_training/autoresearch/briefs), [`plans/`](/Users/mihai/Projects/quant_aware_training/autoresearch/plans), [`results/`](/Users/mihai/Projects/quant_aware_training/autoresearch/results), [`audits/`](/Users/mihai/Projects/quant_aware_training/autoresearch/audits), and [`logs/`](/Users/mihai/Projects/quant_aware_training/autoresearch/logs).

## Runtime Mapping

| Stage | Claude | Codex |
|-------|--------|-------|
| Brief | `model="haiku"` | `reasoning="medium"` |
| Plan | `model="opus"` | `reasoning="high"` |
| Execute | `model="sonnet"` | `reasoning="high"` |
| Audit | `model="opus"` | `reasoning="high"` |
| Status | `model="haiku"` | `reasoning="medium"` |
| Agenda Refresh | `model="opus"` | `reasoning="high"` |
| Diagnose | `model="haiku"` | `reasoning="medium"` |

If Codex cannot spawn child workers in your setup, run the stages sequentially in one thread or separate fresh threads. The important part is that each stage reads the prior stage's file output from disk instead of relying on chat history.

## Codex Kickoff

Use a fresh Codex thread at the repo root with web access enabled. Start with:

```text
Read `autoresearch/program.md` and execute the autonomous research loop continuously until interrupted. Use the Codex worker mapping in that file. If child workers are unavailable, emulate them by treating each stage as a fresh context and using only the files on disk as stage handoff state.
```

Expected environment:
- Current working directory is the repository root.
- `.venv` is available for Python runs.
- Git is available for per-cycle commits.
- Web access is enabled for literature items.
