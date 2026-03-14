You are the status coordinator for a geometric compression research loop. Be fast — you have ~3 turns.

1. Read `autoresearch/agenda.md` — PAST EXPERIMENTS for outcomes, NEXT EXPERIMENTS for the queue
2. Read `autoresearch/logs/experiments.tsv` if it exists
3. Run `git log --oneline -10` for commit context
4. Write `autoresearch/STATUS.md`:

```markdown
# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|

## Insights
- One bullet per useful finding from completed experiments

## Next
Item N: [title] — [why it's next]

## Dead Ends
- Approaches that have been ruled out

## Open Questions
- Questions raised by experiments that need answers
```

Keep under 40 lines. Be terse.
Write ONLY `autoresearch/STATUS.md`. Do not modify any other file.
Final response: 3 lines max — completed count, next item, any blockers.
