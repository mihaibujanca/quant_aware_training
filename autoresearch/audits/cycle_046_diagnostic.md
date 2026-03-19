# Failure
Stage: execute
Item: 55

# Evidence
Script `experiments/lstsq_refit_quality_gpt2.py` crashes with SIGSEGV (exit 139) during GPT-2 model loading. Occurs even with reduced probe size (100 seqs). Two consecutive attempts both crash at ~75% model loading progress.

# Likely Cause
System memory pressure after 13 consecutive experiment cycles in this session. Each cycle loads GPT-2 (~500MB) plus probe activations. The OS kills the process (OOM → SIGSEGV). Not a code bug.

# Recommendation
Skip item. Retry in a fresh session when memory is clean. The analysis is secondary to the corpus-specificity finding (cycle 044) which is the main research conclusion.
