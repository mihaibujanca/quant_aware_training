cycle = 375
item = 383
title = Adapter dead-neuron census and exact pruning — ReLU MLP adapter (R2)
status = REFUTED
guardrails_flagged = none
key_result = 0/384 adapter neurons are dead or always-active; 100% mixed; pct_prunable=0.0%
implementation_correct = yes — dead defined as max(pre_relu)<=0, always_active as min(pre_relu)>0, both correct for ReLU; pre-relu recomputed faithfully from MLP input inside hook
reproducibility = adequate — single seed, deterministic census, 25600 tokens (200 seqs x 128) sufficient for 64-neuron layer
interpretation_fair = yes — 0% prunable is unambiguous; 25600 tokens provides ~400 samples per neuron dimension, more than enough to detect dead neurons
learning_quality = h=64 adapter is fully utilized under joint training (proj+adapter, 1500 Adam steps); to reduce adapter params, shrink h directly rather than prune
method_failure_risk = low — joint training co-adapts all params, which naturally distributes load; a frozen-projection setup might leave more neurons idle, but that is not the deployed config
interesting = no
