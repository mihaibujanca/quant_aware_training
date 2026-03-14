quantization correction: can you front-load the errors by making earlier layers orthonormal and correcting later? we've already seen that the initial layers have an outsized impact and that makes sense, they're the first to chop up the biggest space and subsequent layers work on subspaces. so maybe it makes sense to keep the most linearly correctible error (or no error)in initial layers and delaying correction until later layers?I thin

New insight: only look at linearly correctable points for estimating low-rank correction.
