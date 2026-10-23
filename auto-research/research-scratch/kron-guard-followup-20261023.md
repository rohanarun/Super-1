# E01 handoff: qualify a factored core with an interval fallback

Status: proposed, not authorized, implemented or executed. This note follows completed [PR #26](https://github.com/rohanarun/Super-1/pull/26), not an extension of its compute approval. Historical approval and unused CPU allowance do not authorize this experiment.

## Question and predecessor evidence

Can a wrapper retain the ordinary factored path and use interval products only when a nonzero log increment is rounded away, while passing a wider fixed fidelity grid?

The [completed bounded canary](https://github.com/thakrarsagar/Super-1/blob/de935c82d4d50efbd1a647bd6cb01a0d206bb28d/auto-research/reports/kron-decay-canary-20261023-v1.md) supplies both reasons to test this and reasons for caution:

- Source factored FP32 passes the single k16/25101 model row but fails16 of50 core stress cases.
- Interval-direct and interval-scaled pass all50 core cases but fail that model's parameter-gradient gate. Scaling alone preserves the prefix failure.
- The observed FP32 two-token prefixes for g=[-2^27,-1] coincide, producing .75 instead of .5919698476791382. The centered range is zero. FP64 has the analogous loss at2^56.
- All current model variants pass the one-step optimizer check, including variants whose gradient gate fails. An optimizer-only screen would miss that distinction.

These are reused mechanics fixtures, not independent confirmation, language results or the historical failure's root cause.

## Proposed policy — sketch only

Keep the parent equations, gates, normalization floor and reset semantics. In the same compute dtype as the source, detect an exactly lost nonzero increment:

```python
g_compute = g.to(source.acc_dtype(q))
prefix = g_compute.cumsum(dim=1)
lost = ((prefix[:, 1:] - prefix[:, :-1]) == 0) & (g_compute[:, 1:] != 0)
use_interval = bool(lost.detach().any()) if g_compute.shape[1] > 1 else False
# True: existing experimental unscaled interval core.
# False: unchanged source kron_chunk_factored.
```

This is a proposed dispatch predicate, not a tested implementation. It detects complete loss of an increment, not every inaccurate nonzero difference. A chunk-wide predicate can choose a different rounding path when a future token or chunk boundary changes; causal/streaming and gradient checks remain necessary. No runtime benefit is inferred; it adds work and sometimes selects the quadratic interval construction. No epsilon or threshold should be tuned from gate outcomes.

## Concrete next experiment to preregister and approve

Create a new run ID and freeze exact source/protocol hashes before execution. Suggested bounded matrix:

1. Reuse all50 core cases from PR #26. Compare source factored, unscaled interval and the new wrapper against the same sequential reference.
2. Add three explicitly new scalar controls in both FP32 and FP64: one-token g=[-2^27], S0=1, no reset; two-token g=[0,0], S0=1, no resets; three-token g=[-2^27,-1,-1], S0=1, before-token reset at index1. All have q=k=v=1 and beta=write=.5. This gives56 cases/variant,168 core comparisons. Retain branch decisions for full/chunked/intervened calls.
3. Repeat the original [qualify.py mechanics construction](https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/qualify.py#L26-L67): five Kron arms, seeds25101/25102, parameter dtypes FP32/FP64, B2/T19, resets0/7/13 and the original nonzero convolution taps. Compare wrapper and unchanged factored path separately with paired sequential references:20 rows per variant. Add the10 FP32-parameter/FP64-core positive controls and the four unchanged Transformer rows. Total54 model pairs, at most one AdamW step per copy (108 synthetic steps), no corpus training or saved checkpoints.
4. Check all logits/state, loss, every live input/parameter gradient, one-step parameters and optimizer state, streaming, reset and causality. Freeze the existing atol/rtol values unchanged. Primary core gate and wider model gate remain separate. Preserve all failures; any matrix/source change needs a new record and approval.

Suggested ceiling for a new approver: one CPU Torch/inter-op/BLAS thread,300 cumulative numerical CPU seconds/600 execution wall seconds, zero paid spend, existing CPU2.9.1 environment or a separately approved local install, at most1GiB environment/temp/combined outputs and50MiB result artifacts. Enforce caps across attempts, preserve raw results/resource receipts and prohibit GPU/cloud, corpus/checkpoint download, language training and official-test access. This is a proposed budget, not approval.

Success is passing the frozen core and model mechanics gates with controls intact. Failure leaves a useful negative result and a specific gate for further work. Even a pass qualifies only the stated finite mechanics grid; long-sequence behavior, performance and language quality would remain untested.

## First step and credit

First read the completed raw result, source and [reset/adjoint clarification](https://github.com/thakrarsagar/Super-1/blob/7f63085a5ba6bc52d84a59757c3bfc2ee364bc1b/auto-research/research-scratch/kron-reset-source-clarification-20261023.md). Obtain exact approval for the new experiment, then implement and freeze the wrapper/harness without modifying archived sources. Do not run the old throughput/training section of qualify.py as part of a mechanics-only approval.

Credit: @JRowe47's original minKron adaptation/research in merged PR #15, Rohan Arun/project contributors and preserved NX-AI/KronLSTM MIT/provenance. The exact oracle and bounded canary are this session's AI-assisted follow-ups, not independent verification or a new architecture. This proposed handoff is substantially AI-assisted by Codex for @thakrarsagar. Session selection seed was dcbb91d07e51cf13ee6fda1fefc14ca2; the project has no seeded task picker.
