# Completed decay canary: optimizer coverage clarification

Date: 2026-10-23. This is an additive post-run scope clarification for
`kron-decay-canary-20261023-v1`, discovered during final verification. It does
not amend the frozen protocol, implementation, raw receipts, tolerances or
results, and no additional tensor or model computation was performed.

## The wording discrepancy

The frozen primary-success sentence includes “optimizer mechanics tolerances
on every declared core case”. That is broader than the executed checks. The
same approved protocol restricts synthetic optimizer steps to the one named
model qualification row, and lists the AdamW step and optimizer state among
that row's checks. See the pinned [primary rule and authorization](https://github.com/thakrarsagar/Super-1/blob/de935c82d4d50efbd1a647bd6cb01a0d206bb28d/auto-research/results/kron-decay-canary-20261023-v1/CANARY_PROTOCOL.json#L13-L27)
and [model-row design](https://github.com/thakrarsagar/Super-1/blob/de935c82d4d50efbd1a647bd6cb01a0d206bb28d/auto-research/results/kron-decay-canary-20261023-v1/CANARY_PROTOCOL.json#L111-L128).

No per-core-case optimizer experiment was run. The completed run therefore
does **not** establish that broader literal primary-success sentence.

## What the recorded gates establish

- Each of the 50 synthetic core cases per variant checks output/final state,
  gradients of the seven core inputs, streaming, reset isolation where a
  boundary exists, causality and finiteness. The recorded
  `primary_core_gate_passed` flag is the conjunction of those core fields for
  the interval-direct variant; it contains no optimizer gate. See the pinned
  [core check](https://github.com/thakrarsagar/Super-1/blob/de935c82d4d50efbd1a647bd6cb01a0d206bb28d/auto-research/results/kron-decay-canary-20261023-v1/tensor_canary.py#L94-L140)
  and [recorded gate definition](https://github.com/thakrarsagar/Super-1/blob/de935c82d4d50efbd1a647bd6cb01a0d206bb28d/auto-research/results/kron-decay-canary-20261023-v1/tensor_canary.py#L257-L263).
- Optimizer mechanics were checked only on the named fixed `k16/25101` model
  row: one AdamW step each for candidate and reference, across five variants
  plus one labeled mixed-core positive control. That is six pairs and twelve
  synthetic steps in total. All six pairs passed the parameter-update and
  optimizer-state comparisons. See the pinned [optimizer checks](https://github.com/thakrarsagar/Super-1/blob/de935c82d4d50efbd1a647bd6cb01a0d206bb28d/auto-research/results/kron-decay-canary-20261023-v1/tensor_canary.py#L231-L235).

The bounded findings remain: both interval variants pass all 50 **specified
core checks**, while both fail the separate FP32 model parameter-gradient
gate. Those core passes are not full protocol qualification, per-core
optimizer verification, or general FP32 model qualification. Every failed
gate remains in the completed evidence. No extra experiment is authorized
by this clarification or by unused numerical budget.

Credit: @JRowe47's PR #15, Rohan Arun and project contributors, and the
preserved NX-AI/KronLSTM sources and MIT provenance. This clarification is
substantially AI-assisted by Codex for @thakrarsagar; it is not independent
human verification.
