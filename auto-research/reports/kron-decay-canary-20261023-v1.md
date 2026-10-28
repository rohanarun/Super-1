# Experiment report: kron-decay-canary-20261023-v1

Status: completed approved synthetic numerical canary. Primary interval-direct core gate passed; interval candidate model-gradient gates failed. No backend promotion.

Approval: the donor explicitly replied “Approve this bounded canary” to the prepared exact protocol. [Approval](../results/kron-decay-canary-20261023-v1/APPROVAL.json), [claim #25](https://github.com/rohanarun/Super-1/issues/25). Base `847bb75042e01b2c879272f97e72746f285c8872`; session seed `dcbb91d07e51cf13ee6fda1fefc14ca2`. E01 numerical fidelity follows the fast report's faithful-recurrence handoff and PR #22's exact analysis.

The [full protocol](../results/kron-decay-canary-20261023-v1/PROTOCOL.json), SHA-256 `1c03fddb8afd93ed9d06b415f0db29a8da48b7af91018bf3ef34e2bbfa8cb8b5`, and numerical source were frozen before installation/execution at `6636bc315022930108de66db2b6906e733429f19`. The [executable protocol](../results/kron-decay-canary-20261023-v1/CANARY_PROTOCOL.json) SHA is `6629c637046d6cad43024447b0638a4fdf62ca16cf09a4ed5fbf91e633e5d7b5`. Resource guard/preflight were frozen at `73b51b0`. Archived parent files were not modified.

## Question, matching and fixed scope

Does subtraction of rounded cumulative decay logs lose an intervening decay, and can interval products preserve this core behavior? Five variants separate prefix/interval construction and unscaled/scaled coordinates; unchanged sequential `kron_recurrent` is the paired reference. All preserve gates, before-token resets, state equations and the1e-18 norm floor. Candidate code derives from the preserved direct core, not a new architecture.

There are32 fixed scalar tiny cases (two dtypes, four decay sequences, two initial states and two reset controls), plus18 seeded core cases with nonzero state, resets, irregular splits, zero/tiny values and independent input-gradient probes. All five variants receive exactly the same50 cases. FP32 uses atol2e-5/rtol2e-4; FP64 uses atol1e-8/rtol1e-6; no tolerance changed.

The single model row is k16/seed25101, FP32 parameters with `recurrence_fp64=false`, random token IDs B2/T19, resets at0 and example-specific7/13, matching the archived construction. It checks logits/state, loss, all parameter gradients, one AdamW step per copy (LR.001, epsilon1e-4), optimizer state, streaming/reset/causality. A separately labeled FP64-core control retains FP32 parameters. These are reused mechanics fixtures, not fresh scientific confirmation or language evaluation.

## Core results

All250 comparisons were finite. Each row below has50 cases; reset and causal checks pass50/50 in every variant.

| Variant | Forward/state pass | Input-gradient pass | Streaming pass | All gates pass |
|---|---:|---:|---:|---:|
| source_direct |34|34|38|34|
| source_factored |34|34|38|34|
| prefix_scaled |34|34|38|34|
| interval_direct |50|50|50|50|
| interval_scaled |50|50|50|50|

All18 random cases pass every gate in every variant. Each prefix variant fails12 FP32 and four FP64 extreme tiny cases; those are the16 forward/gradient failures, not a claim about ordinary model decay distributions.

For q=k=v=1, beta=write=.5, zero initial state and g=[-2^27,-1], the sequential target is `.5+exp(-1)/4`. The huge first log controls prefix representability; zero initial state removes its carried-state contribution. FP32 stores both cumulative logs as `-134217728`; direct, factored and prefix-scaled final state is `.75`, versus reference/interval `.5919698476791382`. FP64 retains the increment at2^27 and all variants produce `.5919698602928606`. At2^56, FP64 prefixes also lose the increment: prefix variants again return `.75`, while reference/interval return `.5919698602928606`. The two-token centered range is zero in the failing cases, so a centered-range guard alone does not detect this precision loss.

The direct source already masks invalid log differences before exp. This is a cumulative-prefix precision counterexample, not a newly alleged masked-exp overflow. Unit value norms also make the scaling control identical in exact coordinates; scaling alone does not fix the observed prefix loss.

## Single model-row results

Only the parameter-gradient gate fails where indicated. Every variant passes the other seven checks, including the optimizer step/state. Elementwise absolute-plus-relative tolerances determine the gate; a maximum absolute error above atol alone need not fail.

| Variant / core policy | Maximum gradient absolute error | Maximum normalized tolerance error | Model gate |
|---|---:|---:|---|
| source_direct / FP32 |3.635883331298828e-5|1.1143568889|fail: embedding.weight|
| source_factored / FP32 |2.872943878173828e-5|.9563628466|pass|
| prefix_scaled / FP32 |5.02467155456543e-5|1.6460524233|fail: embedding.weight|
| interval_direct / FP32 |3.5643577575683594e-5|1.1277802810|fail: embedding.weight|
| interval_scaled / FP32 |5.036592483520508e-5|1.6332032940|fail: embedding.weight|
| source_factored / FP64 core, FP32 parameters |0|0|pass|

Normalized error is `abs(a-b)/(atol+rtol*abs(reference))`, computed in FP64 for diagnostics; the frozen `torch.allclose` gate is unchanged. Diagnostic casts do not alter core execution. The factored FP32 control's ordinary-row pass does not erase its stress failures, and interval stress success does not erase its model gradient failure. Neither establishes general fully FP32 qualification. The ordinary direct gradient failure is a fresh bounded observation, not proof of the historical failure's cause.

## Execution evidence and verification

One completed attempt used Torch2.9.1+cpu, build `5811a8d7da873dd699ff6687092c225caffcf1bb`, Python3.13.5 and one Torch/inter-op/BLAS thread. Harness21.824649 CPU /21.946903 wall seconds; charged process/supervision24.792752 CPU /23.213876 wall seconds. No resource stop. Sampled install peak972656640 bytes and run peak778162176 bytes stayed below1GiB; conservative install preflight was995013531 bytes. Maximum child RSS460364KiB is an operational observation, not an architecture memory benchmark. Installation CPU is recorded separately and excluded from the approved numerical cap. No timing is used as a comparative speed result.

[Raw result](../results/kron-decay-canary-20261023-v1/execution/attempt-01.stdout), SHA-256 `2ff304b2b83cea8b7c63e807410e2de1b0abccc452e7a9088bedf0fe86028093`, and every raw diagnostic/control are retained with environment, exact dependency-wheel lock, installation report and resource receipts. NumPy was absent; its optional initialization warning is preserved. No further dependency was installed to silence it.

The stdlib receipt verifier reconciles250 rows, all50 case identities and160 tiny-reference records, all six model rows/gates, raw stderr duplicates, source hashes and resource accounting. Eight tests pass normal/-O with ten corruption rejections. Tiny sequential reference outputs also agree with an elementary scalar analytic recurrence under the frozen tolerances. This is a saved-receipt and limited analytic check, not an independent tensor replay.

## Decision and unchecked work

The primary bounded core hypothesis is supported. Both interval candidates fail the named model-gradient gate; keep them experimental. A faithful faster recurrence remains an open engineering task. A future preregistered dispatcher could retain the factored path ordinarily and fall back when a nonzero log increment is rounded away; that variant is unimplemented and untested here and needs separate exact approval.

Unchecked: full gradient arrays, all original model arms/seeds/dtypes, broader/extreme domains, an independent numerical replay, throughput/latency/architecture memory, corpus or language quality, and the historical failure root cause. No corpus/checkpoint body, GPU/cloud, paid spend or official test was used. Twelve synthetic one-step optimizer updates occurred across the six candidate/reference pairs; no checkpoint was saved and no corpus training run occurred. Remaining CPU allowance does not authorize a new variant or wider experiment.

Credit remains @JRowe47's merged PR #15, Rohan Arun/project contributors and preserved NX-AI/KronLSTM MIT/provenance. New canary/tooling is substantially AI-assisted by Codex for @thakrarsagar. Existing source/evidence, previous registry content and accepted results are unchanged.
