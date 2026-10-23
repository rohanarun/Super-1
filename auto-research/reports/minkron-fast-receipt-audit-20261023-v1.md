# Experiment report: minkron-fast-receipt-audit-20261023-v1

Status: completed stored-evidence verification. No new training or model evaluation; no promotion of an accepted result.

Approval reference: the donor's 2026-10-23 contribution request, up to 480 minutes, posting and one focused PR. [Claim #17](https://github.com/rohanarun/Super-1/issues/17).

Base: `847bb75042e01b2c879272f97e72746f285c8872`. Session seed: `dcbb91d07e51cf13ee6fda1fefc14ca2`. Super-1 has no seeded picker or track catalogue; its brief and adapter request evidence-driven selection. Current issues and every PR state were checked; no open research handoff or overlapping claim/PR was found. The parent is @JRowe47's merged [PR #15](https://github.com/rohanarun/Super-1/pull/15).

Protocol SHA-256: `d081c4ad416845c5f96eaa3e2b35097196e8f398d5e165c2c6efc314e2721e48`, frozen in `dfc950d91d6de33fc4b0fa429b6a55efe7494a97` before cross-receipt analysis. [Evidence directory](../results/minkron-fast-receipt-audit-20261023-v1/), [audit output](../results/minkron-fast-receipt-audit-20261023-v1/AUDIT.json), [input identities](../results/minkron-fast-receipt-audit-20261023-v1/INPUTS.json).

## Question and controls

Do the stored receipts for `minkron-fast-20261021-v1` support its reported selection, continuation comparison, target exposures, numerical qualification status and reconstruction bindings? This addresses E00/E01 evidence and replay readiness for the newest small-model package. The competing explanation is a summary/receipt mismatch or an unbound replay input. No architecture hypothesis is newly tested.

The cheapest control is standard-library recomputation from already published scalars. Both the repository verifier and the parent's archived `verify_run.py` pass as positive controls. The new audit adds cross-receipt arithmetic, complete identities and step/LR schedules, rather than changing the original verifier. Its semantic failure controls deliberately alter 19 independent obligations after input-hash validation. Two further controls exercise hash and inventory rejection. A false pass on any such corruption would falsify the new verifier.

The numerical recomputation tolerance is the preregistered `rel_tol=1e-12, abs_tol=1e-12`; printed values use the original report's rounding. The parent useful-effect rule stays at a mean development improvement of at least 0.02 nat/token with every paired seed improving. No threshold was changed.

## What actually ran

Only Python standard-library receipt analysis and its tests ran on the existing sandboxed session CPU. No dependency was installed. The original model, training, qualification, data-preparation, analysis/plotting and replay modules were read or hashed, never imported or executed. The baseline repository verifier also hashes its original 21 selected checkpoint files without deserializing them.

`INPUTS.json` binds 86 existing files, totaling 1,307,809 bytes: 68 JSON files, 11 Python sources, three Markdown files, two failed-qualification logs, a compressed scalar trajectory and requirements metadata. The parent registry entry is bound by canonical JSON, allowing new registrations without changing that historical entry. Source/protocol/addendum/profile identities, preregistered data-manifest identity, included tokenizer metadata and the embedded earlier FineWeb manifest agree.

The audit reconciles all 18 screen runs and three continuations against 42 individual run/confirmation receipts, 21 checkpoint metadata records, 9,089 saved optimizer-step records, 2,688 development-window means and 5,376 first-confirmation-window means. It checks development-only selection, resumed step/token cursors, every recorded learning rate against the original schedule, the one-step overshoot of each 300-second continuation, per-run throughput, lifetime/additional time, final training loss, and terminal counts.

The first development invocation failed because the new table parser included the separate deployment table. That implementation error was corrected by restricting parsing to the Results section; a regression test covers unrelated later tables. `DEVELOPMENT_CHECKS.json` preserves the failed invocation. Parent evidence, the frozen protocol and tolerances were unchanged. No scope pivot was needed.

## Results

**Outcome: verification. No scalar, identity, exposure, recorded schedule or published Results-table inconsistency was found within the declared scope.** `AUDIT.json` contains every recomputed per-seed value, paired gap, sample SD and gate. The following are recomputed saved measurements, not new evaluations:

| Arm | CE seed 25101 | CE seed 25102 | CE seed 25103 | Mean paired CE delta vs 8×8 | Development / confirmation gate |
|---|---:|---:|---:|---:|---|
| 8×8 | 6.3956 | 6.4504 | 6.4641 | 0.0000 | reference |
| 16×16 | 6.3681 | 6.4413 | 6.4387 | -0.0207 | met / consistent |
| 32×32 | 6.3909 | 6.4741 | 6.3704 | -0.0249 | not met / not consistent |
| 16×16, lean | 6.3688 | 6.3955 | 6.4248 | -0.0404 | met / consistent |
| 16×16, lean + conv4 | 6.3549 | 6.4010 | 6.4641 | -0.0301 | met / consistent |
| Transformer, lean | 6.7945 | 6.7141 | 6.6699 | +0.2894 | not met / not consistent |

Units are nat/token under the same saved 4,096-entry tokenizer and 128-token evaluation windows. State was reset by the parent at window starts and document boundaries. The parent evaluated 16,384 development and 32,768 confirmation targets per endpoint. All 21 saved confirmation duplicate differences reconcile and satisfy `<=1e-7`, bound to the recorded selected checkpoint container identities. Omitted checkpoint bodies were not fetched or deserialized.

Lean 16×16 remains the development-selected recurrent configuration: saved development delta `-0.04434033755205746`, confirmation delta `-0.04035514400796577` nat/token, all three paired signs negative, 1,528,002 parameters. The 32×32 confirmation mean is favorable but one paired seed regresses; the audit preserves its failed gate. The convolution does not beat lean 16×16 on average.

Each screen run completed 186 updates / 190,464 target exposures. The seed-25101 continuations added 1,190, 1,433 and 3,118 updates for 8×8, lean 16×16 and Transformer respectively. Total lifetime targets are 1,409,024 / 1,657,856 / 3,383,296; saved final confirmation CE is 5.2788 / 5.1443 / 4.8594. The Transformer/lean continuation training-throughput ratio recomputes to `2.176252046305677`. These are equal added-time continuations with unequal lifetime time and exposure. They do not establish a longer equal-token winner.

Total saved main-run training: 9,307,136 target exposures over 1,488.8454165759622 seconds. Setup, evaluation, serialization, the old 39.51-second reconstruction check, and this audit are outside that number. No fresh timing comparison was made.

All 24 saved qualification identities are present: six arms × two seeds × two parameter dtypes. Each of the 144 recorded nonnegative error maxima is below the corresponding frozen absolute tolerance; the largest FP32-parameter maximum is `1.0728836059570312e-6` versus `2e-5`, and the FP64-parameter maximum is `3.985700658404312e-14` versus `1e-8`. This checks saved maxima, not raw tensor parity. Both parameter dtypes use FP64 recurrence, as declared in addendum 02. The saved replay receipt's expected/actual model-tensor digests match the k16/25101 checkpoint metadata; this audit did not retrain it.

A nonblocking documentation discrepancy remains in the frozen `model.py` header: it describes recurrent state/arithmetic as FP32, while `Config.recurrence_fp64=True`, addendum 02, selected configuration and the parent report explicitly declare the FP64 recurrence. The archived header was not edited. Read dtype labels as parameter dtype; they do not demonstrate a fully FP32 recurrent backend.

Three seeds and six screened arms support descriptive within-project ablation evidence only. Development and confirmation regions were reused from earlier work; duplicate evaluation is numerical repeatability, not an additional seed or generalization test. No new significance, reasoning, long-memory, SOTA or novelty claim follows.

## Deployment measurements

Unmeasured by this audit: weights resident at inference, runtime metadata, persistent tensors, activations/workspace, allocator/device/process peaks, prefill/decode p50/p95, concurrency, and fresh latency. The parent `DEPLOYMENT.json` is hash-bound but its timing/state measurements were not replayed or independently recomputed. The historical training RSS is not isolated inference peak memory. No total-memory advantage is inferred.

## Decision

The preregistered stored-evidence consistency criterion is met within scope. This strengthens the checkability of the parent report; it changes neither its exploratory status nor any accepted-result record. All reported gains and failed gates remain the parent's evidence, with its original limitations.

The dominant scientific questions remain unresolved: a fully FP32 faithful implementation, fresh document-disjoint evaluation, stronger equal-token/more-seed comparisons, and measured deployment tradeoffs. None is answered by matching saved scalars.

**Handoff:** begin a new, separately authorized E01 run from the lean 16×16 configuration and preserve the FP64 reference plus both failed numerical receipts. First implement an explicitly FP32 recurrent path and test the frozen output/state/gradient/complete-optimizer-step/streaming/reset/causality tolerances over the original arm/seed/dtype coverage, with nonzero convolution taps. Stop on a failed gate without relaxing tolerance; retain the failure. Only after a pass should an approved canary measure performance and price a language comparison. No model execution or resource budget for that follow-up is granted by this audit. Its cost must be measured by that canary, not inferred from these saved timings.

## Integrity and cleanup

`TERMINAL.json` and `VERIFICATION.json` record completion and checks. The new protocol, inputs, audit code/output, tests, development failure and report are retained. The parent directory, historical reports, source snapshots, registry entries and top-level accepted-result status are preserved; only the new audit entry and manifest/credit bookkeeping are added.

No corpus/checkpoint download, checkpoint deserialization, training, model evaluation, official-test access, paid resource, cloud provisioning, scheduling, merge or other person's PR closure occurred. No unrelated workload or artifact was changed. No cloud resources were created, so cloud cleanup is not applicable.

This is author-checked work, substantially AI-assisted by Codex for @thakrarsagar. Original minKron experiments and publication remain credited to @JRowe47's PR #15, Rohan Arun and the preserved project contributors. The NX-AI/KronLSTM recurrence attribution and MIT notices remain in the original publication package; this audit claims no invention of the model or original experiments.
