# Experiment report: minkron-replay-guard-20261023-v1

Status: completed engineering/control-flow verification; no real model execution.

Approval: donor's 2026-10-23 contribution request and subsequent authorization for multiple focused PRs within the original 480-minute session. [Claim #19](https://github.com/rohanarun/Super-1/issues/19). Base `847bb75042e01b2c879272f97e72746f285c8872`; seed `dcbb91d07e51cf13ee6fda1fefc14ca2`.

Protocol SHA-256 `d64b4a6c7dd107edc1dc19864349accf32afbbdf3ea6e61642ff26ff61b1f01f`, frozen at `1c908e63c83e7ace3e0b5b5dae3a962a52cf00d9` before control-flow reproduction or implementation tests. [Evidence and code](../results/minkron-replay-guard-20261023-v1/).

## Question and controls

Can a separate reconstruction entry point report mismatches as failures and protect existing artifacts while preserving the original replay code? This addresses E00 / Stage 0's requirement that reconstruction has meaningful gates and no hidden source substitution. The strongest alternative explanation is that the original command already fails or protects outputs through another path.

The archived source at `minkron-fast-20261021-v1/replay.py` has SHA-256 `f56b076e800f704989273d542b7e9dd6e0c3eaebbc71c101416d9ed7d7ba8dc9`. We extracted only its `main` AST and executed that function with no-op tensor/loss/optimizer stubs and synthetic metadata/files. No import or execution of a real model was necessary to test command control flow. The positive control is a matching synthetic replay receipt bound to a real saved target's expected digest; failure controls alter each checked boundary.

## What actually ran

Existing session CPU and Python standard library only. No PyTorch installation is present in this interpreter. No package was installed, real model imported, training optimization performed, corpus downloaded, checkpoint deserialized or score evaluated.

The 16-test suite passes under normal and optimized Python. It covers all 21 saved plans, all 15 unsupported extension combinations, four broken schedule cases, 13 malformed/mismatched child-result cases, missing/changed source and arrays, existing files/dangling symlinks, the execution-boundary output recheck, output collisions after computation, the frozen-parent boundary, and metadata-only CLI behavior from an unrelated working directory. The exact commands and timings are in `TEST_RESULTS.json`. Tests clean only their synthetic session-local scratch directories.

No original source was patched. The checked runner uses the pinned parent script in a child process only when explicitly invoked without `--plan`. Before that boundary it checks all 14 bound source/metadata inputs, the requested checkpoint/terminal identity, complete ordered replay steps and required local array hashes. A private staging output prevents the old runner from seeing a pre-existing final artifact. Exact receipt/expected-digest checks precede atomic, non-replacing publication.

## Results

Three original behaviors were reproduced with stubs:

1. `main` returns normally and prints `matches_original_model_tensors: false` after a digest mismatch. The module's CLI directly calls `main` and has no later failure gate, so this is a successful command status despite an unsuccessful exact replay.
2. Normal compilation rejects an existing output with `AssertionError`; compiling the same archived AST with optimization removes that guard and overwrites a synthetic existing file.
3. An unsupported continuation target constructs a fake model and performs its fake step before the late checkpoint lookup raises `StopIteration`.

These are control-flow counterexamples, not new model runs or evidence of a numerical failure in the published checkpoint.

The new entry point accepts the matching synthetic control and rejects inconsistent receipts, wrong expected/actual digests and false flags even when a child returns zero. It rejects missing, empty or symlinked candidate artifacts; child failures keep stdout/stderr and any candidate in the named diagnostic directory. It protects an existing output, a dangling symlink, an output created after planning, and a destination created during the fake child. The concurrent writer's bytes remain unchanged and the staged candidate is retained. No final file is published on a replay mismatch.

All 18 screen plans contain their original 186 updates. The three continuation plans contain 1,376 / 1,619 / 3,304 lifetime updates for k8 / lean 16×16 / Transformer respectively. `TARGET_PLANS.json` preserves every endpoint and expected model-tensor digest. `--plan` returns an explicitly metadata-only status; it cannot be mistaken for a completed exact replay.

## Deployment measurements

No model memory, inference latency, throughput, quality or hardware speed was measured. Unit-test times are verification overhead. A real invocation still requires the parent's numerical runtime, local arrays and explicit experiment/budget approval; compatibility with an actual PyTorch reconstruction remains untested here.

## Decision

The preregistered control-flow criterion is met: within the declared tests, there are no false replay successes or overwritten final artifacts. The checked entry point can be reviewed as a reconstruction-tooling contribution without changing historical evidence or accepting any model result.

The remaining integration obligation is one separately approved real checkpoint reconstruction, using the recorded trajectory and matching runtime. Its first step is `--plan`, then local array hash validation; success requires the exact saved model-tensor digest and safe final publication. A mismatch is a retained failure, not a reason to weaken the check. No compute budget or actual replay is authorized by this report.

Publication uses same-filesystem hard links. Filesystems that do not support them fail with the staged candidate retained. The guard checks the model-tensor digest emitted by the pinned runner; it does not independently deserialize the saved container or qualify optimizer-state identity. No stronger claim is made.

## Integrity and cleanup

The parent run/source/report and every prior registry entry are unchanged. The new protocol, input binding, checked entry point, tests, target plans, test and terminal receipts and this report are additive. Root changes are limited to registry, manifest and contributor bookkeeping. The original scientific results and accepted-result status are preserved.

No GPU/cloud resource, additional paid spend, sealed official test, scheduling, merge, other person's PR closure or unrelated cleanup occurred. No failed real checkpoint was created or removed because no real replay ran.

This is author-checked work, substantially AI-assisted by Codex for @thakrarsagar. It is distinct from open PR #18's stored-scalar audit and does not depend on that PR. Original experiment and reconstruction credit remains @JRowe47's merged PR #15, Rohan Arun, the preserved project contributors and the NX-AI/KronLSTM provenance/notices.
