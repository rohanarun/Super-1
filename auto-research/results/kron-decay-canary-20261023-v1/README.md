# Bounded minKron decay-loss canary

The approved canary found a rounded-prefix decay-loss case. Both interval-product variants passed all 50 declared core cases, while their FP32 model parameter-gradient gates failed. The current factored FP32 control passed the one model row but failed the same 16 core stress cases as the direct/prefix controls. No backend is generally qualified by this result.

The original [approved draft](APPROVED_DRAFT.json), [explicit session approval](APPROVAL.json), [project protocol](PROTOCOL.json), [executable canary protocol](CANARY_PROTOCOL.json), [source inputs](SOURCE_INPUTS.json), exact source, environment/wheel lock and every raw outcome are retained. Protocol and numerical implementation were frozen at `6636bc315022930108de66db2b6906e733429f19`; disk preflight and cumulative guard at `73b51b0`.

## Inspect without executing a model

From the repository root:

```sh
python3 -B auto-research/results/kron-decay-canary-20261023-v1/verify_results.py
python3 -B -m unittest discover -s auto-research/results/kron-decay-canary-20261023-v1 -p test_verify_results.py -v
python3 -O -B -m unittest discover -s auto-research/results/kron-decay-canary-20261023-v1 -p test_verify_results.py -v
python3 -B scripts/verify_evidence.py
```

These stdlib checks reconcile saved coverage, all gate flags, raw diagnostics, resource receipts and tiny analytic references. Eight tests include ten receipt-corruption rejections. They do not repeat tensor execution or reconstruct full gradient arrays. The global verifier checks stored bytes and historical receipts, not new metrics.

`SUMMARY.json` presents the complete control counts and model outcomes. `execution/attempt-01.stdout` is the complete JSON tensor result (250 core / six model rows); `execution/attempt-01.stderr` preserves every progress/result diagnostic plus the optional-NumPy warning. NumPy was not installed or used. No execution failure occurred; scientific gate failures remain in the result.

## Recorded execution and limits

One numerical attempt used Torch `2.9.1+cpu`, Python3.13.5, one Torch and inter-op/BLAS thread. It charged 24.792752 CPU / 23.213876 wall seconds including the execution guard. Harness-only time is reported separately in the raw result. The guard allowed at most the remaining cumulative600 CPU /900 execution wall seconds, with reserve for supervision. Source/input hashes were checked before importing the model. No checkpoint or dataset body was loaded.

The CPU wheel's frozen SHA-256 is `39b3dff6d8fba240ae0d1bede4ca11c2531ae3b47329206512d99e17907ff74b`. `REQUIREMENTS-CPU-CP313.txt`, `INSTALL_REPORT.json` and `INSTALL_PREFLIGHT.json` retain the exact wheel set. Offline installation peaked at sampled972656640 bytes, under the1073741824-byte environment/temp/results cap; conservative expansion preflight was995013531 bytes. Temporary wheels were removed after verification; the canary's sampled peak was778162176 bytes. No GPU/cloud, paid spend, corpus training or official-test access occurred. The canary includes one synthetic AdamW step on each candidate/reference copy in six pairs; this is mechanics, not language training.

The recorded invocation was `python3 -B run_process.py numerical attempt-01 -- <session-venv>/bin/python -B tensor_canary.py`. This is historical source, not permission to repeat it: `APPROVAL.json` applies only to the recorded donor session. A new tensor execution or new variant requires its own exact approval/preregistration and resource accounting. A zero exit from the tensor harness denotes its primary core gate; model gates are separate and include failures.

No performance comparison, language-quality gain, general full-model FP32 qualification or historical failure root cause is established. Full gradient/tensor arrays were not archived; saved maxima/gates and raw tiny values are author execution receipts. Read the [report](../../reports/kron-decay-canary-20261023-v1.md) before interpreting them.

Credit: @JRowe47's original minKron adaptation/experiments in merged PR #15, Rohan Arun and project contributors, and preserved NX-AI/KronLSTM MIT/provenance. New canary tooling is substantially AI-assisted by Codex for @thakrarsagar. Archived sources and accepted results remain unchanged.
