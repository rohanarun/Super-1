# minKron fast stored-receipt audit

Run: `minkron-fast-receipt-audit-20261023-v1`. Status: completed stored-evidence verification; no new model execution.

The [report](../../reports/minkron-fast-receipt-audit-20261023-v1.md) explains the result and limits. `PROTOCOL.json` was frozen in commit `dfc950d91d6de33fc4b0fa429b6a55efe7494a97` before cross-receipt analysis. `INPUTS.json` binds 86 existing files at base `847bb75042e01b2c879272f97e72746f285c8872`, plus the canonical parent registry entry. It includes all directly consumed receipts and the preserved numerical failures.

From the repository root, using Python 3.9 or later and its standard library:

```sh
python3 -B auto-research/results/minkron-fast-receipt-audit-20261023-v1/audit.py
python3 -B -m unittest discover -s auto-research/results/minkron-fast-receipt-audit-20261023-v1 -p test_audit.py -v
python3 -B scripts/verify_evidence.py
```

`audit.py` prints the deterministic contents of `AUDIT.json` to stdout and writes no files. It imports no archived module, PyTorch, tokenizer package or dataset loader. It checks input hashes first, then identities, window-level arithmetic, selection, checkpoints' recorded identities, exposure and walltime accounting, saved step/LR schedules, qualification maxima, metadata bindings and the three published Results tables. It leaves the archived `verify_run.py` unchanged. Run that original check with:

```sh
python3 -B auto-research/results/minkron-fast-20261021-v1/verify_run.py
```

The test suite includes 19 semantic corruptions after the hash boundary, two input-inventory/hash rejection cases, the unmodified positive control and a report-section regression check. Temporary mutations exist only in memory. Checks remain active under `python3 -O`.

Passing this audit means that saved receipts agree. It does not reload the omitted minKron checkpoints or corpora, recompute logits/gradients, establish unseen evaluation, or measure deployment performance. The 24 qualification rows use FP64 recurrence in both parameter dtypes. The Transformer continuation has one seed and more target exposures; confirmation regions were reused. These limits remain part of the result.

## Open next questions

Can a separately authorized, genuinely FP32 recurrence meet the original numerical qualification before language training? Start in a new run with the lean 16×16 configuration, preserved two failed qualification logs and both addenda. First require the original output/state/gradient/optimizer/streaming/reset/causality tolerances with nonzero convolution taps and the existing arm/seed/dtype coverage. Preserve failures and the FP64 reference; benchmark only a passing implementation. This audit authorizes no such execution.

## Credit

Original experiments and evidence: @JRowe47's [PR #15](https://github.com/rohanarun/Super-1/pull/15), Rohan Arun and the preserved project contributors. The recurrence's NX-AI/KronLSTM provenance and notices remain in the parent publication package. This new receipt audit was substantially AI-assisted by Codex for @thakrarsagar. Its checks are author-checked; no separate human or external review is claimed.
