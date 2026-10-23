# minKron experiments and publication record

Start with the [review report](../../reports/minkron-publication-20261021-v1.md).
This directory preserves completed experiments from October 21, 2026. Publication
does not run training, rerun model evaluations, or change the production model.

| Directory/file | Contents |
|---|---|
| [compact/](compact/) | Synthetic minKron/minGRU budget experiment: executable source, 8 LR pilots, 12 main runs, compact result receipts and checks |
| [integration/](integration/) | Original common-decoder implementation and its exact CPU diagnostic source/receipts |
| [nanogpt/](nanogpt/) | Later CPU implementation, ~124M scale run, resource probes, LR selection, all 10 generations and report |
| [IMPORT.json](IMPORT.json) | Source-archive hashes, every included/omitted member, authorization and checkpoint identity |
| [VERIFICATION.json](VERIFICATION.json) | Publication audit of stored evidence; no new model execution |
| [verify_import.py](verify_import.py) | Reproducible file/metric/prediction audit |

The included source files and compact receipts are byte-for-byte preserved.
Bulk artifacts are deliberately omitted; see [reconstruction placeholders](RECONSTRUCTION_REQUIRED.md). Repeated files are
content-deduplicated by Git. Their historical README/status text describes the
original deliveries: statements such as "no source was pushed" and "the archive
includes token arrays" are historical, not a description of this PR. The current
inventory and exclusions are authoritative in `IMPORT.json`.

The small decoder's FineWeb/GPU comparison remains proposed. The separate 124M
CPU run completed; it is not that comparison. The `integration` snapshot retains
the earlier model source so its saved checks are not attributed to the later
factored CPU implementation.

## Verify the published results

From the repository root, with Python 3 (standard library only):

```sh
python3 scripts/verify_evidence.py
python3 auto-research/results/minkron-publication-20261021-v1/verify_import.py
```

The first command checks the repository manifest. The second checks snapshot
seals, compiles the Python source without executing it, reconciles per-seed
count receipts and means, checks per-stage CPU/update caps and checkpoint
associations, and reconciles the scale-run summary and all ten generations. Neither
command imports Torch, downloads data, trains, or executes a model. The full raw
prediction/trace audit was completed locally before slimming and is preserved in
`PREPARATION_AUDIT.json`; it is not repeated by the compact audit. Excluded
checkpoint identities are checked against archive records, not reloaded here.

## Reproduce the experiments

Use disposable copies and each snapshot's own requirements. Original verification
and reporting scripts can overwrite their receipt files. Do not run them in a
sealed snapshot. See [compact/README.md](compact/README.md),
[integration/README.md](integration/README.md), and
[nanogpt/docs/CPU_124M.md](nanogpt/docs/CPU_124M.md) for commands and scope.
The compact task is synthetic and needs no downloaded corpus. Its fresh-run
queue refuses to overwrite existing run directories; remove copied `results/`
and `logs/` only in a disposable reproduction directory before starting.

Large model/optimizer checkpoints, corpus token arrays, per-document corpus
metadata, and redundant full-source patches are excluded from Git. All original
archive member hashes are retained. The compact archive has the 12 final small
checkpoints; the 124M checkpoint is a separate 1,480,861,098-byte artifact with
SHA256 `fde8183f3206b16851cfd4857255695176d4c2cbc5adf8cab89380c26c97c326`.
These external artifacts are not public download dependencies of the audit.
Contact the contributor for the original bundles/checkpoint if replaying the
existing checkpoints. This PR does not claim to publish their bytes or to have
newly verified the large checkpoint.

For a fresh CPU text run, obtain the exact FineWeb shard identified in
`nanogpt/data/fineweb_cpu/source_metadata.json` and `manifest.json`. In a disposable
copy, preserve the original manifest outside `data/fineweb_cpu/`, place the shard
at `data/fineweb_cpu/source.parquet`, and run:

```sh
python -m experiments.cpu_scale.prepare_fineweb
```

The preparation script refuses to replace an existing manifest. Compare the
rebuilt split-array and document-manifest hashes against the preserved manifest.
The preparation-time field changes, so its whole JSON hash will change even when
the arrays match; keep the newly generated manifest with a new run. Use the
specified Python, NumPy and tokenizer versions for byte-level reproduction.
Training and assessment commands are in `docs/CPU_124M.md`. A wall-time schedule
is machine-dependent and is not promised to replay bit-for-bit. The 124M
assessment needs a trained checkpoint; it is not part of the stored-evidence audit.

## Attribution and scope

The compact recurrence is derived from [NX-AI/KronLSTM](https://github.com/NX-AI/KronLSTM),
with source provenance and MIT notices in each relevant snapshot. The hierarchy
comes from Super-1, with its MIT notice retained. The common-decoder work cites
its NanoGPT reference and pinned data pipeline in the original contract and
vendor source. "minKron" here names this compact adaptation, not a newly claimed
invention of the underlying recurrence.

This contribution was substantially assisted by ChatGPT/Codex under Joshua
Rowe's direction. The user's explicit request to publish upstream supersedes
historical fork-only handoff wording for these results. No automatic merge is
requested.
