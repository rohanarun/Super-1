# All located Super-1 experiments: archive and restore index

This archive covers the Super-1/function-weight/hierarchy research found across **270 source directories** on the SSD and in the iCloud backup. It maps **30,795 source paths to 16,815 unique content objects** (33,982,565,377 uncompressed bytes). Source-directory and path counts include duplicate snapshots; they are not counts of independent experiments.

The owner selected **source, results, logs, graphs and final checkpoints**. The archive includes final/best/selected checkpoints, final model files named by terminal records, and the latest located checkpoint in each saved lineage. There are 754 unique selected checkpoint objects: most are small and packed with the evidence; 162 larger objects are individual release assets. Intermediate checkpoints and raw tensor dumps were not exhaustively uploaded.

[SUMMARY.json](SUMMARY.json) is the publication status. `UPLOAD_IN_PROGRESS` means some checkpoint objects remain pending; `ALL_SELECTED_OBJECTS_LOCATED_AND_UPLOADED` means every selected object has a recorded location. An in-progress index must not be presented as a completed archive.

- [Browse the experiment/source catalogue](CATALOG.md)
- [Release assets](https://github.com/rohanarun/Super-1/releases/tag/all-experiments-20261018)
- [Compressed file/provenance index](FILES.json.gz)
- [Upload receipts with asset sizes and SHA-256](UPLOAD_RECEIPTS.json)
- [Source roots searched](SOURCE_ROOTS.json)
- [Excluded-file index](EXCLUSIONS.json.gz)

## Research covered

The catalogue includes the original hierarchy and function-weight reproduction packages; Fourier weight surfaces; paired-spectral and W560 perturbation optimization; CE-versus-distillation comparisons; direction, tensor-block and scalar refinement; projection/hierarchy/Transformer controls; spinning and phase-multiplexed weights; WikiText-2 and enwik8 studies; disparity/coordinate constraints; geometry and pair-energy controls; frozen transfer and optimizer-fidelity canaries; and the recent Super-1 sampling, adaptive-depth, KV-memory, compression and minGRU experiments.

Failed, cancelled, incomplete and unpromoted runs remain as recorded. Names containing “SOTA,” “winner,” or “success” are historical filenames, not conclusions of this upload. The archive is a preservation operation: it does not rerun those studies or independently confirm their scientific claims. Read each run's protocol, terminal receipt and limitations before interpreting a result.

## Storage and provenance

Identical bytes are stored once. `FILES.json.gz` maps each original source path to a safe restore path, its byte count/SHA-256 and one of:

1. An unchanged file already in the pinned Super-1 Git commit.
2. An already verified Super-1 release asset.
3. An `objects/<sha256>` member of a source/evidence archive.
4. An individual checkpoint release asset.

Source/evidence archive members were reread and hashed before upload. Each new release asset is checked against GitHub's recorded SHA-256 and size. This verifies transferred bytes; it does not execute a checkpoint or validate model quality.

The original iCloud chart HTML files that were dataless during the earlier recovery were subsequently read successfully, alongside the four chart PNG variants. Their successful-read receipts are in [ICLOUD_VISUALIZATION_READS.json](ICLOUD_VISUALIZATION_READS.json). Six old remote checkpoint references resolved to a matching local SHA-256; see [the reconciliation](MISSING_FINAL_CHECKPOINTS.json).

## List or restore one experiment

List paths without downloading anything:

```sh
python3 -B auto-research/results/all-experiments-archive-20261018-v1/restore_experiments.py \
  --prefix 'icloud-documents/trin/experiments/disparity_constraints/'
```

After publication is complete, restore a chosen prefix using a mounted writable SSD for both output and cache:

```sh
python3 -B auto-research/results/all-experiments-archive-20261018-v1/restore_experiments.py \
  --prefix 'icloud-documents/trin/experiments/disparity_constraints/' \
  --restore \
  --output '/Volumes/SP AI 01_16/CodexWorkspaces/research-restored' \
  --cache '/Volumes/SP AI 01_16/CodexWorkspaces/research-download-cache'
```

The tool verifies downloaded assets and restored file hashes, rejects path traversal, and refuses to overwrite different content. It never imports or executes historical source or checkpoints. Cache and restored files both occupy disk space; restoring every source alias can consume substantially more space than the unique 34 GB of content. Choose prefixes to avoid reconstructing duplicate snapshots unnecessarily.

Check the index locally, or verify remote asset metadata with an authenticated GitHub CLI:

```sh
python3 -B auto-research/results/all-experiments-archive-20261018-v1/verify_archive.py
python3 -B auto-research/results/all-experiments-archive-20261018-v1/verify_archive.py --remote
```

## Boundaries

This is all selected, readable material in the recorded search roots, not a claim to have found every experiment ever run or deleted remote artifact. Installed environments, caches, browser profiles, Git internals, credentials, raw datasets, teacher payloads and intermediate tensor dumps are outside the selected scope. Existing [input](https://github.com/rohanarun/Super-1/releases/tag/replication-inputs-20261015), [runtime](https://github.com/rohanarun/Super-1/releases/tag/recovered-runtime-20261015) and [120M checkpoint](https://github.com/rohanarun/Super-1/releases/tag/recovered-120m-checkpoints-20261015) releases remain available separately. Reproduction archives may retain small bundled fixtures or already packaged ancillary files.

The excluded index records omissions, and dependency-tree exclusions are summarized by count rather than reproducing hundreds of thousands of installed-package paths. Raw data and missing external services may still be required to run historical code. Historical paths, configurations and launch permissions remain historical; this upload does not authorize a new experiment or change any active model.
