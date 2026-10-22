# Reconstruction required for omitted artifacts

Large bodies are intentionally excluded from Git at the user's request. This run contains actual completed training, not pre-existing pretrained weights. The code, small tokenizer specification, compact scalar trajectories, outputs and hashes are included.

## FineWeb data

The original shard is `HuggingFaceFW/fineweb`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`, path `sample/10BT/000_00000.parquet` (2,147,292,183 bytes), SHA256 `6b552ea48424648dc86d00df276f93fdfc55e9ad342ce3e4affc23a3a370792b`.

Obtain this pinned source or recover the original GPT-2 arrays from the earlier CPU experiment. Use its archived `prepare_fineweb.py` in a disposable project copy, then this run's `prepare_data.py`. The original and retokenized array identities are both embedded in `data/manifest.json`. Whole-document hash-based splits and exact text deduplication are preserved; no sealed official benchmark is involved.

| Omitted local array | Token count | SHA256 |
|---|---:|---|
| data/train.npy | 6,757,905 | `e9cc19dd52a173cdcab5557acbdedd73710b148a9d3ab791ba620fc4a7b5cc1b` |
| data/development.npy | 88,890 | `29894020b5ae4e82d64698d053bceea480b8e83044bf27f8441d94f38824b55d` |
| data/confirmation.npy | 88,756 | `5d26cce81fe1a8340b6fe2a04f6f9a7fb338a64125726cc397634fdab2ee3b01` |

Per-document maps are also omitted; their hashes are in the same manifest. Re-running data preparation recreates them. The 4096-entry tokenizer is included in `data/tokenizer.json`; it is not an external model dependency.

## Checkpoints

Every final screen and continuation checkpoint is listed in `CHECKPOINTS.json`, with path, container size/SHA256 and a canonical model-tensor SHA256. Bodies remain in the execution workspace under this run's `checkpoints/` and were hash-checked locally; they are not claimed to be durably hosted as downloadable weights.

To rebuild a chosen model, reconstruct the data and run `replay.py` as shown in README.md. `ALL_STEPS.csv.gz` preserves the exact number/order of updates and actual learning rates, including the time-based continuation schedule. Replaying a new five-minute budget would produce a different exposure and schedule. Identical model tensor hashes depend on matching CPU/PyTorch floating-point behavior; serialized checkpoint container bytes may differ. One complete 186-update screen checkpoint (k16, seed25101) was reconstructed from the compressed CSV and matched its canonical model-tensor SHA256 exactly in this runtime; see REPLAY_VERIFICATION.json. Other checkpoints were not retrained.

## Larger logs and maps

Per-step JSON files are omitted because their complete scalar content is represented in `ALL_STEPS.csv.gz`. Periodic progress logs, each final per-seed result, all confirmation scores, numerical failure receipts, source hashes and all fixed prompt samples remain included. No omitted arrays or checkpoints are required by the default saved-evidence verifier; `verify_run.py --local-artifacts` additionally verifies recovered local bodies.
