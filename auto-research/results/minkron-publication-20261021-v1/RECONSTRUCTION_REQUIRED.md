# Reconstruction required for omitted artifacts

This PR intentionally contains code, compact result receipts and reconstruction
placeholders. It omits model/optimizer checkpoints, raw prediction arrays, corpus
arrays, large histories/logs, generated HTML/figures, redundant patches and
supporting delivery material. No placeholder contains usable model/data bytes.

`IMPORT.json` is the machine-readable placeholder inventory. Each original
archive member has an exact path, byte count and SHA256. `published_path` means
the file is included; `excluded_reason` means reconstruction or restoration is
required. The named source archives and their hashes are retained there too.

| Missing artifact | Reconstruction route |
|---|---|
| Compact final checkpoints and `predictions.npz` | Restore the exact named files from `Compact_Kron_budget_experiment_2026-10-21.zip`, or train fresh in a disposable copy using `compact/README.md`, then run confirmation. A fresh run is new evidence and need not reproduce the original weights/timings. |
| Compact learning histories, logs, PNG and HTML reports | Restore original histories/logs from that archive; `compact/report.py` regenerates presentation artifacts from complete saved results. The Markdown review report already contains the principal measured results. |
| Small integration diagnostic outputs/figure | Run the preserved verification and smoke scripts in a fresh copy, or restore originals from `minKron_NanoGPT_Integration_2026-10-21.zip`. Its small text fixture is included. |
| FineWeb corpus arrays and per-document manifests | Rebuild using the pinned source shard and `nanogpt/experiments/cpu_scale/prepare_fineweb.py`; compare array hashes with the included data manifest. See the detailed README for the existing-manifest guard. |
| 124M weights and AdamW state | Restore the separate checkpoint whose hash/size are in `IMPORT.json`, or run fresh training using `nanogpt/docs/CPU_124M.md`. The checkpoint is not contained in the source/results ZIP. |
| Full CPU traces and rendered figures/reports | Restore original members from `minKron_124M_CPU_25min_Source_and_Results.zip`, then use the included report generators; alternatively generate new artifacts from a new run. |

The final checkpoint's required original location is
`nanogpt/research-scratch/cpu-124m/main/checkpoint.pt`, size 1,480,861,098 bytes,
SHA256 `fde8183f3206b16851cfd4857255695176d4c2cbc5adf8cab89380c26c97c326`.
Exact historical bytes require restoration; rerunning a wall-time-based schedule
cannot promise the same checkpoint hash. Original bundle availability is through
the contributor; this PR does not publish a new public download for them.

`PREPARATION_AUDIT.json` records the completed local raw-prediction/trace check
before the publication was slimmed. The final `verify_import.py` uses only the
standard library and included receipts; it explicitly does not claim to recheck
the omitted raw predictions or execute a model. Nothing needs reconstruction
merely to review the code, read the results or run this compact audit.
