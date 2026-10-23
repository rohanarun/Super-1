# Small minKron iteration experiment

Run ID: `minkron-fast-20261021-v1`. AI-assisted implementation and execution.

Six models of roughly 1.5–2.0M parameters compare matrix head size, a half-width MLP, and a causal convolution against the 8x8 reference and a RoPE Transformer. Read `PROTOCOL.json` together with both numbered addenda. `PROFILE.json` freezes the throughput-derived token budget before corpus training. Numerical qualification failures are retained.

The full report is `../../reports/minkron-fast-20261021-v1.md`. `SUMMARY.json` provides compact results, `METRICS.json` the full per-seed records, `SAMPLES.json` all fixed-prompt completions, and `DEPLOYMENT.json` measured latency/state bytes. None of these files are promises of general language-model competence.

## Precision and fairness

Weights, projections, MLP and logits are FP32. The recurrent normalization, decay, state update and output normalization use FP64 after the fully FP32 implementation failed the predeclared gradient/optimizer fidelity checks. AdamW epsilon is 1e-4 for every arm. State memory and speed include that cost. No tolerance was loosened.

All arms see identical target windows within each seed, with EOT resets and no carried state between training windows. The vocabulary is a lossless 4096-entry prefix of GPT-2 BPE; do not compare these perplexities to the prior GPT-2-vocabulary run. The Transformer has a lean MLP and is a practical comparator, not a parameter- or FLOP-matched control.

## Reproduce

Use Python3.12, PyTorch2.9.1 CPU and tiktoken0.14.0; exact executed versions are in `ENVIRONMENT.json`. No GPU is needed. A new independent experiment should use a new directory and protocol; completed evidence must not be overwritten.

1. Reconstruct the original GPT-2 arrays and their document manifests using the pinned FineWeb revision and the older `../minkron-publication-20261021-v1/nanogpt/experiments/cpu_scale/prepare_fineweb.py`. That script expects a fresh `data/fineweb_cpu/` containing `source.parquet` and the archived `source_metadata.json`; run it from a separate working copy of the archived `nanogpt` project, with that project on `PYTHONPATH`. It requires pyarrow. Do not run it in the frozen archival directory or overwrite its manifest. See `RECONSTRUCTION_REQUIRED.md` for exact input hashes.
2. From this directory, run `python prepare_data.py --source /path/to/reconstructed/fineweb_cpu`. The script checks original array hashes and lossless round trips. Confirm the resulting hashes against the published manifest.
3. To reconstruct a specific checkpoint, use the saved actual learning-rate/step trajectory:

   ```bash
   python replay.py --arm k16 --seed 25101 --stage screen --output /path/to/new/replayed.pt
   ```

   `replay.py` reports whether the model tensor hash matches. Only the selected variant, k8 and Transformer have extension checkpoints. Same source/data/seed alone does not guarantee identical CPU floating-point results on every system. Checkpoint container bytes may differ even when model tensors match.
4. `qualify.py` documents the executable numerical tests. `train.py` is the original bounded launcher and refuses to overwrite a terminal run. Its timed continuations are hardware-dependent; use `replay.py` plus `ALL_STEPS.csv.gz` for recorded-exposure reconstruction.
5. `inspect_models.py` generates fixed samples and deployment measurements from retained checkpoints. `analyze.py` builds the numeric report scaffold and figures from receipts without training or evaluation; the published report also includes a reviewed interpretation.

Corpus arrays, checkpoint bodies and bulky document maps are intentionally omitted from Git. Their identities and reconstruction requirements are explicit. This is code and compact results, not a model-weight distribution.

## Attribution

The minKron recurrence derives from NX-AI/KronLSTM through this project's preserved compact/common-decoder adaptation. The original provenance and MIT notices remain in `../minkron-publication-20261021-v1/`; this experiment does not claim to invent that recurrence. The added changes are the small model configuration, local convolution ablation, numerical execution policy, training harness and measurements.

Reconstruction was checked end to end for the k16 seed25101 screen checkpoint: all model tensors matched exactly (REPLAY_VERIFICATION.json). The additional verification took39.51 seconds; it did not change experiment outcomes or selection.
