# Approximately 124M minKron / NanoGPT CPU exploration

This is the user's separately authorized 25-minute CPU scale experiment. It
does not change `comparison_contract.json`, substitute for the original GPU
comparison, or establish a win against a Transformer or minGRU.

The decoder has 12 layers, width 768, GPT-2 tokenization and a tied 50,257-token
embedding/output matrix, pre-RMSNorm, a ReLU-squared MLP, and minKron mixers.
There are no attention blocks, n-gram tables, GSA hierarchy, or pretrained
weights. Thus "NanoGPT" describes the decoder/training setting rather than an
unmodified GPT-2 architecture.

## Resource spread

All layouts use 8 by 8 minKron matrices. Adjusting MLP width holds the total
weight budget near 124M while varying recurrent heads and readout width.
These are systems probes, not independently trained quality comparisons.

| Heads per layer | MLP width | Parameters | Readout channels per layer | FP32 state, all 12 layers |
|---:|---:|---:|---:|---:|
| 12 | 4,352 | 123,562,992 | 96 | 36 KiB |
| 24 | 4,096 | 123,500,928 | 192 | 72 KiB |
| 48 (selected) | 3,584 | 123,376,800 | 384 | 144 KiB |
| 72 | 3,072 | 123,252,672 | 576 | 216 KiB |

The selected configuration balances a wider recurrent readout with CPU time.
Short probes favored four threads, FP32, foreach AdamW, chunks of 32, and two
256-token sequences per update. The CPU supports AVX-512, and PyTorch dispatches
its optimized CPU operations automatically. BF16 and eight-thread probes were
slower in this environment; no general hardware claim follows from that result.

`kron_chunk_factored` factors pairwise decay into GEMMs and avoids T by T by K
temporary tensors where numerically safe. It falls back to the original stable
core when the centered cumulative decay exceeds 40 in magnitude. This is an
algebraic implementation change, not a different recurrence. Source, full
correctness checks, every benchmark and all learning-rate outcomes are retained.
Short random-token throughput is not the final trained-model throughput.

## Data and training protocol

The archive includes the exact tokenized train/development/confirmation arrays,
their document manifests, source revision and hashes. The source is the first
FineWeb `sample-10BT` parquet shard at revision
`9bb295ddab0e05d785b879661af7260fed5140fc`. The 2.15 GB source parquet is omitted.
`data/fineweb_cpu/manifest.json` identifies it unambiguously if rebuilding.

Splits use the SHA256 of exact document text: modulo 100 values 0 and 1 go to
development and confirmation; remaining values go to training. IDs and exact
text hashes are deduplicated globally. This is a custom disjoint split, not the
official NanoGPT speedrun evaluation. Near-duplicate leakage is not ruled out.

The training pool has five million available targets. One fixed permutation of
256-token windows is consumed without repetition for the time budget. State
resets at window starts and EOT inputs. Full BPTT covers each window. The run is
from scratch, one seed, with 512 targets per optimizer update, FP32 parameters,
recurrence and AdamW moments. Gradient norm is clipped to 1.0. A development-only
16-update pilot sweep selected peak learning rate 0.001 from 0.0003/0.001/0.003.
Main training restarts from initialization and warms up over 5% of wall time,
then uses cosine decay to 10% of peak.

The 1,500-second main timer includes data loading, training, periodic monitoring,
checkpoint writes and logging. Setup, initial/final evaluation, final save,
pilots, systems probes and generation are additional elapsed work. Checkpoint
selection is the final timed checkpoint, not the best development checkpoint.

## Commands

Use Python 3.12 and install `requirements-cpu-scale.txt` with an appropriate CPU
PyTorch wheel. This run used PyTorch 2.9.1+cpu and NumPy 2.x. Run commands from
the project root. Choose a new output directory for any new training experiment.

```bash
python tests/check_factored_cpu.py
python -m experiments.cpu_scale.run --layout h48 --core factored \
  --precision fp32 --threads 4 --batch 2 --length 256 --accumulate 1 \
  --optimizer foreach --lr 0.001 --seed 24101 --seconds 1500 \
  --eval-tokens 8192 --out research-scratch/cpu-124m/new_run
python -m experiments.cpu_assess --run research-scratch/cpu-124m/new_run
```

For an interrupted run, use the identical command and output directory with
`--resume`; source hashes, data and configuration must match. The checkpoint
holds model, optimizer, wall-time scheduler, RNG states and shuffled-window
cursor. There is no recurrent state to carry across independent windows.
This new driver has checkpoint serialization; a bitwise interrupted/resumed
25-minute trajectory has not been established because its schedule is based on
wall time, which depends on machine load and checkpoint overhead.

The delivered checkpoint belongs at
`research-scratch/cpu-124m/main/checkpoint.pt`. The assessment output includes
8,192 held-out confirmation targets, a same-target unigram baseline using only
training targets actually seen, a small reset-every-token context diagnostic,
and all ten prespecified generations. The context diagnostic is not a long-term
memory benchmark. Sampling is temperature 0.8, top-k 50, maximum 64 new GPT-2
tokens, seeds 25101 through 25110, with EOT stopping. Samples are retained once,
without choosing nicer alternatives.

See `research-scratch/cpu-124m/main/summary.json`, `assessment.json`,
`training.jsonl`, `curves.json` and `generations.jsonl` for the measured outcome.
The HTML report presents the same measurements and raw continuations.
