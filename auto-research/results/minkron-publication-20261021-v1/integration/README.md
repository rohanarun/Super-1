# minKron NanoGPT integration — 2026-10-21

The common decoder is implemented and passes CPU reference/integration checks.
The 15,146,800-parameter minKron model completed a GPT-2-tokenized, real-text
overfit diagnostic: 100 updates on 128 fixed next-token positions reduced NLL
from 10.86636 to 0.00036185, with final accuracy 100%. This is a memorization
sanity check, not a FineWeb quality result. No GPU was available.

The three LR pilots, 20M-token paired-seed confirmation runs, custom GPU kernel,
BF16 qualification, and GPU serving measurements have **not** run. No language
quality, reliability, cost, or speed winner is established by this work.

## Implemented

- Four decoder blocks, width 256, pre-RMSNorm, ReLU-squared MLP, tied GPT-2
  embedding/head, and no dropout or n-gram tables.
- minKron: four 8×8 heads per layer. Persistent FP32 state is 4,096 bytes per
  sequence across all four layers, measured as both logical and backing storage.
  Each layer reads 32 channels before projection to 256; this bottleneck is
  preserved, not hidden by expanding the state.
- Permanent projection packing with exact source initialization and checked
  parameter gradients/AdamW semantics.
- Native PyTorch bounded chunks of 32, plus explicit sequential oracle.
  Pairwise temporary tensors are bounded by chunk length. Training still
  retains BPTT activations across the window; small state is not small peak RAM.
- minGRU-linear and RoPE/SDPA Transformer comparators share copied backbone
  initialization and the same document/window policy. The minGRU scan is a
  PyTorch O(T log T) reference, not an optimized GPU scan.
- Whole-document split assignment, text deduplication, token/document hashes,
  exact final token budgets, development-only selection hooks, and complete
  model/optimizer/scheduler/RNG/cursor checkpoints.

## Reproduce CPU verification

Use a clean environment with Python 3.12 and the packages in `requirements.txt`.
Install an appropriate PyTorch wheel separately for your CPU or GPU. The tested
CPU build was 2.9.1+cpu; GPU runs must record and qualify their actual build.

```bash
python tests/verify.py
python tests/check_data_and_runner.py
python -m experiments.nanogpt_memory.smoke --threads 2
```

`smoke` refuses to overwrite existing results. Use a clean output copy without
`research-scratch/nanogpt-memory/smoke/` if deliberately repeating the diagnostic.
Prior October 21 synthetic training is preserved as source and prior evidence;
it was not rerun. `vendor/compact_kron_20261021/` contains the recovered executable
reference and its licenses. Its CPU shims are not the authors' fused kernels.

## Evidence and next run

- `research-scratch/nanogpt-memory/verification_cpu.json`: reference, gradients,
  resets, causality, splitting, packed optimizer equivalence, full 512-token shape.
- `research-scratch/nanogpt-memory/smoke/`: frozen diagnostic protocol, exact
  token IDs, learning curve, summary and resume result.
- `research-scratch/nanogpt-memory/data_and_runner_checks.json`: document
  isolation, exact token accounting, and interruption/resume for all three arms.
- `research-scratch/nanogpt-memory/model_accounting.json`: weights versus state.
- `docs/GPU_LAUNCH.md`: remaining GPU/data qualification and pilot commands.
- `comparison_contract.json`: original handoff contract, unchanged.

No paid compute was provisioned. No source was pushed to GitHub in this session.
The archive excludes runtimes, full datasets, duplicate old checkpoints and
the 181.8 MB temporary diagnostic checkpoint. That checkpoint was serialized,
restored, checked, then discarded because this short diagnostic is reproducible
and has no claim to be a research checkpoint/control.
