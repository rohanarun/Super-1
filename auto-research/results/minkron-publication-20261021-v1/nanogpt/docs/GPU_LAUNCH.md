# GPU continuation and remaining gates

## Current status

CPU reference/integration and a real-text memorization diagnostic passed.
FineWeb has not been downloaded or qualified. CUDA, fused kernels and BF16
have not been tested. The original proposed contract remains unfrozen.

The launch harness uses native PyTorch FP32. Its single-GPU gradient-accumulated
reference can establish quality and correctness; it is not yet a competitive
NanoGPT speedrun. A GPU runner and data access are required for the next stage.

## 1. Build the shared data manifest

Export FineWeb `HuggingFaceFW/fineweb`, configuration `sample-10BT`, from one
explicit dataset commit into JSONL, retaining `id`, `text`, and preferably
`dump` and `file_path`. Record exact source shard names/checksums in the export
provenance. This package does not supply a fabricated dataset revision or
pretend that the data manifest has already been resolved.

The pinned speedrun pipeline `data/fineweb.py` is vendored at commit
`4ea6b937337a4889b8cfe3f38a93d120048d8f71`. It prepends GPT-2 EOT to each document
and uses `encode_ordinary`. Its first-shard validation split can divide a
document at the validation/train shard boundary. Our handoff requires document
disjointness, so the new preparation code assigns whole documents by their
content hash before writing split arrays. Exact-text duplicates are removed
globally. The adaptation must be used by all three arms and shared with the
minGRU chat before any comparative claim. It is not the official speedrun split.

```bash
python -m experiments.nanogpt_memory.data \
  --input-jsonl /path/to/pinned_fineweb_export.jsonl \
  --source-revision DATASET_COMMIT_SHA \
  --output data/fineweb_document_disjoint
```

This creates 20M train, 1M development and 1M confirmation scored targets (each
array has one additional token), exact document records and SHA256s. Only the
last retained document in a split may be truncated to meet the exact token
budget; its remainder is never assigned to another split. Scoring predicts
EOT from a document's final text token; the EOT input resets memory/attention
before it predicts the next document's first text token. Independent windows
also reset state, including windows beginning inside a document.

Document/source provenance must be completed and reviewed before freezing. A
real FineWeb export is required; a text fixture is not an acceptable substitute.

## 2. Qualify the GPU and freeze all three arms together

Install the correct CUDA-enabled PyTorch build, `einops`, `numpy`, `tiktoken`
and `requests`. Keep hardware and runtime identical across arms. Then:

```bash
CHECK_DEVICE=cuda python tests/verify.py
python tests/check_data_and_runner.py
python -m experiments.nanogpt_memory.qualify \
  --data data/fineweb_document_disjoint \
  --verification research-scratch/nanogpt-memory/verification_cuda.json \
  --output frozen_gpu_contract.json
```

Qualification refuses a CPU-only session, stale model checks, wrong data size,
changed runtime/GPU, missing dataset commit, checksum failures, and overlapping
document IDs/text hashes. The frozen file explicitly resolves original draft
choices: FP32 qualification baseline, RMS epsilon 1e-5, RoPE base 10000,
chunk 32, shared normal initialization .02 with residual-MLP scaling,
microbatch 1 with accumulation to 16,384 targets, and exact partial final step.
These are implementation choices for a common revision, not experimental wins.

Before main training, complete GPU checks for long streaming sequences, optimizer
trajectory, numerical accuracy, and any optimized backend. Mixed-precision
projections must earn a separate gate against FP32, including zero/tiny vectors,
reset positions, nonzero initial state, partial chunks and long context.

## 3. Pilot and confirmation

Run every arm at LR 0.0003, 0.001 and 0.003, seed 22901, exactly 2M targets per
run. A pilot command is:

```bash
python -m experiments.nanogpt_memory.train \
  --frozen frozen_gpu_contract.json --data data/fineweb_document_disjoint \
  --output gpu-runs/pilot/minkron_8x8_lr0.001 \
  --arm minkron_8x8 --phase pilot --seed 22901 --lr 0.001
```

Repeat the prescribed grid for `mingru_linear` and `transformer` too. Keep failed
runs and their elapsed compute. Choose LR by final development NLL only; do not
inspect confirmation. Freeze a derived contract containing `selection` with
`learning_rates` for all arms and `development_target_nll`. The proposed target
rule is the best Transformer pilot's final development NLL. If this pilot is
invalid or failed, resolve the target before starting confirmation.

The derived frozen file must retain the original manifest/source/device fields
and describe the selection's supporting run hashes. Apply the same file to
all arms. Then run seeds 23101, 23102, 23103 at each selected LR using
`--phase confirmation`. The harness requires 20M scored training targets and
evaluates confirmation only after the final checkpoint. If promoted, run both
23104 and 23105. `--resume` restores the complete checkpoint and refuses a
changed source/config/data/frozen contract.
The append-only attempt history retains caught failures and their spent compute
across resume. A hard process kill cannot flush its final timing record; preserve
runner/job logs and mark unrecorded time as unknown rather than assuming zero.

Timing in the research harness includes data movement, forward/backward,
optimizer, logging, periodic development evaluation and checkpoint I/O. Setup
and final confirmation have separate fields. The CPU smoke's 27.94 seconds is
only the sum of optimizer-step wall times, explicitly excluding evaluation and
checkpoint testing; never compare it to the research timing field.

## 4. Kernel and serving work that remains

First profile the full LM on the actual GPU. Packed projection is implemented;
the bounded triangular solve remains an unfused PyTorch backend. State is never
padded to 16×16. Report physical workspace if a future kernel pads internally.

For small heads, a fused one-token inference kernel can keep each 8×8 matrix
in registers, apply row decay, compute the key/state/value contraction, perform
the gated rank-one write, then read with q. Retain separate erase/write gates,
the 1e-18 norm floor, SiLU, q/k/v normalization, post-read RMS normalization and
sigmoid output gate. Validate the kernel before reporting throughput.

For training, benchmark a bounded-chunk solve against a recomputing recurrence
backward on this hardware. Do not reuse minGRU's elementwise associative scan as
though it were the same operator. No custom kernel or backward is delivered or
claimed tested in this CPU-only continuation.

After correctness and quality qualification, benchmark interleaved batch sizes
1/8/32 and contexts 512/2048/8192. Separate prefill from steady decode, track
weights, recurrent/KV bytes, working allocations and reserved memory, and keep
one common context policy. Beyond 512 tokens, label quality extrapolation.
Serving benchmarks, time-to-quality comparisons and conclusions remain pending.

This 4-layer pilot does not reproduce the pinned eight-H100 speedrun. The GSA
feedback hierarchy, curve-generated weights and Zipfy extension remain separate.
