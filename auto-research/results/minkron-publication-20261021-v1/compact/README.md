# Compact Kron under a minGRU resource budget

This experiment tests one 8×8, four 4×4, and two 4×4 Kronecker memories against
the existing width-64 linear-candidate minGRU fast/segment/document hierarchy.
The model here has no curve-generated weights. All comparisons are synthetic
four-register retrieval tasks, not language-model benchmarks.

## Resource contract

Each sequence has 768 bytes of shared FP32 hierarchy state. minGRU and the
first two compact Kron variants add 256 bytes; two 4×4 adds 128 bytes. Kron
retains no convolution cache and no redundant last-output copy. Learned
write/erase, vector forgetting, output gates, SiLU, unit q/k/v normalization,
and gated RMSNorm are retained. All compact models also have fewer weight bytes.

Each arm gets two pilot learning rates with a 100-process-CPU-second ceiling
per rate, one common seed and a separate validation split. Main runs use three
fresh paired seeds. minGRU completes 10000 short, 1000 medium and 3000 long
updates. Each compact arm gets no more CPU time AND no more updates than its
paired minGRU in each stage. Thus slower implementations receive fewer examples
but still reach the long-delay curriculum. Stage streams are deterministic
paired prefixes, not identical full streams when update counts differ.

CPU clocks include generation, forward/backward, clipping and AdamW. Checkpoint
I/O, validation, setup, and engineering are excluded. Every worker uses one
Torch/OpenMP/MKL/OpenBLAS thread. Baseline uses three concurrent workers;
compact training uses six. This is a measured shared-host CPU budget, not an
equal-FLOP experiment or a prediction of GPU performance. All source hashes and
selection rules are frozen in protocol.json before main training.

## Implementation and checks

The native CPU compact core is algebraically identical to the authors'
normalized, decoupled-erase Kron recurrence. Within each hierarchy segment it
solves a triangular system directly, instead of constructing its inverse. It
does not pad to 64 tokens. Independent gate projections share a packed matrix
multiply; parameter values and functions remain separate. These changes allow
dimensions below the upstream fused-kernel minimum without claiming those
Triton kernels were executed.

Verification covers FP32/FP64 outputs, final state and input gradients against
the authors' token recurrence; irregular sequence splits, full parameter
gradients, tokenwise execution, packed/unpacked projections, oracle labels,
shared initialization, live gradients in every component and tiny-batch fit.
Persistent backing storage is counted, not just tensor view sizes.

## Reproduce

Install requirements.txt in a CPU Python environment, then run:

```bash
python verify.py
python benchmark_quick.py
python run_queue.py pilot
python pipeline.py
python intervene.py
python audit.py
python report.py
```

The queue refuses to overwrite existing training run directories. To reproduce
from scratch, use a clean copy without results/. Training writes
recovery checkpoints every 500 updates and retains final model/optimizer states.
Pipeline stages log continuously and record a review every 30 minutes.

Final confirmation uses 1024 unseen cases for each primary task (competing and
overwrite at nominal age 512) and 256 per other task/age through 2048. Only final
checkpoints are evaluated; test data are never used to select learning rates or
checkpoints. Reports show individual seeds and means; three seeds do not support
strong universal claims. Resource timing is serial and interleaved after all
training/confirmation jobs finish. PyTorch profiler reports peak live tensor
allocation separately from persistent state and parameter storage.

The first main seed also receives independent state-zeroing diagnostics after
256 tokens, preserving counters. Main accuracy conclusions use the untouched
confirmation set, not those intervention examples.

Upstream source provenance is in source_manifest.json and protocol.json.
Vendored MIT licenses accompany the sources. Raw predictions, per-stage CPU
budgets and hashes, pilot results, timing samples, and verification are retained.
