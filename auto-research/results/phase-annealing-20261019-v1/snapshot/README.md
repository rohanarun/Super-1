# Super-1: annealing before, between and after full short BP runs

Standalone experiment `phase-annealing-20261019-v1`, authorized on October 18,
2026 in America/Denver (October 19 UTC). It tests forward-only simulated
annealing as a separate phase, with no permutation matching or continuous
learning. The model sources are Rohan's merged ordinary GRU, linear minGRU and
ParaGRU adaptations at `492a87a5e263e806b0a2581c57259e81f1a000e6`.

Read `auto-research/results/phase-annealing-20261019-v1/REPORT.md` first.
`TERMINAL.json` is the authoritative completion record. The protocol and source
were sealed before confirmation results. Per-branch JSON contains every loss
trace, annealing proposal/acceptance, timing, sample exposure and state hash.

## Reproduce

Use a fresh extracted copy and preserve the original evidence before running.
The runner refuses to overwrite a started experiment. To replay the same
protocol in a disposable copy, move its `START.json` out of the result directory
first; do not alter the preserved original archive. Source or method changes
require a new run ID and new registration.

```sh
python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 tokenizers==0.22.1 matplotlib==3.10.8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python prepare_data.py
python validate.py
python experiment.py freeze
python experiment.py run
python auto-research/results/phase-annealing-20261019-v1/verify_saved.py
python auto-research/results/phase-annealing-20261019-v1/analyze.py
```

The original runtime was Python 3.12.14, PyTorch 2.8.0+cpu and NumPy 2.3.5 on
one CPU thread. Extra-backprop counts depend on measured wall time; they are
not bit-reproducible across machines. The frozen run has a 30-minute main
budget and records an incomplete status if a slower machine exhausts it.

The supplied tokenizer and `prepare_data.py` reconstruct the data from a pinned
WikiText-2 TRAIN URL. Confirmation consists of 38 documents that were held out
and never evaluated in the prior state-sketch pilot. The prior evaluated monitor
documents and entire prior confirmation partition are excluded. Official
validation/test data are never downloaded. All document identities and byte
hashes are in `data/manifest.json`.

## Interpretation

BP A has 192 updates and BP B has 96, each with a complete warmup/cosine schedule.
These are full short scheduled phases, not training to convergence or an
experiment on a full-size language model. The primary test is the sandwich
with Adam retained, compared with approximately equal-time extra BP. Pre/post,
Adam-reset and zero-temperature greedy-search conditions are also reported.

SA returns the best accepted chain state, including the starting checkpoint.
The final chain state can differ from that returned model. Its loss and hash
are logged, but the final-chain endpoint followed by BP is not a separate arm.
This is one concrete annealing configuration, not a test of every cooling,
proposal, phase-length or return policy.

## Compact artifact

The archive contains source, tokenizer, data provenance, protocol, all 99
outcomes, audit and analysis scripts, report and chart. Installed dependencies,
downloadable corpus text/token arrays, Python bytecode and tensor checkpoints
are omitted. Three sandwich checkpoints are retained in the working session
for fresh-process verification; exact tensor recovery later requires rerunning.

Upstream model files retain their MIT license in `upstream/LICENSE`.
