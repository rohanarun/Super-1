# Super-1 state-sketch perturbation pilot

Local experiment authorized on 2026-10-18. Includes Rohan's merged ordinary
GRU, linear minGRU, and ParaGRU implementations, pinned at
`492a87a5e263e806b0a2581c57259e81f1a000e6`.

Start with
[`REPORT.md`](auto-research/results/state-sketch-perturb-20261018-v1/REPORT.md).
The full frozen protocol and per-seed results are beside it. This archive is an
unpublished research artifact, not a patch pushed into the upstream repository.

## Reproduction

Use a fresh extracted copy so the original results remain intact. This CPU pilot
used Python 3.12.14, PyTorch 2.8.0+cpu, NumPy 2.3.5, and tokenizers 0.22.1.
Install the CPU torch build from its official index, then the other dependencies:

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
python auto-research/results/state-sketch-perturb-20261018-v1/analyze.py
```

`prepare_data.py` downloads only a pinned copy of WikiText-2 TRAIN. The existing
upstream BPE tokenizer is included. It recreates document-held-out train,
monitor, and confirmation partitions. The frozen protocol checks dataset and
source hashes before training. The run stops on nonfinite values, failed checks,
or its 30-minute main-run budget. A slower machine may hit that budget and record
an incomplete result; do not silently alter a frozen experiment to bypass it.

The 54-branch run includes three seeds, three recurrent variants, six conditions,
192 parent warmup updates, 96 intervention updates, and 32 follow-up updates.
Results and source hashes are tied to this exact protocol. A source or protocol
change requires a distinct run ID and a new registration, not an overwrite of
this result. Replay-backprop counts depend on wall timing, so that arm is not
bit-reproducible across machines. Other arm state hashes are recorded for a
matching deterministic runtime, without promising cross-platform bit identity.

## Contents

- `upstream/`: byte-exact upstream model sources and their MIT license.
- `UPSTREAM_PROVENANCE.json`: pinned source paths and Git blob hashes.
- `data/tokenizer.json`, `data/manifest.json`: tokenizer and reproducible
  document/data provenance; corpus text and token arrays are omitted.
- `geometry.py`: consistent hidden-coordinate permutations and normalized
  function-changing secant proposals.
- `sketches.py`: activation capture, CountSketch, randomized Hadamard, DSP and
  FFT-power diagnostics, and exact trace comparison.
- `validate.py`: historical/parallel equivalence and permutation/optimizer checks.
- `experiment.py`: sealed experiment runner and all six intervention conditions.
- `auto-research/results/state-sketch-perturb-20261018-v1/`: frozen protocol,
  preflight, per-branch loss traces, proposals, runtime data, analysis, and report.
- `auto-research/research-scratch/CURRENT.md`: final handoff and decision.
- `ARTIFACT_MANIFEST.json`: SHA256/size inventory of the compact deliverable.

Installed packages, downloaded corpus files, and tensor checkpoints are excluded
to keep this artifact small. Every branch includes its final state hash, but
recovering checkpoint tensors requires rerunning. Full checkpoints remain only
in the current working session. Official validation/test data were never read.

The passive metric benchmark is not a seven-arm training comparison: only
CountSketch-64 was used as the guided optimization treatment. See the report for
the inference and timing limits.
