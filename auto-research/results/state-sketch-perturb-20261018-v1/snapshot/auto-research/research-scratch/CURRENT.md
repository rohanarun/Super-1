# Current research handoff — 2026-10-18

Run `state-sketch-perturb-20261018-v1` is COMPLETE: 54/54 paired branches.
Decision: DO NOT ADVANCE this perturbation configuration; all three architecture
families failed the frozen improvement gate. The sketch cost gate passed.

Included upstream ordinary GRU, linear minGRU and ParaGRU fast-cell adaptations,
parallel/batched execution, slow hierarchy, tokenizer and optimizer recipe from
`rohanarun/Super-1@492a87a5e263e806b0a2581c57259e81f1a000e6`.

Three seeds per architecture, six conditions, ~173–178k parameters on CPU.
The CountSketch treatment did not improve on random pairs or scrambled anchors
on average. Extra replay backprop achieved lower held-out CE than the guided
perturbation in all nine paired seeds. CountSketch and Hadamard were substantially
better similarity fingerprints than the tested DSP-statistics and FFT-power
summaries. This does not establish an optimization benefit from structural
similarity or rule out other perturbation mechanisms.

See `../results/state-sketch-perturb-20261018-v1/REPORT.md` and `ANALYSIS.json`.
The protocol, every raw branch record, fingerprint probes, provenance, and
checks are preserved alongside them. All eight numerical preflight cases passed;
all 54 in-process repeat evaluations matched; one checkpoint per architecture
also reproduced its hash and held-out evaluation in a fresh process.

User authorized these experiments and explicitly added the merged minGRU and
other recurrent variants. No GitHub publication was performed. This is a small
pilot, not the full 120M model, not the continuous-learning branch, and not
Rohan's unpublished perturbation implementation. The experiment consumed no
new examples at perturbation events, but did use forward passes on replayed
training examples to select candidates.

No post-confirmation tuning was done. A new hypothesis, radius, cadence or
phase-preserving fingerprint intervention requires a separately registered run
and fresh confirmation data. Do not silently overwrite this completed result.

Compact packaging excludes installed dependencies, re-downloadable text/token
arrays, and tensor checkpoints. Final-state hashes and deterministic rerun
instructions remain; replay counts depend on wall timing. Checkpoint tensors
are available only in this working session unless separately preserved.
