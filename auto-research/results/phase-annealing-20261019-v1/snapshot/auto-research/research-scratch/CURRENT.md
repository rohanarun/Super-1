# Current research handoff

Completed `phase-annealing-20261019-v1`: 99 outcomes, three paired seeds per
architecture, GRU / linear minGRU / ParaGRU. The standalone SA phases were
placed before all BP, between complete short runs A and B, or after both.

Primary sandwich gate: failed for every architecture. The only numerical
exploratory pass was GRU post-SA; it returned the unchanged baseline in all
three seeds while extra BP worsened held-out CE. This is not a learning gain.
Do not scale this configuration on the basis of this pilot.

Pre-SA modestly improved mean CE against the no-SA baseline, but its equal-time
extra-BP control beat it in all nine paired seeds. Sandwich SA was effectively
neutral against no-SA and lost to extra BP in all nine pairs. Post-SA never
improved held-out CE relative to stopping after BP. Adam reset did not rescue
the sandwich comparison.

## Evidence

Read `../results/phase-annealing-20261019-v1/REPORT.md`, then `ANALYSIS.json`.
`TERMINAL.json` records successful completion in 26.92 minutes and the source
seal. `ARTIFACT_AUDIT.json` checks all 99 records and exact checkpoint reload
scores for one sandwich checkpoint from each architecture. Every branch also
has two identical in-process final evaluations.

Upstream: `rohanarun/Super-1@492a87a5e263e806b0a2581c57259e81f1a000e6`.
Protocol SHA256:
`53664aa1acffebda396104cf95a52c2c4c0bf467b2b08a74dbde83a07c81f0f9`.
The model and training source files remain byte-identical to the sealed run.
No GitHub publication was performed.

## Interpretation limits

This is one 96-proposal Gaussian Metropolis schedule on 173–178k-parameter
models and a fixed 4×512 training calibration batch. A has 192 BP updates and B
has 96; complete short schedules do not imply convergence. Three seeds per
architecture are descriptive evidence, not 99 independent replications.

SA returns the best accepted calibration checkpoint, including its starting
weights. Its cooled-chain endpoint is logged but was not separately followed
by BP. Different proposal geometry, cooling, selection data, return policies,
or converged/full-scale training are not tested. Extra BP reuses the same
calibration batch for approximately the same measured wall time.

The new confirmation set contains 38 documents from the old monitor partition,
excluding every previously evaluated monitor document and the entire old
confirmation partition. Do not reuse these scores to tune another experiment;
a follow-up needs a newly registered protocol and a fresh evaluation design.

The archive omits installed dependencies, corpus text, token arrays and tensor
checkpoints. It includes reconstruction code, tokenizer, source and data hashes,
all raw JSON traces, analysis and audit scripts, report and chart. Reruns need a
disposable copy, preservation of original evidence, and removal of START.json
as described in README.md. Equal-time update counts depend on machine speed.
