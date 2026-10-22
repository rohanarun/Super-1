# Research catalog and recovery status — 2026-10-15

## Compact minKron and CPU scale feasibility — 2026-10-21

[Report](reports/minkron-publication-20261021-v1.md) and
[source/evidence index](results/minkron-publication-20261021-v1/README.md): three-seed
synthetic retrieval at the minGRU state/training-CPU budget, all compact layouts,
common-decoder integration, and the separate 123.38M 25-minute CPU text run.
One 8×8 memory improves the two mean primary accuracies with fewer parameters,
but decode is 3.41× slower and training tensor allocation is 13.1% higher; the
overall resource screen fails. The scale run establishes early learning and
resource feasibility, not a language-quality win. All ten generations and
negative/failed outcomes are retained. This is retrospective publication with
no new training or model evaluation.

## Perturbation optimizer recovered — 2026-10-18

The [W560 perturbation optimizer](PERTURBATION_OPTIMIZATION.md) is now preserved as source: 85 original frozen files plus a separately inventoried later MPS snapshot and all three located completed decisions. The recovery includes 137 imported files, fresh passing toy tests, and a read-only audit of all 46 saved candidates. One historical promotion and two rejections remain visible; scores are from reused optimization data. Checkpoints/token arrays stay external and no new language training or W560 evaluation was launched.

## Located input uploads completed — 2026-10-15

The exact 4GB training corpus (three ordered parts), validation corpus, teacher weights, tokenizer/config files and located pinned benchmark caches are now in the [private input release](https://github.com/rohanarun/function-weights-research/releases/tag/replication-inputs-20261015). Every remote asset digest matches its local SHA-256. See [download/reconstruction instructions](results/recovered-20261015/DOWNLOAD_INPUTS.md) and [input manifest](results/recovered-20261015/INPUT_MANIFEST.json). Earlier external-only statements are historical and superseded. Unlocated artifacts remain explicitly unresolved.

## Benchmark caches located — 2026-10-15

All four required evaluation caches (HellaSwag validation, PIQA validation, LAMBADA test, ARC-Easy validation) are readable in the SSD promotion-benchmark bundle. Directory revisions match the historical contract; current hashes are recorded in [BENCHMARK_CACHE_RECOVERY.json](results/recovered-20261015/BENCHMARK_CACHE_RECOVERY.json). No historical Arrow content hashes were available for byte-identity comparison, and no evaluation was rerun. Earlier missing-cache notes below are superseded. NanoGPT final checkpoint, three exact core sources, and the original runtime remain unresolved. The SSD-wide scan was partial, not exhaustive.

## Hidden-staging checkpoint recovery — 2026-10-15

The original pure hierarchy, in-house Transformer and local hierarchy final checkpoints were located under the SSD's hidden .codex-staging/qwen06b-benchmark.k3YJBn directory. All three match their recorded SHA-256. This corrects earlier statements that the hierarchy final checkpoint was unlocated; hidden staging files had been missed.

See [checkpoint recovery receipt](results/recovered-20261015/CHECKPOINT_RECOVERY.json) and [private release assets](https://github.com/rohanarun/function-weights-research/releases/tag/recovered-120m-checkpoints-20261015). The receipt distinguishes local hash verification from completed remote uploads.

The nanoGPT final checkpoint remains unlocated. Three deployed source files, offline benchmark caches and the exact executable runtime remain unresolved. Current source identity is still compatible/statistical reproduction, not a promise of byte-identical rerun. Earlier dated recovery notes below remain historical records, superseded by this update where applicable.

This catalog links original research rather than merging different models or metrics. Research remains paused. Archives preserve existing bytes; a stored result or a hash inventory alone is not a new confirmation.

## Recovered historical study and nanoGPT comparison

[Original report, expanded source, result JSON, configs, frozen archives and provenance](../artifacts/nanogpt-10m-20260815/). [Artifact index](../artifacts/nanogpt-10m-20260815/ARTIFACT_INDEX.json).

Exact training/validation corpus and teacher weights exist on SSD with matching hashes. Original final hierarchy/nanoGPT checkpoints remain unlocated locally. A previous cloud disk inspection was blocked by credential-file permissions; no availability/deletion conclusion follows.

Raw nanoGPT task mean 0.2863377193 exceeds hierarchy 0.2802061404. Hierarchy wins longer-context CE, nanoGPT wins 512 CE. The 14,002.6x figure describes analytic sequence state, not measured total memory.

## Research families preserved in this recovery

| Archive | Coverage | Interpretation |
|---|---|---|
| research-evidence-20261002.zip | Energy geometry, adjoint/optimizer parity, execution canaries, shape controls, transfer, original hierarchy, incomplete word sweep | Mixed completed, negative, synthetic and draft material; inspect terminal receipts |
| disparity-constraints.zip | Constrained distance, capacity/inference budgets, distance/dot/gated weight experiments, phase/energy and hierarchy pilots, local nanoGPT comparisons | Different hypotheses and tasks; do not pool scores |
| disparity-coordinates.zip | Earlier disparity-coordinate experiments and source | Exploratory records; do not infer capacity or novelty |
| historical-reproduction.zip | Original 120M in-house study reproduction/provenance/results | Compatible snapshot with exact recorded config, not every deployed core file |
| function-weights-reproduction.zip | Earlier function-weight reproduction source and evidence | Separate prior work, not the nanoGPT 120M result |

Archives are under [results/recovered-20261015/](results/recovered-20261015/); [INVENTORY.json](results/recovered-20261015/INVENTORY.json) records each member's source, bytes and SHA-256, plus exclusions/read errors. Only files actually present are archived. This is a catalog of located research, **not an assertion that every historical run or checkpoint has been recovered**.

The existing [confirmed byte pilot metrics](../data/confirmed_byte_pilots.json), [selected checkpoints](../checkpoints/) and [incomplete word sweep](../data/incomplete_word_sweep.json) remain unchanged.

## Additional findings that change priorities

### Dedicated odd-query-shape control

The saved energy-vs-odd-query confirmation completion reports zero duplicate drift but **overall_local_advance_gate_passed = false**. For seed 301892410, odd-query shape CE 1.6214159419 beats pair-energy 1.6235875611. Pair-energy wins the other two seeds. Therefore do not claim energy is uniformly superior to equal-capacity nonlinear flexibility, and do not describe this control as never tested.

[Read-only bound-receipt audit](results/recovered-20261015/ADDITIONAL_RECEIPT_AUDIT.json) distinguishes hash-checked saved receipts from numerical re-evaluation. Raw terminal and confirmation files are in research-evidence-20261002.zip.

### Frozen transfer

TinyStories transfer-run/RESULT.json contains saved frozen-transfer results and duplicate records. Preserve and inspect its protocol, data provenance, source gate and transfer receipt before making an independent-domain claim. Deliberately corrupted receipt-fault fixtures are excluded from primary evidence archives. Presence of RESULT.json alone is not a successful transfer gate.

### Faster analytic execution

ROOT_ADAMW_FIDELITY_FAILURE_20261002.json and ANALYTIC_OPTIMIZER_TRAJECTORY_CANARY_20261002.json document optimizer-fidelity concerns despite close forward/gradient comparisons. Do not promote a faster backward path solely from matching logits. Keep the native reference and complete optimizer-step checks.

### Transport/readout

Original-hierarchy-reference contains exact-reference tests, cross-scale rotation pilot, transport invariance diagnostics and a draft readout factorial. Synthetic nonzero gradients and preserved scale mean do not prove language improvement. No draft factorial should be called completed.

### Word-level scaling

Word-energy-native has six completed trial candidates and lacks a terminal completed sweep. The Transformer-favoring first-seed selection result must remain visible. Do not fabricate a completion receipt to unblock another run.

## What remains necessary for full replay

1. Recover final student checkpoint bytes from an existing authorized remote artifact store/preserved disk, verifying the published hashes. A metadata query is not checkpoint recovery. Do not provision/attach resources without approval.
2. Recover the remaining byte-identical deployed hierarchy core files or explicitly retain compatible/statistical reproduction status. Original common.py is present in the historical bundle provenance.
3. Verify benchmark caches at the pinned dataset revisions and the tokenizer assets, not merely teacher weights. Missing cache bytes are not equivalent to missing benchmark result JSON.
4. Check runtime/container availability: the archived CUDA/PyTorch versions are recorded, not verified installable today.
5. Verify each additional research family's source/checkpoint/data bindings and terminal gates before promoting its archived claims.
6. Complete a new fair train/eval-context experiment only after explicit authorization. Recovery itself does not authorize training.

## Retrieval versus missing data

The earlier initial GitHub audit searched incomplete SSD project copies and missed dedicated repro-packages and the relocated iCloud Documents project. Those findings should not be interpreted as deletion. Large model/corpus files were intentionally omitted from lightweight code archives. Their absence from a ZIP is not evidence that they were lost.

## Function-weight KV text descent, 2026-10-18

[Report](reports/function-weight-kv-text-descent-20261018.md): two three-seed text pilots with learned KV pooling, preserved function weights and sampling, a fresh 8K passage diagnostic, hashed profile libraries, and a separate 100K memory-mechanics measurement. Passed the narrow quality gates; original short/8K total-memory gates failed. No SOTA or 100K quality claim.

## Extended perturbation records — 2026-10-18

The [SSD/iCloud follow-up audit](reports/perturbation-extended-recovery-20261018.md) preserves 271 additional historical files, 21 later heads/decisions, tensor/scalar refinement, CE/KD parent retention, benchmark regressions, one-backward scalar screening, historical test skips and pipeline failures. Source/receipt recovery does not establish new concept learning or authorize a new experiment.
