# Super-1

Hierarchical recurrent memory and function-generated weights for compact sequence models.

Super-1 explores carrying context in learned, multi-timescale state and generating transitions through structured functions. This repository contains model implementations, benchmark reports, checkpoints, training records, and a roadmap toward 1B parameters.

## Benchmark highlights

| Comparison | Result | Study |
|---|---|---|
| Hierarchy vs upstream nanoGPT at 2K | **11.64% lower cross-entropy**: 5.291 vs 5.988 | Approximately 120M parameters, 10M training tokens, one seed |
| Hierarchy vs upstream nanoGPT at 1K | **3.63% lower cross-entropy**: 5.070 vs 5.261 | Same historical study |
| Hierarchy vs in-house Transformer | **16–50% lower cross-entropy** across 512–2,048 context | Same historical study; rounded source values |
| Persistent sequence state at 2K | **Approximately 14,000× smaller** than nanoGPT KV-cache accounting | 5,504 bytes vs 73.5 MiB; not total memory |
| Pair-energy transitions vs local Transformer | **8.03% lower mean byte cross-entropy**: 1.596404 vs 1.735856 | Separate three-seed pilot |
| Cross-scale rotation vs original hierarchy | **0.73% lower mean byte cross-entropy**: 1.733472 vs 1.746261 | Separate three-seed pilot |

Cross-entropy is lower-is-better. These results compare the named baselines, not published SOTA. The historical models trained at 512 tokens; nanoGPT won at that context (5.126 vs 5.260). Its learned absolute-position extrapolation affects the longer-context comparison. The byte pilots are separate studies, not a pooled leaderboard.

## Architecture

The function-weight Transformer and its learned inference sampling remain intact. [Cached inference](sampling_library/INFERENCE.md) evaluates the same learned weight curves and router; the optional [hierarchical KV backend](sampling_library/HIERARCHICAL_KV.md) changes only attention storage/readout. The hierarchy is experimental. New [three-seed byte-text pilots](auto-research/reports/function-weight-kv-text-descent-20261018.md) preserve near-full-attention quality at 128/512-byte contexts, with a separate 8K passage diagnostic and a 100K memory-mechanics measurement. These tiny-model results do not establish SOTA quality or useful 100K context. Earlier gated-delta hybrid results describe a separate architecture and must not be attributed to this implementation.

![Function-weight hierarchy: separate text quality and memory experiments](graphs/function_weight_kv_text_descent.png)

A separate [function-weight compression pilot](auto-research/reports/function-weight-int4-20261018.md) quantizes shared bases and tied embeddings while retaining learned sampling. Across three tiny-model seeds, weight tensors are **3.58× smaller** with **0.68% higher mean perplexity** than the matched FP16 recovery control. Lossless compact packaging gives **3.14× smaller deployment files**; the original file-size gate failed and is preserved. Total process RAM remains roughly unchanged at this scale. [Load the experimental profiles](sampling_library/PACKED_WEIGHTS.md) or read the [progress and handoff file](FUNCTION_WEIGHTS_HANDOFF.md).

![Packed function weights: actual storage and text quality](graphs/function_weight_int4.png)

The original hierarchy uses learned gates and cross-scale updates to maintain fast, segment, and document-level memory. Its recurrent state stays fixed-size as context grows. Function-generated transition experiments explore structured transformations, relative geometry, and rotations.

The historical pure hierarchy, local-plus-function-weight arm, and later geometry pilots are separately identified in the evidence: they are not one identical trained architecture.

## Context scaling

![Cross-entropy across context lengths](graphs/historical_context.svg)

| Context | In-house Transformer | Pure hierarchy | Local + function weights | Upstream nanoGPT |
|---|---:|---:|---:|---:|
| 512 | 6.29 | 5.26 | 5.41 | 5.126 |
| 1,024 | 8.55 | 5.07 | 5.20 | 5.261 |
| 2,048 | 10.65 | 5.29 | 5.43 | 5.988 |

Approximately 119.7M parameters, a Qwen3-0.6B-Base teacher, 4,883 updates, and a shared distillation objective of 0.1 CE + 1.0 forward KL. The historical study uses one seed and limited evaluation batches. These are preserved report results, not a fresh benchmark replay.

## Compact persistent state

![Persistent sequence-state comparison](graphs/persistent_state.svg)

Four recurrent tensors × width 688 × two BF16 bytes = **5,504 bytes per sequence**. At 2K, the reported ratios are approximately **5,430×** versus the in-house Transformer and **14,000×** versus nanoGPT's KV-cache accounting. This excludes weights, activations, workspaces, and allocator overhead; it does not measure total inference memory or speed.

## Geometry experiments

![Every seed in the byte-model pilots](graphs/confirmed_byte_pilots.svg)

Both three-seed pilots above retain every seed and saved exact duplicate FP32 evaluations. FLOPs and wall time were not matched. See the [research catalog](auto-research/RESEARCH_CATALOG.md) for controls and protocols.

Negative findings remain available too: the latest [synthetic rotation experiment](auto-research/reports/synthetic-rotation-pr-20261015-v1.md) did not meet its promotion threshold and was not promoted.

## Perturbation-based weight optimization

The recovered [forward-only weight optimizer](auto-research/PERTURBATION_OPTIMIZATION.md) scores paired weight perturbations and uses a trust-region line search without backpropagation. Its original W560 solver/controller, frozen source manifests, 11 passing integrity tests and saved MPS decisions are now included. This historical implementation is separate from the current Transformer and minGRU models; the saved improvement is on reused optimization data, not a held-out benchmark.

![Historical W560 progress by perturbation optimization method](graphs/w560-perturbation-methods-20260822.png)

![Historical W560 local optimizer promotions and rejected candidates](graphs/w560-local-optimizer-20260816.png)

These user-supplied graphs are historical snapshots from August 22 and August 16, 2026. The first explicitly shows G11 checkpoint materialization in progress. Their full trajectories have not been independently replayed in this recovery; the [guide](auto-research/PERTURBATION_OPTIMIZATION.md#historical-graphs) separates them from the audited three-decision ledger.

The [extended research audit](auto-research/reports/perturbation-extended-recovery-20261018.md) preserves later direction, tensor-block and scalar refinement through recorded generation 28, the CE-versus-distillation comparison that retained the unchanged parent, and benchmark canary results without accuracy gains. It also explains why these results do not demonstrate new concept learning and how a symmetry-guided proposal would need to be tested.

## Indexed experiment archive

The [indexed SSD and iCloud research archive](auto-research/results/all-experiments-archive-20261018-v1/README.md) preserves source, results, logs, graphs and selected final checkpoints across the located Super-1 research. It includes failed and incomplete runs, file provenance, checksums and selective restore tools. Its [reconciled status](auto-research/results/all-experiments-archive-20261018-v1/SUMMARY.current.json) records verified upload coverage and any remaining pending objects. The release is a draft requiring authorized GitHub access. Merging the index does not itself publish release assets or back up pending files.

## Reproduce and inspect

```sh
git clone https://github.com/rohanarun/Super-1.git
cd Super-1
python3 scripts/verify_evidence.py
python3 scripts/render_charts.py
```

These commands verify checked-in evidence and regenerate charts; they do not train models or access an official test set.

- [Historical nanoGPT comparison and frozen replication package](artifacts/nanogpt-10m-20260815/)
- [Exact-source and full reproduction instructions](auto-research/results/cloud-recovery-20261015/REPRODUCTION_RECOVERY.md)
- [Runtime setup instructions](auto-research/results/cloud-recovery-20261015/RUNTIME_REPLAY.md)
- [Checkpoints, replication inputs, teacher, and runtime downloads](https://github.com/rohanarun/Super-1/releases)

Historical receipts remain unchanged for traceability. Their original URLs identify the source archive; corresponding release assets are also copied to this repository.

## Research toward 1B

Read [auto-research.md](auto-research.md) for agent instructions and [the scaling roadmap](auto-research/SCALING_1B.md) for proposed experiments. Priorities include efficient recurrence, isolated geometry controls, matched-budget nanoGPT comparisons, multiple seeds, and measured quality–memory–latency tradeoffs.

Protocols, results, logs, and all outcomes live in [auto-research/](auto-research/). Training and cloud spending require explicit approval; this publication starts no runs.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Model and experiment implementations |
| `checkpoints/` | Selected small-model checkpoints |
| `artifacts/` | Historical reports and replication packages |
| `data/`, `evidence/` | Metrics, protocols, and hash-bound records |
| `graphs/` | Benchmark figures and previews |
| `auto-research/` | Instructions, run registry, source snapshots, and results |
| `scripts/` | Evidence verification and chart generation |

## License

[MIT](LICENSE) for original code and documentation. Third-party material retains its original licenses and notices.

## Experimental sampling parameter library

[Learned sampling positions and layer-count profiles](sampling_library/README.md) preserve exact model compatibility, benchmark evidence and rejected candidates. Text-only pilots currently retain the four-layer research reference; two- and three-layer profiles expose measured quality/CPU tradeoffs. This is not a universal optimum or broad language-performance claim.
