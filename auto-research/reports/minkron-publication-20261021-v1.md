# minKron: compact memory results and a 124M CPU feasibility run

Status: completed external experiments, retrospectively published for review.
The user explicitly requested this PR to `rohanarun/Super-1`. Source development
and publication were substantially assisted by ChatGPT/Codex under Joshua Rowe's
direction. [Evidence index, commands and attribution](../results/minkron-publication-20261021-v1/README.md).

The one-head 8×8 minKron is a promising compact-memory candidate: at the same
1,024-byte persistent hierarchy state and a slightly smaller training CPU budget,
it achieved higher mean accuracy on both primary synthetic retrieval tasks than
linear minGRU, with fewer parameters and less training exposure. Its current CPU
implementation is slower and uses more training working memory. This supports
further investigation of the capability/cost tradeoff, not an overall efficiency
win or established language-model superiority.

## Question and controls

The compact experiment asks whether Kronecker memory can improve retrieval within
the established width-64 minGRU hierarchy's state, weight and training-CPU budgets.
It retains the fast/segment/document organization, shared embedding/readout and
paired initialization. It substitutes the fast memory only. There are no
curve-generated weights in this comparison.

Three compact arms retain normalized SiLU q/k/v, vector forgetting, independent
write/erase gates, output gating and gated RMSNorm. They remove the convolution
cache and redundant stored output. The CPU core uses a direct triangular solve,
with packed projections; these are adaptations of the authors' KronLSTM math,
not executions of the upstream fused GPU kernels.

Each arm receives two learning-rate pilots, up to 100 process-CPU seconds each,
on seed 21901. The selected rates were frozen before main runs: minGRU 0.003,
1×8×8 0.001, 4×4×4 0.001, and 2×4×4 0.003. Main seeds are 22101–22103.
Every compact stage is capped by its paired minGRU CPU time **and** update count.
Slower arms therefore see fewer examples. Deterministic stage streams are paired
prefixes, not identical total exposure. This is a shared-host process-CPU-budget
comparison, not an equal-FLOP or equal-token comparison.

The [original protocol](../results/minkron-publication-20261021-v1/compact/protocol.json)
and its source seals are preserved unchanged. Main training uses one CPU thread
per worker, FP32, AdamW, batch 32 and a short/medium/long-delay curriculum.
MinGRU completes 10,000/1,000/3,000 updates. Its paired CPU mean is 443.12 seconds;
1×8×8 uses 440.41 seconds. CPU accounting includes data generation, forward,
backward, clipping and optimization; it excludes setup, checkpoint I/O, validation
and engineering. Baseline/compact worker concurrency differs (three/six).

## Synthetic results

Final-checkpoint confirmation uses 1,024 examples per primary task and seed at
nominal age 512 (actual target age 509–512). Final checkpoints, rather than the
best confirmation result, are scored. Three seeds support a descriptive result;
they do not establish a broad reliability advantage or statistical significance.

| Arm | Parameters | Persistent state | Competing writes, mean | Delayed overwrite, mean | Training input tokens, mean |
|---|---:|---:|---:|---:|---:|
| Linear minGRU | 111,048 | 1,024 B | 98.9258% | 99.3815% | 30.262M |
| minKron 1×8×8 | 95,009 | 1,024 B | **99.9674%** | **99.8372%** | 19.593M |
| minKron 4×4×4 | 102,064 | 1,024 B | 98.4701% | 98.2096% | 10.184M |
| minKron 2×4×4 | 94,590 | 896 B | 94.3685% | 97.2331% | 12.041M |

State includes the 768-byte shared hierarchy state and the fast memory. Tensor
backing storage is checked, not inferred only from view shapes. Token counts
above are input exposures in a synthetic task, not language-model target counts.

| Arm | Seed | Competing writes | Delayed overwrite |
|---|---:|---:|---:|
| minGRU | 22101 | 100.0000% | 100.0000% |
| minGRU | 22102 | 100.0000% | 100.0000% |
| minGRU | 22103 | 96.7773% | 98.1445% |
| minKron 1×8×8 | 22101 | 99.9023% | 99.8047% |
| minKron 1×8×8 | 22102 | 100.0000% | 99.9023% |
| minKron 1×8×8 | 22103 | 100.0000% | 99.8047% |
| minKron 4×4×4 | 22101 | 99.4141% | 98.6328% |
| minKron 4×4×4 | 22102 | 96.7773% | 97.7539% |
| minKron 4×4×4 | 22103 | 99.2188% | 98.2422% |
| minKron 2×4×4 | 22101 | 94.9219% | 96.5820% |
| minKron 2×4×4 | 22102 | 97.8516% | 98.4375% |
| minKron 2×4×4 | 22103 | 90.3320% | 96.6797% |

The 1×8×8 advantage is +1.0417 and +0.4557 percentage points in the two means.
It is driven mainly by minGRU's third seed: the first two minGRU seeds are perfect
on both tasks. MinKron 1×8×8 has lower observed dispersion, but it does not win
every paired seed. Both alternative compact layouts underperform minGRU in mean
primary accuracy. A smaller, 32-value minGRU was not tested, so the reduced-state
arm does not establish the optimal minGRU frontier.

Count receipts for all four task modes, ages 32/128/512/2048, and state-zeroing
diagnostics are retained. Raw individual predictions require reconstruction. Longer-delay results are mixed: at age 2048, 1×8×8's
repeat-task accuracy ranges from 89.06% to 98.05%, whereas minGRU ranges from
98.05% to 100%. The primary-task result should not be generalized to every memory
condition. [Detailed report reconstruction](../results/minkron-publication-20261021-v1/RECONSTRUCTION_REQUIRED.md).

The chart generator is included; rendered figures are omitted per the user's
request for code and reconstruction placeholders.

## Measured deployment and training cost

Serial interleaved FP32 CPU timings, one thread; training/prefill length 513,
training batch 32. These measurements are separate from concurrent training CPU
budgets. They are medians in the recorded timing protocol, not GPU forecasts.

| Measure | minGRU | minKron 1×8×8 | Ratio |
|---|---:|---:|---:|
| Decode, batch 1, per step | 0.1272 ms | 0.4333 ms | 3.41× |
| Decode, batch 32, per step | 0.2297 ms | 0.6706 ms | 2.92× |
| Training step | 198.35 ms | 307.56 ms | 1.55× |
| Prefill | 67.26 ms | 99.40 ms | 1.48× |
| Peak live training tensor allocation | 123.22 MiB | 139.35 MiB | 1.13× |
| Peak live inference tensor allocation | 44.73 MiB | 44.67 MiB | ~1.00× |
| FP32 parameter storage | 444,192 B | 380,036 B | 0.86× |

Profiler allocation excludes Python/runtime memory and untracked library
workspace. It is not whole-process RSS. GPU allocator-reserved memory, optimized
GPU latency and p95 serving latency are unmeasured here. Equal persistent state
does not imply equal total memory. No compact arm passes the stored descriptive
screen requiring both mean accuracies, decode latency and peak working memory to
be no worse than minGRU; that negative overall result is preserved.

## Common-decoder integration and the separate 124M run

The initial integration implements minKron, minGRU-linear and RoPE/SDPA Transformer
mixers in a common small decoder. The 15.15M minKron diagnostic memorized 128 fixed
GPT-2-tokenized positions in 100 updates (NLL 10.8664 → 0.000362). This verifies
learning mechanics, not held-out language quality. The paired 20M-token GPU
comparison in `comparison_contract.json` has **not run**.

A separately authorized CPU experiment trained a 123,376,800-parameter minKron
decoder from scratch: 12 layers, width 768, 48 8×8 heads per layer, MLP width 3,584,
tied 50,257-token embedding/head, pre-RMSNorm and ReLU-squared MLP. It contains no
attention blocks, n-gram tables or GSA hierarchy. "NanoGPT" refers to the decoder
and training setting, not an unmodified GPT-2 model.

| Resource probe: heads/layer | Parameters | Readout channels/layer | FP32 state, all layers |
|---:|---:|---:|---:|
| 12 | 123,562,992 | 96 | 36 KiB |
| 24 | 123,500,928 | 192 | 72 KiB |
| 48, selected | 123,376,800 | 384 | 144 KiB |
| 72 | 123,252,672 | 576 | 216 KiB |

These were systems probes, not trained quality comparisons. Four threads, FP32,
foreach AdamW, chunks of 32 and two 256-token sequences per update were selected.
BF16 and eight threads were slower in this environment. PyTorch reports AVX512;
no custom AVX-512 kernel or GPU kernel was implemented. A 16-update development
pilot selected LR 0.001 from 0.0003/0.001/0.003; the main run restarted from
initialization with seed 24101.

| CPU run outcome | Measured value |
|---|---:|
| Timed training loop | 1,501.16 s / 25.02 min |
| Optimizer updates / next-token targets | 262 / 134,144 |
| Full-loop throughput | 89.36 targets/s |
| Process peak RSS | 4.40 GiB |
| Development NLL, same 8,192 targets | 10.9841 → 7.2686 |
| Held-out confirmation NLL / perplexity | 7.2867 / 1,460.75 |
| Held-out token accuracy | 9.6069% |
| Batch-1 sampled decode throughput | 31.08 tokens/s |
| Nonfinite steps / clipped steps | 0 / 262 |

The 1,500-second budget includes monitoring and periodic saves; setup, pilots,
initial/final evaluation, final save and generation are extra. The final timed
checkpoint is used. Training consumes a non-repeated permutation of windows
from a pinned FineWeb `sample-10BT` shard, with text-hash document-disjoint splits,
exact deduplication, reset at window/EOT boundaries and full BPTT within windows.
Near-duplicate leakage is not ruled out. This is a custom split, not the official
NanoGPT speedrun evaluation. Full source/tokenizer/split hashes are retained.

The model beats a same-target unigram diagnostic (NLL 7.8279), and resetting
recurrent state every token worsens NLL by 0.1129 on a small 2,048-target diagnostic.
Those show early predictive learning and some context use. The ten fixed samples
remain broken prose and do not follow their prompts reliably. This is a severely
undertrained, one-seed feasibility run without a trained Transformer/minGRU
control at this scale. It does not establish model quality or comparative speed.
[All ten generations](../results/minkron-publication-20261021-v1/nanogpt/artifacts/cpu-124m/minKron_124M_CPU_10_Generations.txt)
are included without resampling. The HTML report and figures have
[reconstruction instructions](../results/minkron-publication-20261021-v1/RECONSTRUCTION_REQUIRED.md).

The factored GEMM core avoids a large pairwise temporary when numerically safe.
It falls back to the stable core for excessive cumulative decay. In the main
training receipt, 19,368 of 29,760 calls fell back (65.1%), so the short systems
probe's 203.4 targets/s did not translate to final-run throughput. Improving this
path while retaining output/state/gradient/optimizer parity is a concrete
engineering opportunity, not a demonstrated future speedup.

## Verification, failures and preservation

All 405 entries in the three original archive manifests matched their bytes
during local preparation. Before slimming, the audit reconciled all 67,584 raw
predictions, 192 confirmation cells, 12 main budget records, 8 LR pilots, the
262-step scale trace and all ten generations. Its receipt is preserved as
`PREPARATION_AUDIT.json`; it does not imply those bulk artifacts are in this PR.

Following the user's request, the final PR includes 182 byte-identical source and
compact receipt files, plus the publication report, standard-library verifier and
explicit reconstruction placeholders. Every omitted original member retains its
path, size and hash in `IMPORT.json`. The current verifier checks the included
hashes, compiles 85 Python sources without executing them, reconciles count
receipts/means and CPU budgets, and verifies the scale summary and ten generations.
The repository's pre-existing evidence check passed 2,479 hashes before import;
the final manifest covers all included publication files.

Original numerical checks cover reference recurrence, gradients, state,
chunk/reset behavior, packed projections and optimizer steps. They are preserved
receipts, not a new independent replication. The factored-core test uses combined
absolute/relative tolerances; its raw absolute gradient error reaches 0.0166 in
the tiny-value stress case. It should not be described as bitwise equality.
Saved trained chunk/recurrent logits differ by at most 2.38e-6 in the CPU run's
assessment. The final large checkpoint was not fetched or reloaded for this PR.

The lightweight publication omits 226 archive members, including checkpoints,
raw prediction arrays, corpus arrays/document manifests, detailed histories/logs,
generated images/HTML and redundant patches. The separate 1.48GB CPU checkpoint's
hash and recorded byte count are included, but its bytes are external. Dataset
source/split manifests and reconstruction code remain in Git. Restoring original
bundles gives exact historical bytes; a fresh run can reconstruct the experiment
but cannot promise identical wall-time trajectories or checkpoint hashes. No
runtime installation, corpus download, training, model evaluation, paid compute
or official-test access was performed for this publication.

## Decision

Keep 1×8×8 minKron as a promising experimental option. The compact study supports
a small synthetic accuracy/observed-stability advantage with fewer parameters and
examples under the measured CPU cap. The strict overall resource screen failed,
and the other compact layouts are weaker. The 124M run adds scale feasibility,
not evidence of language-model superiority.

The next useful quality comparison is the existing common-decoder minGRU/minKron/
Transformer experiment with matched data and multiple seeds; the next systems
target is the factored-core fallback. Neither is launched by this PR. A new
compute budget would be needed for substantive training. All existing production
model paths and training defaults remain unchanged.
