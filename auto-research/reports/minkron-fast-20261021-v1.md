# Small minKron CPU experiments: minkron-fast-20261021-v1

Status: completed exploratory architecture screen and continuation. AI-assisted implementation, execution and analysis.

The user authorized smaller, faster minKron experiments on 2026-10-21. No GPU, external compute account, spending, sealed official test, automation or merge was used.

Protocol SHA256: `4fe1bd5d9339d53232b5a52f0bb6690b3ef6069d0668fc731fe7518ac596ac8d`. Read the protocol with both numbered addenda; they were recorded before language training. Exact executable source hashes are in SOURCE_MANIFEST.json.

**Finding:** the lean 16×16 model is the best recurrent candidate in this screen: 1.528M parameters, a three-seed confirmation improvement of 0.0404 nat/token versus8×8, and faster training. The short convolution did not beat it on average, and 32×32 gains were inconsistent. In the single-seed five-minute continuation, the Transformer wins quality per training minute (confirmation CE 4.859 vs 5.144) because it trains about 2.18× faster and sees more tokens. This does not establish an equal-token winner at the longer budget.

**Capability check:** all three final models still produce largely incoherent text and fail the simple factual/arithmetic prompts in the saved samples. They are useful cheap learning experiments, not useful assistants. Removing prior-token context worsens lean minKron's loss by 0.418 nat/token on the 4096-target diagnostic: it is using context, but long-memory and reasoning capability remain untested.

## Question and controls

Does increasing the matrix head size, halving the feed-forward width, or adding a short causal convolution improve a small all-recurrent language model's quality/cost tradeoff? The 8×8 baseline is the primary comparator. A RoPE Transformer with the lean MLP supplies a practical baseline. None of these are pretrained or instruction-tuned.

All arms use 3 layers, residual width 192, vocabulary 4096, tied embedding/head, RMSNorm, squared-ReLU MLP, 128-token training windows, batch 8 and 1024 targets/update. Recurrent readout width is 96: 12 heads of 8×8, 6 of16×16, or 3 of32×32. MLP hidden width is768 or 384. Conv4 is depthwise on q/k/v, identity-initialized and reset at document boundaries. Parameters and FLOPs are not matched. Three paired seeds: 25101,25102,25103; one shared LR recipe with no per-arm tuning.

## What actually ran

Each of 18 screen runs completed 186 updates/190,464 targets. The baseline, development-selected recurrent variant `k16_lean`, and Transformer then continued for 300 training seconds each on seed 25101. Fixed-token endpoints and fixed-time continuations answer different questions. Continuations resume the next training window and optimizer, with a shared time-based cosine LR restart; total lifetime training seconds differ because screening times differ.

CPU: Linux, 8-core quota/8GiB memory, 4 PyTorch threads, torch 2.9.1+cpu, Python 3.12, tiktoken 0.14.0. Total timed training: 24.81 minutes. Setup, synthetic qualification, evaluations, serialization and publication are additional. No CUDA kernels or GPU speed claim.

Two failed numerical checks are retained. The first caught Adam amplifying FP32 rounding error; epsilon was changed from 1e-8 to 1e-4 for every arm. The second caught a gradient mismatch. Recurrent normalization/decay/matrix updates/state/output normalization therefore run in FP64; projection weights, embedding, MLP and logits remain FP32. The Transformer is FP32. The original error tolerances were never loosened. All 24 final qualification rows (FP32 or FP64 parameter dtype, with FP64 recurrence in both) passed output/state/parameter-gradient/optimizer-step/streaming/reset/causality checks, including nonzero convolution taps. This precision choice adds recurrent compute and state bytes.

Data: the same document-disjoint FineWeb slice used in the 124M pilot, decoded and reencoded losslessly with GPT-2's first 4095 merge ranks plus EOT. No vocabulary fitting to heldout data. Training contains 6,757,905 small-vocabulary tokens. Development and confirmation regions were already used in prior project work: this is internal ablation evidence, not a fresh transfer test. Loss/perplexity is not comparable with the prior 50,257-token vocabulary. State resets at window starts and EOT; no long-context or persistent cross-window capability claim.

## Results

Development used 16,384 targets; confirmation used 32,768, scored twice from each frozen checkpoint. All confirmation duplicate differences are <=1e-7. Lower cross-entropy is better.

| Arm | Parameters | Dev CE mean | Confirmation CE mean ± sample SD | Training tokens/s | Mean screen seconds |
|---|---:|---:|---:|---:|---:|
| 8×8 baseline | 1,970,364 | 6.4346 | 6.4367 ± 0.0362 | 4,895 | 38.91 |
| 16×16 | 1,970,370 | 6.4132 | 6.4160 ± 0.0415 | 5,196 | 36.66 |
| 32×32 | 1,980,777 | 6.4163 | 6.4118 ± 0.0549 | 5,474 | 34.79 |
| 16×16, half MLP | 1,528,002 | 6.3903 | 6.3964 ± 0.0280 | 5,965 | 31.93 |
| 16×16, half MLP + conv4 | 1,531,458 | 6.4032 | 6.4067 ± 0.0548 | 5,336 | 35.69 |
| Transformer, half MLP | 1,672,512 | 6.7383 | 6.7261 ± 0.0632 | 10,451 | 18.22 |

Every confirmation seed, in nat/token:

| Arm | Seed25101 | Seed25102 | Seed25103 | Paired mean Δ vs8×8 | SD of paired Δ |
|---|---:|---:|---:|---:|---:|
| 8×8 baseline | 6.3956 | 6.4504 | 6.4641 | +0.0000 | 0.0000 |
| 16×16 | 6.3681 | 6.4413 | 6.4387 | -0.0207 | 0.0101 |
| 32×32 | 6.3909 | 6.4741 | 6.3704 | -0.0249 | 0.0612 |
| 16×16, half MLP | 6.3688 | 6.3955 | 6.4248 | -0.0404 | 0.0141 |
| 16×16, half MLP + conv4 | 6.3549 | 6.4010 | 6.4641 | -0.0301 | 0.0263 |
| Transformer, half MLP | 6.7945 | 6.7141 | 6.6699 | +0.2894 | 0.0991 |

Five-minute continuations (seed 25101 only):

| Arm | Total tokens seen | Additional tokens | Total training seconds | Final dev CE | Final confirmation CE |
|---|---:|---:|---:|---:|---:|
| 8×8 baseline | 1,409,024 | 1,218,560 | 338.41 | 5.2648 | 5.2788 |
| 16×16, half MLP | 1,657,856 | 1,467,392 | 331.82 | 5.1298 | 5.1443 |
| Transformer, half MLP | 3,383,296 | 3,192,832 | 318.18 | 4.8449 | 4.8594 |

Three-seed uncertainty is descriptive. Six arms were screened and one variant selected using development; no significance or familywise-error claim. The continuation has only one seed and a shorter learning history than a serious capability run. SAMPLES.json preserves every fixed-prompt completion, with no cherry-picking or retries.

## Deployment measurements

| Final model | Weight bytes | State at 128 tokens | State at 256 tokens | Prefill 128 p50 | Decode p50 / p95 |
|---|---:|---:|---:|---:|---:|
| 8×8 | 7,881,456 | 18,432 B | 18,432 B | 22.55 ms | 1.54 / 1.92 ms |
| Lean 16×16 | 6,112,008 | 36,864 B | 36,864 B | 14.16 ms | 1.77 / 1.95 ms |
| Transformer | 6,690,048 | 592,944 B | 1,185,840 B | 6.12 ms | 1.78 / 3.50 ms |

These are short CPU microbenchmarks, not a broad inference-speed finding. Recurrent state stays constant at the two measured lengths; this says nothing about successful recall at those or longer lengths.

See DEPLOYMENT.json: FP32 weight bytes, actual persistent tensor/storage bytes, batch 1 prefill 128 and decode 64 p50/p95. Two warmups; five measured prefills. Peak process RSS in run receipts includes training/runtime allocations. Isolated inference process peak and activation/workspace peaks were not measured; persistent state is not total memory.

## Decision

Exploratory gate: >=0.02 nat/token mean development gain over8×8, with all three paired seeds agreeing. Confirmation is a reused-region consistency check.

- 16×16: development gate MET; confirmation consistency MET.
- 32×32: development gate NOT MET; confirmation consistency NOT MET.
- 16×16, half MLP: development gate MET; confirmation consistency MET.
- 16×16, half MLP + conv4: development gate MET; confirmation consistency MET.
- Transformer, half MLP: development gate NOT MET; confirmation consistency NOT MET.

For continued minKron research, use lean 16×16 without the convolution as the small working configuration. Keep the Transformer alongside it: it is the stronger quality-per-minute control in this continuation. The likely next engineering target is a faithful faster recurrence implementation; the FP64 workaround remains a cost. No next training job is started.

No result promotes a general architecture-superiority, reasoning, long-memory or SOTA claim. A stronger next comparison should use more equal-token training, more than three seeds if gaps remain small, fresh document-disjoint evaluation and a separately qualified FP32 kernel. A further run is not scheduled or running.

## Integrity and cleanup

TERMINAL.json records actual completion. CHECKPOINTS.json contains retained local checkpoint hashes; corpus arrays/checkpoint bodies are deliberately excluded from Git, with RECONSTRUCTION_REQUIRED.md. Full scalar trajectories including actual LR values are compacted in ALL_STEPS.csv.gz. Source, tokenizer and data hashes are preserved. The prior archived experiment is unchanged. No official test was opened. CLEANUP.json records process completion; no unrelated workload or artifact was removed.

## Exploratory context diagnostic

This analysis was specified after 9 screen runs, before continuation selection/scoring, and never used for selection. On 4096 confirmation targets, resetting before every token worsened CE by 0.471 (8×8), 0.418 (lean 16×16), and0.566 (Transformer). Seen-token unigram losses were about 6.84, versus normal losses of 5.227, 5.107, 4.839 on this diagnostic region. It demonstrates short-context use beyond a unigram predictor. It does not establish comprehension. All 24 fixed-prompt samples are retained in SAMPLES.json.

One complete 186-update screen checkpoint (k16, seed 25101) was replayed from the saved step/LR trajectory and matched every model tensor by SHA256. This post-results reconstruction check took 39.51 seconds, with no extra quality evaluation or selection. Main training plus replay used under 25.5 minutes of walltime (setup/evaluation additional). Other checkpoints were not independently retrained.
