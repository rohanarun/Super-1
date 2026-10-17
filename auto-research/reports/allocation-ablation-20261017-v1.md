# Experiment report: allocation-ablation-20261017-v1

Status: **incomplete — preregistration and analytic analysis complete; primary experiment blocked before execution.**
Approval reference: user instruction of 2026-10-17 authorising this analysis, graphs and PR. **No training was authorized and none occurred.**
Protocol: `auto-research/results/allocation-ablation-20261017-v1/PROTOCOL.json` (frozen before execution; no execution occurred)
Evidence: `auto-research/results/allocation-ablation-20261017-v1/`, `auto-research/graphs/allocation-ablation-20261017-v1/`

## Question and controls

**Primary question.** At a matched total-parameter envelope, is the hierarchy's deficit against nanoGPT partly an artifact of parameter allocation rather than of recurrence?

**Why this question and not another.** The repository holds no measured quality win for the hierarchy. Its two quality comparisons both favour the baseline:

| Comparison | nanoGPT | Hierarchy | Verdict |
|---|---:|---:|---|
| 512 matched, historical | 5.126 | 5.260 | nanoGPT wins |
| 2K extrapolated from 512 training | 5.988 | 5.291 | hierarchy "wins", but extrapolating |
| Synthetic readout CE, 3 seeds | 2.098 / 2.110 / 2.098 | 3.907 / 4.037 / 3.879 | nanoGPT wins, and sits on the ln(8)=2.079 floor |

**Strongest alternative explanation to the architecture claim: allocation.** At V=151936 the recovered hierarchy at d=688 spends **104,531,968 of 119,694,800 parameters (87.3%)** on the tied embedding, leaving **15,162,832** for the body. A parameter-matched nanoGPT at d=640/6L holds **29,534,720** body parameters — **1.95x more**. The hierarchy is simultaneously body-starved and FLOP-starved, so a parameter-matched comparison is not a capacity-matched comparison.

**Second alternative: the embedding is undertrained at this budget.** At the historical 10M-token exposure, each of the 151,936 embedding rows receives **66 tokens**. The dominant term in the loss is a lookup table that neither architecture controls.

**Named comparator.** Pinned upstream nanoGPT, parameter-matched within 1% per arm, with a separately labelled FLOP-matched track and a strong pinned RoPE positional control.

## What actually ran

Nothing was trained. This run executed:

1. `parameter_allocation.py` — arithmetic on two published formulas (`P(V,d)=V*d+32*d^2+23*d` for the recovered native hierarchy; `12*d^2+11*d` per nanoGPT block with tied `lm_head`). No device, no dataset, no checkpoint. Output `PARAMETER_ALLOCATION.json`.
2. `make_graphs.py` — three figures derived **only** from already-verified artifacts: the 12 arm/seed rows of `synthetic-rotation-radial-20261016-v1/execution/*.json`, each of which carries its own `checkpoint_sha256` and `freeze_sha256_before_confirmation`, plus `PARAMETER_ALLOCATION.json`. No metric was recomputed, re-selected or tuned.
3. `PROTOCOL.json`, frozen before any execution, and `BLOCKER.json`.

**No corpus was fetched.** `train.bin` is a 4 GB GitHub release asset and was not downloaded. **No GPU, MPS or cloud resource was used. The supplied third-party API key was not used, not stored, and is not in this repository.**

**Fidelity note on a defect I introduced and fixed:** the first render of figure 1 plotted a floor computed as `ln(2)*8 = 5.545` instead of `ln(8) = 2.079`. The wrong floor was visible only as a misplaced vertical line and would have misrepresented nanoGPT's position. It was corrected to `math.log(8)`, verified against the archived report's 2.07944, and the figure regenerated. No stored metric was altered; only the plotting constant changed, and the correction is recorded here rather than silently absorbed.

## Results

**These are analytic parameter counts. There is no quality result in this run.**

| Vocabulary | Hierarchy total @ d=688 | Embedding | Body | Body % | Tokens/embedding row @ 10M |
|---|---:|---:|---:|---:|---:|
| 151,936 | 119,694,800 | 104,531,968 | 15,162,832 | 12.7% | 66 |
| 50,257 | 49,739,648 | 34,576,816 | 15,162,832 | 30.5% | 199 |
| 32,768 | 37,707,216 | 22,544,384 | 15,162,832 | 40.2% | 305 |

Holding the recovered body width fixed and shrinking the vocabulary to 32,768 **frees 81,987,584 parameters**. Splitting embedding width from body width reaches 43.1% body share at 68.4M total — but `d_embed` below `d_body` compresses 151,936 tokens into fewer dimensions, which is a real information cost and is precisely why it stays a declared ablation rather than a fix.

**Uncertainty.** None is claimed. These are exact integer counts, not estimates; the uncertainty is in what they predict about loss, which is untested. The three-seed synthetic intervals remain as archived: rotation−radial mean +0.0153 nats, 95% paired interval [−0.0389, +0.0694], crossing zero, n=3 with unverified normality.

**Fairness caveats.** Parameter matching is not byte matching and not FLOP matching. No strong positional baseline has been run. No confirmed run exists for either architecture. The synthetic task's 7:1 command-to-readout weighting is a preregistered limitation under which the hierarchy readout CE *worsens* over training (3.26 → 3.91), which is a real diagnostic and not something this analysis repairs.

## Deployment measurements

**Unmeasured.** No model was instantiated at scale, so there are no weights, no persistent-state bytes, no activation or allocator figures, and no prefill/decode latency. Marked unmeasured rather than inferred. In particular the archived 14,003x ratio remains analytic KV-state accounting and is **not** restated here as total memory, throughput or latency.

## Decision

**Predeclared criterion: not applicable — the run did not execute.** No advance gate was evaluated, because nothing was measured.

**Dominant alternative explanation still unresolved:** whether the hierarchy's deficit is allocation or recurrence. This run shows the allocation gap is large enough to be a credible confound; it does not show it is the cause.

**Next best experiment.** `PROTOCOL.json` in this directory: vocabulary and embedding-allocation ablation, then E02 matched-context (512/512, 1024/1024, 2048/2048) with three paired seeds, equal token budget and equal tuning opportunities, parameter-matched and separately FLOP-matched tracks.

**Disconfirming control.** If the hierarchy still loses to parameter-matched nanoGPT at *every* matched context after reallocation, the allocation explanation is falsified and the recurrence carries the deficit. That is a reportable negative result and must not be reframed as an allocation problem.

**Estimated cost.** Analytic only: ~6×10^15 FLOPs per arm at ~120M/10M tokens, i.e. minutes of GPU each and low single-digit dollars for the full paired pilot. **No dollar figure is asserted**; actual Daytona pricing must be read before any budget request.

**Required new authority.** Two things, both currently missing:
1. **Storage.** The mandated SSD is full — 1.4 GiB free against ~6.5 GiB needed. See `BLOCKER.json`. No fallback to the internal volume was taken.
2. **Compute budget.** No approved budget exists for this pilot or for any 1B run.

## Integrity and cleanup

- Terminal receipt: `BLOCKER.json`, status `BLOCKS_PRIMARY_EXPERIMENT`.
- All artifacts produced by this run are preserved under `auto-research/`; prior run `synthetic-rotation-radial-20261016-v1` is untouched and was not overwritten.
- `python3 scripts/verify_evidence.py` re-run at completion; result recorded in the PR.
- **Sealed official test untouched.** Not opened, uploaded or evaluated.
- **Matching cloud resources: not applicable** — none were created.
- **No credentials committed.** The API key supplied in chat is absent from this repository and should be rotated by the user.
- New artifacts are well within the historical 50 MiB guidance: three figures (~215 KiB) plus small JSON and scripts.