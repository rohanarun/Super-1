# Contributing and independent review

**Agents and open-ended "contribute" tasks: start with [auto-research.md](auto-research.md).** It has the current objective, the experiment dependency graph (E00–E06), the scientific contract every run must follow, and the result-storage layout. If a user gave you a specific assignment, do that instead of picking a new task. [AGENTS.md](AGENTS.md) points here too.

Super-1 is a research repository, not a finished model. The benchmark highlights in the [README](README.md) compare named baselines at small scale. Several rest on one seed or on historical reports that haven't been freshly replayed. Corrections to the code, the evidence, the claims or the attribution are welcome as issues or pull requests.

## Before you start

- **Check what exists.** Read the open issues and PRs, [auto-research/RUN_REGISTRY.json](auto-research/RUN_REGISTRY.json) and [auto-research/RESEARCH_CATALOG.md](auto-research/RESEARCH_CATALOG.md). Don't repeat a completed experiment as if it were untested. The odd-query-shape control, for example, already has a saved failed advance gate.
- **Verify the evidence first:** run `python3 scripts/verify_evidence.py`. A hash check confirms stored evidence; it is not a new evaluation.
- **Claim larger tasks.** For anything substantial, open an issue titled `[claim] <experiment or task>`. Claims are advisory, and findings stay provisional until their evidence is checked.
- **Write handoffs.** A focused finding, a reproduction or a handoff note is a valid outcome. A PR isn't mandatory.

## Compute and authorization

- **Your own compute.** External contributors run experiments on their own compute, at their own cost. Nothing in this repo authorizes spending on the maintainer's accounts, cloud credits or hardware.
- **Maintainer agents** follow the stricter rules in `auto-research.md`. Training, cloud spending, GPU provisioning and evaluations need explicit approval of the exact experiment and budget, and no open-ended "keep researching" loop is enabled.
- **The official-test seal applies to everyone.** Do not open, upload or evaluate the sealed WikiText-2 official test. Byte CE, training loss and local pilot wins don't satisfy its eligibility gates.

## Kinds of contribution

### 1. An experiment result (improvement, null or negative)

Follow the scientific contract in `auto-research.md`:

- **Preregister.** Freeze a filled-in `auto-research/PROTOCOL_TEMPLATE.json` before running. It needs one primary hypothesis, a named comparator, a primary metric, a minimum useful effect and stop criteria.
- **Compare fairly.** Use the same data, tokenizer, exposure and tuning opportunities for every arm, at least three paired seeds, and every seed shown. Say whether the comparison is parameter-, FLOP- or walltime-matched.
- **Keep the original hierarchy intact.** It's a contract: never silently replace its learned gates, cross-scale coupling or state semantics. An optimization must pass parity tests against the exact reference.
- **Store results in the standard layout:**
  - `auto-research/results/<run_id>/`
  - `reports/<run_id>.md`, filled in from `REPORT_TEMPLATE.md`
  - `graphs/<run_id>/`
  - `logs/<run_id>/`
  - an entry in `RUN_REGISTRY.json`

  Use a new, unique run ID; never overwrite a completed run.
- **Report every outcome.** Negative, null and inconclusive results are welcome and go in the same registry. A tiny, uncertain gain is "inconclusive", not a breakthrough.

### 2. Engineering (kernels, inference, packing, sampling library)

- **Keep the same math.** A faster or fused implementation is a hypothesis until it matches the reference. Check outputs, intermediate and final state, loss, gradients and a complete optimizer step, across seeds, lengths, chunk splits, batch sizes and dtypes.
- **Fix tolerances before measuring.** Declare them in advance and don't relax them after a failure.
- **Measure what you claim.** Report total deployed bytes and latency rather than inferring them, broken down as `auto-research.md` describes. Persistent state is not total memory.

### 3. Review and reproduction

- **Reproduce historical results:** a fresh replay of a historical result such as the nanoGPT comparison or the byte pilots, on matching or documented hardware.
- **Audit claims:** check a README or report statement against its evidence, and catch over-claims, mismatched baselines or pooled results that shouldn't be pooled.
- **Check prior art:** a primary-source comparison of a mechanism (pair-energy transitions, cross-scale rotation, function-generated weights) with existing work.

### 4. Proposals

- **What to include:** an experiment you can't run yet goes in an issue or in `auto-research/research-scratch/`. Give the hypothesis, its disconfirming control, its place in the E00–E06 graph, and an estimated resource cost.
- **Status:** mark it as a proposal. A proposal isn't evidence.

## Reporting a problem

- **Locate it exactly:** name the file, function, run ID, report section or registry entry.
- **Give evidence:** a failing command, a hash mismatch, a counterexample, or the missing control.
- **Separate evidence types:** a synthetic canary or mechanics test is different from a language-quality result. A duplicate evaluation checks numerical repeatability, not generalization.

## Rules for changes

- **Preserve evidence.** Don't modify sealed runs, archived evidence, historical receipts or frozen source snapshots. Add new artifacts instead, and update `MANIFEST.sha256.json` for anything you add.
- **Regenerate, don't hand-edit.** Graphs and charts come from verified artifacts via `scripts/render_charts.py` or the run's own chart source. If you change a generator, include the regenerated output.
- **Keep claims consistent.** When a result changes a claim, update the README, `auto-research.md` and the research catalog together, so no file still calls an older result "current".
- **Commit only safe files.** Never commit credentials, sealed test content, restricted datasets or unrelated personal data. Large checkpoints belong in releases or an external store, recorded with their hashes.
- **Don't overclaim.** Nothing here establishes SOTA or novelty on its own. State the scale, seeds, baseline and what remains unverified.

## Credit and AI disclosure

- **Build on others openly.** Name the work you build on and say what yours adds, even if a parallel result has a stronger headline.
- **Credit is kept.** Useful contributions stay credited in [CONTRIBUTORS.md](CONTRIBUTORS.md), including ones later superseded or not merged.
- **Disclose AI help.** Say if a contribution was substantially AI-assisted, and name the agent or session where useful.
- **Licensing:** contributions are accepted under the [MIT License](LICENSE). Third-party material keeps its original license and notices.
