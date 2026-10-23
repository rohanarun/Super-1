# Checked minKron reconstruction

Run `minkron-replay-guard-20261023-v1` adds a checked entry point around the frozen replay script. It does not change or rerun the original experiment. See the [report](../../reports/minkron-replay-guard-20261023-v1.md).

The archived runner can return normally after reporting a tensor-digest mismatch. Its output `assert` disappears under optimized Python, and it looks up unsupported continuation targets after computation. The new entry point checks source/metadata hashes and target trajectories first, verifies local array hashes before launching, and requires a manifest-bound exact tensor digest before publishing. It uses an atomic same-filesystem link so an existing or concurrently created output is preserved. Failed child outputs/stdout/stderr remain in a named diagnostic directory beside the requested output.

## Metadata-only use

This command performs no array-body read, model import or reconstruction:

```sh
python3 -B auto-research/results/minkron-replay-guard-20261023-v1/checked_replay.py \
  --arm k16 --seed 25101 --stage screen --output /path/to/new/k16-25101.pt --plan
```

All 18 screen targets and only three extension targets are recorded. Extensions exist for `k8`, `k16_lean` and `transformer_lean`, seed 25101. An unsupported target fails before child execution. `TARGET_PLANS.json` lists the saved step/exposure endpoints.

## Real reconstruction requires separate authorization

After the exact experiment/compute approval required by `AGENTS.md`, restore or reconstruct the local arrays and use the same command without `--plan`. This invokes the unchanged parent `replay.py` with its recorded steps and learning rates; it performs actual model optimization. It requires the parent's Python/PyTorch/NumPy runtime. This contribution did not install those packages or run a real reconstruction.

Success exits zero only with `PASS_EXACT_MODEL_TENSOR_REPLAY`; the receipt also records the new container's byte count and SHA-256. A child failure, malformed receipt, false match flag, wrong expected/actual digest, missing output or publication collision exits nonzero. On failure, the error names the retained diagnostic directory; inspect it before deleting it. The final path is never replaced. The output filesystem must support hard links; unsupported publication retains the staged checkpoint and fails instead of falling back to replacement.

Exact model-tensor equality is separate from serialized container equality, optimizer-state replay qualification, quality and generalization. Existing CPU/PyTorch numerical differences can legitimately produce a mismatch; the new command reports it honestly.

## Tests and evidence

```sh
python3 -B -m unittest discover \
  -s auto-research/results/minkron-replay-guard-20261023-v1 -p test_checked_replay.py -v
python3 -O -B -m unittest discover \
  -s auto-research/results/minkron-replay-guard-20261023-v1 -p test_checked_replay.py -v
python3 -B scripts/verify_evidence.py
```

Tests use synthetic files and no-op computation stubs. Three tests execute only the original `main` AST against those stubs to demonstrate its mismatch, optimization and late-validation behavior; they never import PyTorch or a model. Other tests cover all 21 valid plans, all 15 unsupported extensions, incomplete/duplicated/corrupt schedules, malformed or inconsistent child receipts, local input failures, normal/dangling-symlink outputs and publication races. Scratch directories live under the donor session folder and only their synthetic contents are cleaned.

The protocol was frozen in `1c908e63c83e7ace3e0b5b5dae3a962a52cf00d9` before testing. `INPUTS.json` binds 14 parent files, `TEST_RESULTS.json` records checks, and `TERMINAL.json` states the completion limits. This is author-checked engineering, substantially AI-assisted by Codex for @thakrarsagar. Original reconstruction and experiment credit stays with @JRowe47's PR #15, Rohan Arun and the preserved project contributors.

## Open next question

After explicit approval and data restoration, run one real recorded target through this entry point on the matching runtime. Require the exact expected model-tensor hash, preserved failure evidence if it differs, and non-overwritten publication. That integration check remains unperformed here.
