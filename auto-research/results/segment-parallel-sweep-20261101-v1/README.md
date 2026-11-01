# segment-parallel-sweep-20261101-v1

A CPU mechanics canary for E01. Can segment-parallel Jacobi sweeps replace `FastHierarchy`'s sequential loop over segments for the exact historical hierarchy? **No, not at T = 1024:** the trained 120M checkpoint needed 15–16 sweeps against a break-even of 16. A post-hoc diagnostic on two rows finds that the slow feedback path (segment and document states fed back into the fast GRU) causes most of that. The full write-up is [the report](../../reports/segment-parallel-sweep-20261101-v1.md).

| File | What it is |
|---|---|
| `PROTOCOL.json`, `PLAN.json` | Preregistration, frozen before execution (hashes in `TERMINAL.json`) |
| `PROTOCOL_ADDENDA.md` | A1: the post-hoc path-decomposition diagnostic, declared before it ran |
| `APPROVAL.json` | The donor's approvals of their own compute, quoted. Not the maintainer's |
| `sweep_canary.py` | Reference (the recovered class, imported unchanged) and the sweep, with per-sweep checks |
| `diagnostic/` | `path_diagnostic.py` and `DIAGNOSTIC.json` |
| `RESULTS.json`, `SUMMARY.json`, `summarize.py` | Raw per-sweep histories (14 rows) and the summary the report table comes from |
| `toy_check.py` | Pure-Python check of the sweep's indexing (standard library, runs in under a second) |
| `TERMINAL.json`, `CLEANUP.json`, `verify_evidence.txt`, `env.txt`, `ckpt_sha256.txt` | Terminal receipt and supporting receipts |
| `run_canary.sh` | The exact pipeline, with the donor's local wrapper name generalized |

## Replay

Running this executes a model forward on CPU, so under `AGENTS.md` rule 3 it needs approval of its own. Put `run_canary.sh` in an empty folder, clone this repository beside it as `repo/`, and copy `sweep_canary.py` into `work/` next to it. Then run `bash run_canary.sh`. It sets up a private venv (torch 2.8.0 CPU, numpy 2.2.6), downloads `pure-hierarchy-final.pt` from this repository's `recovered-120m-checkpoints-20261015` release, checks its sha256 and runs the 14 rows on one thread. That took about 194 CPU-seconds on the donor's desktop. The diagnostic isn't in `run_canary.sh`. Run it afterwards from the same folder with `venv/bin/python -I work/path_diagnostic.py --source-root repo/auto-research/results/cloud-recovery-20261015/exact-historical-source --checkpoint ckpt/pure-hierarchy-final.pt --out out/DIAGNOSTIC.json`, with `path_diagnostic.py` copied into `work/` (about 44 CPU-seconds). Without the venv, `python3 toy_check.py` needs only the standard library.

## Open next questions

1. Does Anderson acceleration (or a quasi-Newton update) on the slow boundary vector cut K(1e-6) on `trained_uniform_T1024` to 8 or fewer at the same parity?
2. How does K grow with context: T ∈ {2048, 4096, 8192}?
3. What does one sweep really cost on the intended GPU, measured against `FastHierarchy` (B ∈ {1, 16}, bf16, p50/p95)?
4. Does K on real text differ from these synthetic streams, and what residual does bf16 deployment actually need?
