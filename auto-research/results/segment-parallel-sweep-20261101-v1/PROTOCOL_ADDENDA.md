# Protocol addenda: segment-parallel-sweep-20261101-v1

## A1 — 2026-11-01T07:26Z — post-hoc path-decomposition diagnostic (exploratory)

Written after the primary result (max K(1e-6) = 16 on the trained rows, which meets the preregistered falsifier) and before this diagnostic ran. It doesn't change the primary metric, the decision rule or any completed output. `RESULTS.json` and `sweep_canary.py` (sha256 `0a90981a…bac`) stay as they are.

**Approval:** the donor (VibeCodyH) approved these four rows for his own desktop at 2026-11-01T07:26Z: "Yes, run it", with a budget of at most 60 CPU-seconds, the same sandbox (bwrap, no network, one thread, `capped`), the same checkpoint and synthetic tokens.

**Question:** which coupling limits convergence: the fast state carried across segment boundaries, or the segment and document states fed back into the fast GRU?

**Rows:** `trained_uniform_T1024` (stream seed 11, B=3) and `default_w688_s1` (init seed 1, stream seed 101, B=2), each run two ways:
- `fast_only`: segment and document states pinned to their reference values every sweep; only the fast boundary states are iterated.
- `slow_only`: fast boundary states pinned to the reference; only the segment and document states are iterated.

**Metric:** K(tol) on the iterated quantities only, at the same tolerances, plus the true hidden error.

**Reading, declared before running:** if K_fast_only(1e-6) ≤ 4 and K_slow_only is close to the full K, the slow path is the obstruction. The reverse means the fast path is. If both are large, the obstruction is the coupling between them. This is mechanics evidence on synthetic streams only.

### A1 outcome (2026-11-01T07:42Z, recorded after the run)

Completed 4 of 4 rows in 43.99 CPU-seconds. K(1e-6) full / fast-only / slow-only: trained 16 / 5 / 14, default init 13 / 2 / 12. The wavefront held throughout, and every row converged to within 7.6e-13. The slow path is the main obstruction. The trained fast-only K of 5 misses the declared "<= 4" by one sweep, so the fast path also contributes.
