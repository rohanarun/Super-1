# Exact minKron core oracle and scaled-system analysis

Run `kron-exact-core-20261023-v1` is a standard-library algebra study, not a trained model or an FP32 qualification. See [DERIVATION.md](./DERIVATION.md), the [report](../../reports/kron-exact-core-20261023-v1.md), and the [primary-source assessment](../../research-scratch/kron-primary-source-notes-20261023.md).

The independent Fraction implementations compare the sequential recurrence, direct triangular system, detached-center factorization and a diagonal change of solve coordinates. The existing norm floor, before-token resets and gates are preserved. Exact dual numbers check every core-input partial in the smaller declared derivative domain. Outputs omit the same common `1/sqrt(K)` factor in all representations.

The scaled representation uses `r=sqrt(max(||v||²,1e-18))`, `z=v/r`, and `alpha=r*u`. Its matrix is diagonally similar to the original. Under the source's normalized domains, a state-impulse argument bounds both its entries and inverse entries by one, hence `kappa_inf<=T²`. This is a proposed algebraic contract, with an explicit derivation and separate finite checks. It does not bound full model gradients or guarantee improvement for each matrix: all unfavorable reverse-scaling and unchanged unit-norm controls are retained.

## Reproduce

From the repository root, using Python's standard library:

```sh
python3 -B -m unittest discover \
  -s auto-research/results/kron-exact-core-20261023-v1 -p test_exact_core.py -v
python3 -B auto-research/results/kron-exact-core-20261023-v1/verify_exact.py
python3 -B scripts/verify_evidence.py
```

The finite verifier has a fixed 600-second wall cap. This author's run took about 100 seconds on the existing session CPU. It prints JSON to stdout and progress to stderr, without writing artifacts or importing a tensor library. On failure it reports the exact case or incomplete-domain counts. Runtime is verification overhead, not a model benchmark.

`EXACT_RESULTS.json` records 3,840 forward cases, 11,520 whole-block comparisons, 91,136 chunk comparisons, 3,840 state-contraction and scaled-matrix bound checks, and 64 derivative cases / 1,536 directions, with zero exact discrepancies. The deterministic result digest is `ff6df29c68b803ffd59ecd33e46060c420ed1b9decaf3c1a314843b1caea020c`; timings may differ. `PROTOCOL.json` was frozen at `1bb68047e1444ccec97681f7f43e672208a030c7` before computation. `INPUTS.json` binds the original source and failure receipts.

## Limits and open questions

Finite checks do not prove all inputs. Rational fixtures use rational norm roots, positive rational decays and arbitrary positive detached centers. They do not test exp/log, rounded prefix sums, tensor-library solves, upstream normalization derivatives, a full model, performance or language quality. Floor-kink derivatives are excluded. Existing archived FP32 qualification failures remain unresolved.

The primary-source note proposes a two-token cumulative-log precision canary. It remains a numerical hypothesis until separately approved execution in a compatible runtime. A rounded cumulative sum cannot be repaired simply by re-centering it afterward. The algebraic scaling also does not fix that separate representation problem. Run the smallest synthetic canary before proposing a wider qualification or training job; preserve negative results and original tolerances.

## Credit

Original recurrence/adaptation/experiments: @JRowe47's PR #15, Rohan Arun, the preserved project contributors and NX-AI/KronLSTM provenance. Diagonal equilibration is an established numerical technique; this study claims no new model or general novelty. New exact oracle, derivation and finite checks are substantially AI-assisted by Codex for @thakrarsagar; primary-source reading was delegated under the research skill. All new claims remain author-checked.
