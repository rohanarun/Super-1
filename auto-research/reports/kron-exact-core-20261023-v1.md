# Experiment report: kron-exact-core-20261023-v1

Status: completed exact-arithmetic study and proposed numerical contract. No tensor-library/model execution or architecture promotion.

Approval: donor's 2026-10-23 continued contribution authorization, multiple focused PRs within the original 480-minute session. [Claim #21](https://github.com/rohanarun/Super-1/issues/21). Base `847bb75042e01b2c879272f97e72746f285c8872`; seed `dcbb91d07e51cf13ee6fda1fefc14ca2`.

Protocol SHA-256 `4dbb2d89cbedea4be9e1358d25363b487d8e882e825e0234ea6e9be6b8c1596d`, frozen at `1bb68047e1444ccec97681f7f43e672208a030c7` before computation. [Evidence](../results/kron-exact-core-20261023-v1/), [derivation](../results/kron-exact-core-20261023-v1/DERIVATION.md), [primary-source assessment](../research-scratch/kron-primary-source-notes-20261023.md).

## Question and controls

Can the original sequential, direct triangular and factored core be checked independently in exact arithmetic, including resets and the actual value-norm floor? Does the equivalent diagonal change `alpha=sqrt(m)*u`, `z=v/sqrt(m)`, `m=max(||v||²,1e-18)` give a bounded system without changing the equations? This is E01 reference/qualification work, not a new model or speed hypothesis.

The strongest competing explanation is a hidden equation/reset/floor change or a condition-number gain that merely shifts the problem. Controls include a separate sequential recurrence, hand-computed nonzero-state/reset/zero/floor cases, exact directional derivatives, every declared chunk composition, and both favorable and unfavorable two-step conditioning examples. The unchanged unit-norm case prevents presenting scaling as an automatic improvement.

## What actually ran

Python standard library on the existing session CPU. No dependency installation, PyTorch/model import, training, score evaluation, checkpoint deserialization, corpus download, cloud or paid resource occurred.

The four 32-bit words of the supplied seed generate rational vectors/decays/gates. The predeclared finite forward domain contains lengths 1/2/4/8; key/value dimensions (1,1), (2,1), (2,2), (4,3); four seeds; unit/zero/near-floor/mixed regimes; zero/nonzero initial states; all reset patterns through length four and eight explicit length-eight masks. All chunk compositions are checked through length four; five explicit length-eight partitions are checked.

The derivative domain has T=2,K=V=2, four seeds, two value regimes, zero/nonzero state and all four reset patterns. Each of 64 cases checks all 24 independent input directions, including initial state and per-coordinate decays. Derivatives are of core inputs, not raw normalization parameters; the floor kink is excluded. Rational norm roots permit exact dual arithmetic. A common `1/sqrt(K)` output factor is omitted from every implementation, leaving equality and the corresponding fixed-K derivatives unchanged.

All 11 bound parent source/protocol/failure inputs matched. Fourteen hand/edge/control unit tests pass. Full verification took 99.6492 wall seconds / 99.3732 CPU seconds, below the fixed 600-second stop. No implementation correction or scope pivot was needed.

## Results

**Exact finite-domain verification passed**, with zero discrepancies across 3,840 forward cases, 11,520 whole-block comparisons, 91,136 chunk comparisons, 3,840 state-contraction checks, 3,840 scaled matrix/inverse bound checks, and 1,536 exact derivative directions in 64 cases. Result digest: `ff6df29c68b803ffd59ecd33e46060c420ed1b9decaf3c1a314843b1caea020c`. Source hashes, all conditioning controls and coverage counts are in `EXACT_RESULTS.json`.

The derivation expands the original unit-lower system `A*u=b` and shows `B=R*A*R^-1`, `alpha=R*u`, with identical outputs/state. The unchanged normalized domains imply a nonexpansive homogeneous state transition. An impulse argument bounds entries of B and B^-1 by one, yielding `kappa_inf(B)<=T²`. The finite checks support this separately written argument; they do not prove the infinite domain or qualify floating-point implementations.

The largest sampled scaled infinity-condition number was exactly `1185728597/131072000` (about 9.05), on a length-eight case, below its bound of 64. This is a finite observed maximum, not a tight global bound.

All nine predeclared conditioning controls are retained: three improve, three worsen, and three are unchanged. For scalar values `[1,10^-6]`, original/scaled condition numbers are `1000002000001` and `4`; reversing the values gives `1000002000001/1000000000000` and `4`, so scaling is worse. Exact unit values give `4` on both sides. Tiny-value controls keep the original `1e-18` floor. No unfavorable control was removed.

State nonexpansiveness does not imply small input gradients. The largest partial in the finite derivative grid was `20000000003/500` (about 4.0e7). A separate hand-derived one-step control has `dS/dv=-50,000,000+9/400` while `dS/dS0=0.49875`; all four implementations agree exactly. This does not diagnose the saved ordinary-initialization FP32 failure or bound full model gradients.

Primary sources support the factorization and explain its limits. The direct implementation already masks invalid log exponents before exp, and the centered guard bounds specific unmasked products. It does not guarantee accurate cumulative sums or all backward paths. The primary-source note proposes a distinct two-token rounded-prefix canary; it is explicitly unexecuted in this result.

## Deployment measurements

Unmeasured: FP32/FP64 tensor kernels, runtime/model memory, prefill/decode latency, throughput, language quality and scaling. The diagonal change keeps the same quadratic kernels/solve and adds O(T*V) scaling plus T reciprocal square roots. It has no exact conditioning gain for unit-norm values. No speed or deployed-memory gain is inferred from the exact algebra or verification time.

## Decision

The exact finite-domain criterion is met, and the derivation supplies a reviewable numerical contract/oracle for future E01 work. It does not qualify a fully FP32 backend, fix the historical gradient failure, or justify training/promotion.

The next decisive check is the primary-source note's tiny cumulative-log canary, with source sequential/direct/factored controls and actual dtype reporting. An interval-product representation is a separate candidate for that failure mode; diagonal scaling is a separately controlled factor. Any such tensor/model mechanics experiment requires an exact preregistered budget and donor approval, including local dependency/storage authorization. No waiting time or broad donation budget is treated as approval.

After a passing small numerical canary, the known k16/25101/B2/T19 synthetic qualification row is the next control, preserving the original output/state/gradient/complete-optimizer-step/streaming/reset/causality tolerances. A one-row pass would remain a one-row mechanics result. Broader fidelity/performance qualification and all language training remain separate decisions.

## Integrity and cleanup

All output is new: frozen protocol, input binding, independent exact code/fixtures/verifier/tests, finite result, derivation/report, primary-source note and completion receipts. Archived source, failed numerical receipts, previous reports, prior registry entries and accepted-result status are unchanged. Only additive registration and manifest/credit bookkeeping accompany the package. No cloud resources or unrelated artifacts/workloads were created or removed.

This is author-checked work, substantially AI-assisted by Codex for @thakrarsagar, with source reading by a background research agent. Original recurrence/adaptation/research remains credited to @JRowe47's PR #15, Rohan Arun, preserved project contributors and NX-AI/KronLSTM. Equilibration is established numerical practice; no new model, general novelty, SOTA or language-quality claim is made.
