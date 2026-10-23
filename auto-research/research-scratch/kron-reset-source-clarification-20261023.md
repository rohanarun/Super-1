# Reset semantics and adjoint scope: source clarification

2026-10-23. Source reading and author-checked AI algebra only, not independent mathematical verification or a numerical replay. No tensor/model execution, tests, training, downloads, or archived-source edits. The completed [primary-source note][N-prior] was also read; the findings below rest on the cited implementation and derivation.

## Supported reset finding

**No concrete reset implementation defect is established by the cited sources.** The frozen adaptation applies a reset before token processing: it masks the incoming state, applies decay, then writes the current token ([model.py, lines 58–68][M-recurrent]). The exact sequential oracle uses the same ordering ([exact_core.py, lines 84–98][E-sequential]).

The chunk implementations use cumulative document IDs, `docs[t] = sum(reset[u], u <= t)`, and causal document-ID equality. For Boolean resets and `s <= t`, `docs[t]-docs[s] = sum(reset[u], s < u <= t)`, so equality means **no reset in `(s,t]`**. A before-token reset **at `s`** clears older state but permits the update written at `s` to influence later tokens in its document. The direct implementation uses this mask at lines 96–102; the factored version at lines 145–149 ([direct][M-direct], [factored][M-factored]).

The exact oracle separately constructs the cumulative IDs and the interval decay product. It first rejects pairs with unequal IDs, then multiplies decays over `s+1,...,t`; this matches the source's mask and the real-arithmetic identity `exp(G_t-G_s) = product(exp(g_u), u=s+1,...,t)` with `G_t = sum(g_u, u <= t)` ([exact_core.py, lines 112–132][E-pairs]; [model.py, lines 96–104][M-direct]). Document-ID equality is the reset condition; the decay product alone does not encode resets. Positive prefix-product ratios represent the same interval, but the rational factored path requires positive decays; interval products themselves allow zero decays ([exact_core.py, lines 126–132][E-pairs]; [DERIVATION.md, lines 36–43][D-factor]).

Carried initial state has a different mask: `docs[t] == 0`, so **any reset through and including `t`** removes that contribution. Final-state token contributions use `docs[s] == docs[last]`. Both source chunk implementations and the exact oracle implement these distinctions ([model.py, lines 110–117][M-direct], [lines 156–163][M-factored]; [exact_core.py, line 119][E-pairs], [lines 148–161][E-final]).

## Exact wording clarification

In [DERIVATION.md, line 24][D-pairs], “no reset between s and t” should be made endpoint-explicit:

> For `s <= t`, `P_(t,s) = D_t ... D_(s+1)` if no before-token reset occurs at an index in `(s,t]`; otherwise it is zero. `P_(t,t) = I`, including when token `t` resets. `O_t = D_t ... D_1` only if no reset occurs in `[1,t]`; otherwise it is zero.

This removes an ambiguity; it is not evidence that the existing implementation or intended algebra is wrong. Applying a mask that also excludes a reset at `s` would incorrectly discard the new token's write. No implementation correction is recommended from this inspection.

## Precise bound domain

For fixed q/k/v, gates, decays, and Boolean reset locations, assume `||k_t||_2 <= 1`, `m_t = max(||v_t||_2^2, 10^-18)`, `z_t = v_t/sqrt(m_t)`, `0 <= beta_t <= 1`, and diagonal decay in `[0,1]`. Then `||z_t||_2 <= 1` and the exact homogeneous state transition is `J_t = (I - beta_t a_t a_t^T) D_t`, `a_t = vec(k_t z_t^T)`, with `||J_t||_2 <= 1`; here `D_t` acts on the vectorized state, and a reset gives the zero transition. These are wrapper-derived real-arithmetic domains, not constraints enforced by the raw core API ([model.py, lines 238–249][M-wrapper]; [DERIVATION.md, lines 47–71][D-state]).

Consequently, a terminal-state cotangent propagated through these **fixed-input state transitions**, with no intermediate cotangent injections, has Frobenius norm no larger than its terminal norm: `vec(bar_S_(t-1)) = J_t^T vec(bar_S_t)`. For a scalar loss depending also on core readouts, each token injects `q_t bar_o_t^T / sqrt(K)`; the bound adds `sum_t ||q_t||_2 ||bar_o_t||_2 / sqrt(K)` to the direct terminal-state cotangent norm. This does not bound the total independently of loss scale or the number of readouts ([model.py, lines 67–68][M-recurrent]; author-checked transpose/triangle-inequality consequence of [DERIVATION.md, lines 63–71][D-state]).

For the **scaled** fixed system `B alpha = c` of length `T`, the study's impulse argument bounds every entry of `B^-1` by one ([DERIVATION.md, lines 73–93][D-bound]). Thus the solve adjoint `lambda = B^-T bar_alpha` satisfies `||lambda||_p <= T ||bar_alpha||_p` for `p = 1, 2, infinity`: row and column absolute sums of `B^-1` are at most `T`, and `||B^-1||_2 <= sqrt(||B^-1||_1 ||B^-1||_infinity) <= T`. These are author-checked consequences, conditional on the stated exact domain. The same `T` estimate must not be transferred unchanged to unscaled `A`: `A^-1 = R^-1 B^-1 R`, `R = diag(sqrt(m_t))` ([DERIVATION.md, lines 47–56][D-state]).

These bounds scale with the supplied cotangent. They establish neither complete input/parameter-gradient bounds nor floating-point accuracy. When the system changes, `d_alpha = B^-1(d_c - d_B alpha)`; the solve reverse pass has `bar_c = lambda`, `bar_B = -lambda alpha^T` before structural projection. Coordinate scaling and upstream normalization add further derivative paths. Accordingly, line 93's broad “backward cotangents” caveat should distinguish this valid fixed-system adjoint estimate from complete upstream derivatives ([DERIVATION.md, line 59][D-state], [line 93][D-bound], [lines 111–127][D-limits]). “Not established by these bounds” does not mean no other bound could be derived.

## Unresolved items and reviewer check

- The interrupted r3 summary supplies no concrete counterexample or correction. Its phrase “reset-mask correction” cannot establish a defect.
- Existing test code already includes a before-token reset control, nontrivial reset/chunk comparisons, and a deliberately omitted-reset negative control ([test_exact_core.py, lines 27–28][E-tests-reset], [lines 65–71][E-tests-chunks], [lines 108–112][E-tests-negative]); these were read, not rerun. Proposed reviewer check: trace a nonzero incoming state with resets at the first token, at contributing token `s`, and later in `(s,t]`; include repeated resets, a chunk boundary, and the source's single-token branch ([model.py, lines 84–95][M-direct]). Confirm current-token writes survive their own reset while carried state and earlier-document writes do not. Independently review the impulse proof and transpose-norm step before promoting a theorem claim.
- Real-arithmetic agreement of interval products and cumulative-log differences does not establish floating-point agreement. This note does not diagnose the historical gradient failure or consume the parent's canary budget.

[M-recurrent]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L51
[M-direct]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L72
[M-factored]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L121
[M-wrapper]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L238
[E-sequential]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/exact_core.py#L84
[E-pairs]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/exact_core.py#L112
[E-final]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/exact_core.py#L148
[E-tests-reset]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/test_exact_core.py#L27
[E-tests-chunks]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/test_exact_core.py#L65
[E-tests-negative]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/test_exact_core.py#L108
[D-pairs]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/DERIVATION.md#L22
[D-factor]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/DERIVATION.md#L36
[D-state]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/DERIVATION.md#L47
[D-bound]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/DERIVATION.md#L73
[D-limits]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/results/kron-exact-core-20261023-v1/DERIVATION.md#L111
[N-prior]: https://github.com/thakrarsagar/Super-1/blob/f38f0635d968423439fe0a64dcf06aac55512b21/auto-research/research-scratch/kron-primary-source-notes-20261023.md#L7
