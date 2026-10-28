# minKron: primary-source numerical assessment

2026-10-23. Base: `847bb75042e01b2c879272f97e72746f285c8872`. AI-assisted research for session `dcbb91d07e51cf13ee6fda1fefc14ca2`. **Read-only research and an unexecuted mechanics proposal; no new PyTorch/model execution, training, evaluation, installation, or corpus/checkpoint download.** “Fact” below means source-backed; “Inference” means a derivation from the cited equations; “Hypothesis” means a prediction requiring execution. Repository evidence verification passed: 2,812 file hashes and the verifier's saved-receipt checks; this is not a numerical replay.

The centered factorization preserves the real-arithmetic recurrence. Its guard protects particular exponential products, not all forward/backward numerics. The state nonexpansiveness hypothesis is valid with fixed inputs and gates, but does not establish full-model gradient fidelity. The archived fully FP32 qualification failed; the later pass used an FP64 core. A two-token cumulative-sum precision canary below can test a distinct limitation without training. [Local core][L-core], [qualification addendum 02][L-a2].

## Recurrence, domains, and lineage

**Fact.** For one head, set $D_t=\operatorname{diag}(e^{g_t})$, $r_t=\|v_t\|_2^2$, $\bar r_t=\max(r_t,10^{-18})$, and $c_t=\texttt{write}_t$. With a reset applied **before** the token, the local implementation is

$$
\widetilde S_t=D_tS_{t-1},\qquad
y_t=k_t^\top\widetilde S_tv_t,\qquad
u_t=c_t r_t/\bar r_t-\beta_t y_t/\bar r_t,
$$
$$
S_t=\widetilde S_t+u_tk_tv_t^\top,\qquad
o_t=K^{-1/2}q_t^\top S_t.
$$

This includes the zero/tiny-value safeguard; replacing $u_t$ by $c_t-\beta_ty_t$ is valid only for unit-norm values. [Local `kron_recurrent`, lines 51–69][L-core].

**Fact and domain inference.** `MinKron` applies SiLU and `F.normalize` to q/k/v, sigmoid to erase/write, and $g=-e^{A_{\log}}\operatorname{softplus}(f+dt\_bias)$. For finite real inputs these imply q/k/v norms at most one, $g\leq0$, and sigmoid gates in $(0,1)$; the useful closed domain is $[0,1]$. `F.normalize` divides by `max(norm, 1e-12)`, so zero/tiny vectors need not have unit norm. Raw core functions do not enforce these input domains. [Wrapper, lines 222–249][L-wrapper], [PyTorch normalization][P-normalize].

**Fact.** The vendored `naive.py` and `layers/kronlstm.py` are byte-identical to NX-AI revision `3d800c9a153a88ee9624fe6acf7f39a58017a17e`: their Git blob hashes are respectively `4280290762f798579af5cae2bb51b640fa0563dc` and `9f4b9b68ff242cf847c5e78857a37264db50cb55`, matching the local manifest. The local recurrence corresponds to upstream `decouple_erase=True`, `allow_neg_eigval=False` ($s=1$), with normalized values and target gate `c`; reset masking is a local extension. Upstream also offers coupled writes and $s=2$; those are different variants. The paper uses the transposed state convention. Vendored MIT notices credit Songlin Yang, Yu Zhang, Zhiyuan Li, and NXAI GmbH. [Local provenance][L-provenance], [upstream recurrence][U-naive], [upstream layer][U-layer], [paper §2.2][Paper-architecture], [notice][L-license].

## Exact factorization and its numerical limits

**Inference.** Define $G_i=\sum_{t\leq i}g_t$ coordinatewise. For positions in the same document, let $R_{ij}=\operatorname{diag}(e^{G_i-G_j})$. Expanding the recurrence gives the unit lower-triangular system

$$
M_{ij}=\delta_{ij}+\mathbf1_{j<i,\,\mathrm{same\ doc}}
\frac{\beta_i}{\bar r_i}(k_i^\top R_{ij}k_j)(v_i^\top v_j),
$$
$$
b_i=c_i r_i/\bar r_i-(\beta_i/\bar r_i)
k_i^\top\operatorname{diag}(e^{G_i})S_0v_i\,
\mathbf1_{\mathrm{before\ first\ reset}},\qquad Mu=b.
$$

Both chunk functions implement this system and its state/output reconstruction. It is nonsingular in exact arithmetic because its diagonal is one. The upstream chunk derivation is the same Kronecker-feature construction. [Local core, lines 96–117 and 145–163][L-core], [paper Appendix A8.2–A8.4][Paper-chunk].

For any common finite center $h$,

$$
\sum_\ell k_{i\ell}k_{j\ell}e^{G_{i\ell}-G_{j\ell}}
=\left(k_i\odot e^{G_i-h}\right)^\top
 \left(k_j\odot e^{-G_j+h}\right).
$$

**Inference.** Centering therefore changes evaluation order, not the mathematical system or its condition number. Detaching the common center does not remove a required exact derivative: dependence on that center cancels identically. Floating-point cancellation and branch-boundary behavior still require checks. [Local factorization, lines 121–164][L-core].

**Fact and inference.** The direct implementation masks invalid log differences **before** `exp` at line 101; an allegation of upper-triangle masked-exp overflow there is incorrect. With finite monotone cumulative gates, valid causal exponents are nonpositive. The factored implementation checks $|G-h|\leq40$ and otherwise calls the direct implementation. Under the normalized q/k domain, each scaled operand is bounded by $e^{40}$; Cauchy–Schwarz bounds the absolute sum in each unmasked Gram entry by $e^{80}\approx5.54\times10^{34}$, below binary32's maximum. This supports the guard's forward-product rationale, including the upper triangle. It does **not** bound arbitrary cotangents, carried-state magnitudes, errors in cumulative sums, or every later operation. An already-rounded cumulative sum cannot be repaired by centering it afterward. [Local core, lines 99–102 and 137–159][L-core], [PyTorch numerical accuracy][P-numerics].

## State contraction is useful, but is not a full stability certificate

**Inference, valid also below the value-norm floor.** Let $a_t=\operatorname{vec}(k_tv_t^\top)$, and let $\mathcal D_t$ represent row decay on the vectorized state. For fixed token inputs/gates the state Jacobian is

$$
J_t=\left(I-\frac{\beta_t}{\bar r_t}a_ta_t^\top\right)\mathcal D_t.
$$

The rank-one factor has eigenvalues one and

$$
1-\beta_t\|k_t\|^2r_t/\bar r_t\ \in[0,1].
$$

Since $\|\mathcal D_t\|_2\leq1$, $\|J_t\|_2\leq1$. A reset has zero state Jacobian. Thus differences between states following the **same** input/gate sequence do not grow in Frobenius norm. The drive has norm at most one, giving the weaker absolute bound $\|S_T\|_F\leq\|S_0\|_F+T$, not a context-independent bound. These statements follow from the exact local recurrence, not an observed runtime experiment. [Local recurrence and wrapper][L-core], [domain construction][L-wrapper].

**Further conditional inference.** If every value has norm exactly one, then $|M_{ij}|\leq1$. For a column of $M^{-1}$, take $b=e_j$: the injected state at $j$ has norm at most one, subsequent homogeneous transitions contract, and later write coefficients have magnitude at most one. Therefore $\|M\|_\infty,\|M^{-1}\|_\infty\leq T$, hence $\kappa_\infty(M)\leq T^2$. This crude bound is stronger than treating the matrix as an arbitrary unit-triangular matrix, but does not guarantee elementwise gradient tolerances. It does not extend unchanged to tiny values: with scalar keys one, $v_1=1,v_2=10^{-9},e^{g_2}=\beta_2=1/2$, the exact entry $M_{21}=2.5\times10^8$, although the physical-state Jacobian remains nonexpansive. [System derived above from the local core][L-core].

**Limits.** Parameter/input differentiation also changes q/k/v, gates, normalization, and the driving term. The solve derivative contains $M^{-1}(db-dM\,u)$; its reverse pass solves against $M^\top$ and forms a product with $u$. State contraction alone bounds none of these complete paths, nor cross-layer coupling. [PyTorch solve backward][P-solve-backward]. The paper's expected-memory theorems assume an i.i.d. association stream, fixed step size, and positive-definite feature covariance. Those assumptions are not established for this learned, decoupled-gate language model; do not import the theorem as a universal FP32/autograd guarantee. [Paper §2.1 and Appendix A3.5.2][Paper-assumptions].

## What the archived qualification establishes

- **Fact:** the first failure is an optimizer-step comparison: one element differs by `2.4419103283435106e-05` absolute. Addendum 01 attributes sensitivity to AdamW epsilon and changes epsilon from `1e-8` to `1e-4`, retaining tolerances. The saved diagnostic also contains failures for the direct core; this is not evidence that factorization alone caused the failure. [Initial failure][L-f1], [addendum 01][L-a1], [diagnostic][L-diagnostic].
- **Fact:** the second failure is a parameter-gradient comparison for `k16`, seed `25101`, FP32: absolute difference `3.632158041000366e-05`, relative `0.0012213382869958878`. Addendum 02 moves normalization, decay, recurrence, carried state, and output-head normalization to FP64. The final `dtype=torch.float32` rows label the model's outer dtype; `Config.recurrence_fp64=True` still selects an FP64 core. They do not qualify a fully FP32 recurrence. [Second failure][L-f2], [addendum 02][L-a2], [configuration][L-config], [wrapper][L-wrapper], [final receipt][L-fidelity].
- **Inference:** on AdamW's first step, its gradient-dependent update is proportional to $g/(|g|+\epsilon)$; sensitivity near zero scales as $1/\epsilon$. This makes the archived explanation plausible, but is not independent root-cause proof. Changing epsilon changes optimizer semantics and is not a same-optimizer numerical fix. [Official AdamW algorithm][P-adam].

PyTorch explicitly does not promise identical floating-point results for different evaluation orders/batching; FP64 may help without guaranteeing correctness. `unitriangular=True` declares a unit diagonal, not a conditioning certificate. Autograd differentiates recorded operations, so masking invalid results afterward is not generally a repair; this general warning is **not** a finding against the already-masked direct exponent. [Accuracy guidance][P-numerics], [triangular-solve docs][P-solve], [autograd guidance][P-autograd]. Documentation/source links here are pinned to build commit `5811a8d7da873dd699ff6687092c225caffcf1bb`, recorded by the archived environment reporting `2.9.1+cpu`. [Environment][L-env].

## Cheapest decisive next checks — proposals, not executed

1. **Two-token cumulative-sum canary.** Use `B=H=K=V=1`, `T=2`, q=k=v=1, beta=write=1/2, zero initial state, no resets, and FP32 gates `g=[-2**27, -1]`. In real arithmetic the recurrent outputs/final state are $o_1=1/2$, $o_2=S_2=1/2+e^{-1}/4\approx0.59197$. **Hypothesis:** binary32 round-to-nearest stores both cumulative gates as `-2**27`, losing the second decay; both chunk paths then yield `0.75`, while the centered guard sees zero range and does not fall back. This follows from binary32 spacing at that magnitude and the source's `cumsum`/subtraction, but has not been run in PyTorch. It is an extreme finite-domain test, not a diagnosis of the saved initialized-model failure. Record the three cores' outputs/state, cumulative gates, center, branch, and actual dtypes; repeat with identical inputs promoted to FP64. An FP64 control should retain the `-1` increment. [Cumulative-sum definition][P-cumsum], [floating-point guidance][P-numerics], [local implementations][L-core].
2. **Small core checks, then the known failing model row.** A Fraction oracle using rational per-coordinate decays and interval products is a useful independent exact check of recurrence versus triangular algebra, including resets and the norm floor. Finite fixtures do not prove all cases, and such an oracle does not test `exp`, cumsum, autograd, or FP32. Zero decay needs interval products, not division by a zero cumulative product. Separately exercise nonzero incoming state, tiny/zero values, irregular splits, and centered ranges exactly 40 and just above it. For the historical gradient question, use the saved `k16/25101`, `B=2,T=19` synthetic row, preserving initialization/reset construction and explicitly selecting `recurrence_fp64=False`; compare recurrent/direct/factored paths and an FP64 diagnostic reference. Include logits, every state, loss, **all** parameter gradients, and one identical AdamW step at the declared epsilon. Preserve the frozen FP32 `atol=2e-5, rtol=2e-4`; supplemental normwise errors cannot replace this gate. A passing single row remains a canary, not broad qualification. [Saved harness][L-qualify], [configuration construction][L-common], [frozen tolerances][L-protocol].
3. **Choose a remedy only after attribution.** Smaller chunks or computing/rebasing interval log sums before precision is lost are same-real-math candidates; changing the center or lowering 40 afterward cannot restore lost increments. Upstream uses boundary-referenced factors and distinguishes IEEE FP32 dot products from TF32; this is an implementation precedent, not qualification of local CPU code. Selective FP64 prefix/solve arithmetic is another diagnostic or implementation option, but must be described as mixed precision. Use FP64 gradcheck away from floor/branch boundaries; its defaults are designed for double precision. No option has been measured here. [Upstream kernel design/numerics][U-kernels], [PyTorch gradcheck][P-gradcheck], [local core][L-core].

Next action: preserve these as E01 algebra/mechanics hypotheses. Once a separately authorized compatible runtime exists, run the two-token canary first, then the known failing model row. Do not infer speed, quality, global gradient stability, or historical root cause from this note. No archived evidence was edited.

[L-core]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L51-L164
[L-wrapper]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L183-L255
[L-config]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L15-L36
[L-provenance]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-publication-20261021-v1/compact/source_manifest.json#L13-L35
[L-license]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-publication-20261021-v1/compact/upstream/KronLSTM_LICENSE.txt#L1-L14
[L-f1]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/qualification_initial_failed.log#L4-L19
[L-f2]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/qualification_epsilon_failed.log#L16-L34
[L-a1]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/PROTOCOL_ADDENDUM_01.json#L1-L10
[L-a2]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/PROTOCOL_ADDENDUM_02.json#L1-L9
[L-diagnostic]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/FAILURE_DIAGNOSTIC.json#L1-L58
[L-fidelity]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/FIDELITY.json#L137-L168
[L-env]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/ENVIRONMENT.json#L1-L12
[L-qualify]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/qualify.py#L26-L69
[L-common]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/common.py#L12-L31
[L-protocol]: https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/PROTOCOL.json#L87-L97
[U-naive]: https://github.com/NX-AI/KronLSTM/blob/3d800c9a153a88ee9624fe6acf7f39a58017a17e/fla/ops/kronlstm/naive.py#L47-L143
[U-layer]: https://github.com/NX-AI/KronLSTM/blob/3d800c9a153a88ee9624fe6acf7f39a58017a17e/fla/layers/kronlstm.py#L308-L349
[U-kernels]: https://github.com/NX-AI/KronLSTM/blob/3d800c9a153a88ee9624fe6acf7f39a58017a17e/fla/ops/kronlstm/README.md#L140-L174
[Paper-architecture]: https://arxiv.org/html/2610.26834v1#S2.SS2
[Paper-chunk]: https://arxiv.org/html/2610.26834v1#A8.SS2
[Paper-assumptions]: https://arxiv.org/html/2610.26834v1#A3.SS5.SSS2
[P-normalize]: https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/nn/functional.py#L5541-L5575
[P-numerics]: https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/docs/source/notes/numerical_accuracy.rst#L6-L97
[P-solve]: https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/linalg/__init__.py#L2323-L2376
[P-solve-backward]: https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/csrc/autograd/FunctionsManual.cpp#L4530-L4567
[P-autograd]: https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/docs/source/notes/autograd.rst#L92-L151
[P-gradcheck]: https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/autograd/gradcheck.py#L1988-L2003
[P-adam]: https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/optim/adamw.py#L61-L108
[P-cumsum]: https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/_torch_docs.py#L3383-L3413
