# Exact core identities and a scaled-system contract

These are author-derived algebraic statements about the frozen local recurrence. The finite Fraction checks are separate evidence; they are not a proof over every input or a machine-checked proof. No independent mathematical review or floating-point qualification is claimed.

Source: [model.py at the study base](https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L51). The [primary-source note](../../research-scratch/kron-primary-source-notes-20261023.md) traces the original Kronecker-feature recurrence and distinguishes its assumptions from this learned-gate adaptation.

## 1. Keep the actual floor and reset semantics

Work on one batch element/head; different elements/heads are independent at this boundary. Let k and q have K coordinates and v have V coordinates. S is K by V. D_t is diagonal row decay. A reset zeros the incoming state before token t. Define:

```text
n_t = v_t · v_t
m_t = max(n_t, ε), ε = 10^-18
H_t = D_t S_(t-1)             (or zero after a before-token reset)
u_t = w_t n_t/m_t - (β_t/m_t) k_t^T H_t v_t
S_t = H_t + u_t k_t v_t^T
o_t = q_t^T S_t / sqrt(K)
```

Do not replace w_t n_t/m_t by w_t near zero values: that changes the preserved equations. The wrapper normalizes q/k/v with an epsilon, so their norms are at most one rather than identically one in every input regime. Sigmoid gates give 0≤β_t,w_t≤1 in the closed limiting domain, and nonpositive g gives diagonal decay in [0,1]. The source constructs these domains at [lines 238–243](https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L238).

## 2. Expand to the original triangular system

Let P_(t,s) be the product D_t ... D_(s+1) when s≤t and there is no reset between s and t; otherwise it is zero. P_(t,t) is identity. Let O_t be D_t ... D_1 if no reset has occurred, otherwise zero. Then:

```text
S_t = O_t S_0 + sum_(s≤t) u_s P_(t,s) k_s v_s^T
A_tt = 1
A_ts = (β_t/m_t) [k_t^T P_(t,s) k_s] [v_t^T v_s],  s<t
b_t = w_t n_t/m_t - (β_t/m_t) k_t^T O_t S_0 v_t
A u = b
```

This follows by substituting the expanded state into the current score and moving previous updates to the left. A is unit lower triangular and hence nonsingular in exact arithmetic. That fact alone is not a conditioning guarantee. The direct source implements this system and reconstructs output/final state at [lines 96–117](https://github.com/rohanarun/Super-1/blob/847bb75042e01b2c879272f97e72746f285c8872/auto-research/results/minkron-fast-20261021-v1/model.py#L96).

For positive decays, each coordinate of P can be written as a prefix-product ratio. For any positive common center c in that coordinate:

```text
k_t[i] P_(t,s)[i] k_s[i]
  = (k_t[i] prefix_t[i]/c[i]) (k_s[i] c[i]/prefix_s[i])
```

The center cancels, including in exact derivatives if it is treated as detached. The source's log/exp version uses the same identity. The rational factored oracle chooses an arbitrary positive rational center so it can check the identity without exp/log. It does not reproduce floating-point centering errors, and it excludes zero decays because prefix division would be undefined. Interval products and the sequential recurrence do not require that division.

## 3. Change coordinates, not the recurrence

Set r_t=sqrt(m_t), z_t=v_t/r_t, and α_t=r_t u_t. Since m_t>0, this is invertible even for v_t=0. Substitution gives:

```text
S_t = H_t + α_t k_t z_t^T
α_t = w_t n_t/r_t - β_t k_t^T H_t z_t
B_tt = 1
B_ts = β_t [k_t^T P_(t,s) k_s] [z_t^T z_s],  s<t
c_t = w_t n_t/r_t - β_t k_t^T O_t S_0 z_t
B α = c
B = R A R^-1,  c = R b,  α = R u,  R = diag(r_t)
```

Outputs and final state are unchanged because α_t z_t = u_t v_t. The floor, gates, decay, resets and model parameters are unchanged. Where derivatives exist, the same identities hold under differentiation, including the dependence of r on v. The derivative fixtures avoid the floor kink; this study makes no new claim about a floating implementation's branch/subgradient behavior there.

## 4. Fixed-input state nonexpansiveness

The scaled value satisfies ||z_t||²=n_t/m_t≤1. Let a_t=vec(k_t z_t^T). For fixed q/k/v/decay/gates, the homogeneous state transition has the form:

```text
J_t = (I - β_t a_t a_t^T) D_t
```

Here D_t denotes the corresponding row-decay operator on vectorized matrices. Since ||a_t||²≤1 and 0≤β_t≤1, the symmetric rank-one factor has eigenvalues one and 1-β_t||a_t||² in [0,1]. Its Euclidean operator norm is at most one; the diagonal decay has the same bound. Their product is nonexpansive. A reset has zero state Jacobian.

Thus two states following the same input/gate sequence cannot move farther apart in Frobenius norm. The affine write's norm is at most one, giving ||S_T||_F≤||S_0||_F+T. This is not a context-independent absolute state bound and does not constrain gradients through changing inputs/gates or upstream normalization.

## 5. Entry and infinity-condition bounds for B

For every strict-lower entry, Cauchy–Schwarz and the nonexpansive interval decay give:

```text
|B_ts| ≤ β_t ||k_t|| ||P_(t,s) k_s|| ||z_t|| ||z_s|| ≤ 1
```

To bound a column of B^-1, solve B α=e_j with zero initial state and a unit drive only at j. Before j the state is zero. At j, α_j=1 and the injected state k_j z_j^T has Frobenius norm at most one. Thereafter the state follows the homogeneous nonexpansive transition above. Its later coefficients obey:

```text
|α_t| = |β_t k_t^T H_t z_t| ≤ 1, t>j
```

Resets only remove the impulse. Therefore every entry of B^-1 is bounded by one. A row has at most T entries, so:

```text
||B||_∞ ≤ T,  ||B^-1||_∞ ≤ T,  κ_∞(B) ≤ T².
```

This uses the structured recurrence and normalized domains; it is not a property of arbitrary unit-triangular matrices. It is not a bound on a full model's parameter Jacobian, backward cotangents, exp/log rounding, or the source's gate-prefix accumulation. The finite verifier checks these entry/condition bounds on its declared 3,840 inputs, separately from this argument.

## 6. Preserve the unfavorable and null controls

Diagonal scaling can improve or worsen an individual matrix's ordinary condition number. LAPACK's [DGEEQU documentation](https://netlib.org/lapack/explore-html/d5/d8a/group__geequ_ga7238812c5c26fc2fbe8bbbc22b6ee885.html) likewise does not guarantee improvement from equilibration. This particular coordinate choice is analytically useful because it extends the structured bounds to tiny values; it is not a universal conditioning win.

For T=2, scalar keys/decays/β equal one, no resets:

| v values | κ∞(A) | κ∞(B) | Outcome |
|---|---:|---:|---|
| [1, 10^-6] | 1,000,002,000,001 | 4 | better |
| [10^-6, 1] | 1.000002000001 | 4 | worse |
| [1, 1] | 4 | 4 | unchanged |
| [1, 10^-10] | 10,000,000,200,000,001 | 1.21 | better with the unchanged floor |
| [10^-10, 1] | 1.00000000020000000001 | 1.21 | worse |

All nine predeclared controls, including three repeated unit controls, are retained exactly in EXACT_RESULTS.json. When all value norms are exactly one, R=I and there is no exact condition-number gain. This does not explain the archived ordinary-initialization FP32 gradient failure.

## 7. A bounded state can have a large input derivative

Take one scalar step, q=k=1, D=1/2, β=1/4, w=3/4, S_0=2, and v=10^-10. The unchanged floor is active. Direct substitution gives:

```text
S_1 = 399/400 + 3/(4*10^12)
dS_1/dv = -50,000,000 + 9/400 = -49,999,999.9775
dS_1/dS_0 = 0.49875
```

The exact dual-number hand control checks this derivative in all four representations. State nonexpansiveness and large sensitivity to an input therefore coexist. Upstream normalization and full model gradients remain further obligations.

## 8. Price the proposal and state its limits

The scaling leaves the quadratic pair kernels and triangular solve in place. It adds O(T*V) value scaling and T reciprocal square roots; it does not remove the original recurrence's cost. It can be composed algebraically with direct or factored pair kernels, but no floating implementation, latency, memory or quality improvement was measured here.

The exact oracle's finite checks omit the common output factor 1/sqrt(K), use rational vectors with rational norm roots, and do not evaluate exp/log, tensor-library accumulation, autograd kernels, or trained models. A separate proposed tensor canary must first test those numerical boundaries, including the primary-source note's two-token cumulative-log hypothesis. Matching real-arithmetic identities cannot restore a decay already lost by rounded prefix subtraction. [PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) describes why equivalent evaluation orders need not agree numerically.
