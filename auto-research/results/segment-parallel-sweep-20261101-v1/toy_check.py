"""Pure-Python check of the segment-parallel sweep's indexing (not the project's code, not decisive).

Tiny width, segment length 4, document every 2 segments, so every boundary rule is exercised.
Reference = a transcription of HierarchicalDistributionStudent.transition with GRUCell equations.
"""
import math
import random

W, L, DN, N, V = 3, 4, 2, 6, 11
T = N * L
rng = random.Random(7)


def mat(r, c, s):
    return [[rng.uniform(-s, s) for _ in range(c)] for _ in range(r)]


def vec(n, s):
    return [rng.uniform(-s, s) for _ in range(n)]


def gru_params(inp, s=1.2):
    return dict(wih=mat(3 * W, inp, s), whh=mat(3 * W, W, s), bih=vec(3 * W, s), bhh=vec(3 * W, s))


FAST, SEG, DOC = gru_params(3 * W), gru_params(W), gru_params(W)
COMB = mat(W, 3 * W, 0.7)
EMB = [vec(W, 1.0) for _ in range(V)]
sig = lambda x: 1 / (1 + math.exp(-x))


def mv(m, v):
    return [sum(a * b for a, b in zip(row, v)) for row in m]


def gru(p, x, h):
    gi = [a + b for a, b in zip(mv(p["wih"], x), p["bih"])]
    gh = [a + b for a, b in zip(mv(p["whh"], h), p["bhh"])]
    r = [sig(gi[i] + gh[i]) for i in range(W)]
    z = [sig(gi[W + i] + gh[W + i]) for i in range(W)]
    n = [math.tanh(gi[2 * W + i] + r[i] * gh[2 * W + i]) for i in range(W)]
    return [(h[i] - n[i]) * z[i] + n[i] for i in range(W)]


def combine(f, s, d):
    return [math.tanh(v) for v in mv(COMB, f + s + d)]


def reference(ids):
    fast, seg, doc, ssum, st, ns = [0.0] * W, [0.0] * W, [0.0] * W, [0.0] * W, 0, 0
    hidden, fasts = [], []
    for t in ids:
        fast = gru(FAST, EMB[t] + seg + doc, fast)
        ssum = [a + b for a, b in zip(ssum, fast)]
        st += 1
        if st == L:
            seg = gru(SEG, [v / st for v in ssum], seg)
            ssum, st, ns = [0.0] * W, 0, ns + 1
            if ns == DN:
                doc = gru(DOC, seg, doc)
                ns = 0
        hidden.append(combine(fast, seg, doc))
        fasts.append(fast)
    return hidden, fasts


def sweeps(ids, max_k, broken=False):
    z = [0.0] * W
    Hin = [z] * N
    S = [z] * (N + 1)
    D = [z] * (N // DN + 1)
    used_doc = (N - 1) // DN + 1
    log = []
    for k in range(1, max_k + 1):
        out, last, mean = [], [], []
        for j in range(N):
            h, acc, seg_out = Hin[j], [0.0] * W, []
            for p in range(L):
                h = gru(FAST, EMB[ids[j * L + p]] + S[j] + D[j // DN], h)
                acc = [a + b for a, b in zip(acc, h)]
                seg_out.append(h)
            out.append(seg_out)
            last.append(h)
            mean.append([v / L for v in acc])
        Hn = [z] + last[:-1]
        Sn = [z]
        for i in range(N):
            Sn.append(gru(SEG, mean[i], Sn[-1]))
        Dn = [z]
        for m in range(1, N // DN + 1):
            Dn.append(gru(DOC, Sn[m * DN], Dn[-1]))
        res = max(max(abs(a - b) for x, y in zip(Hn, Hin) for a, b in zip(x, y)),
                  max(abs(a - b) for x, y in zip(Sn[:N], S[:N]) for a, b in zip(x, y)),
                  max(abs(a - b) for x, y in zip(Dn[:used_doc], D[:used_doc]) for a, b in zip(x, y)))
        hidden = []
        for j in range(N):
            for p in range(L):
                lastp = p == L - 1 and not broken
                s = Sn[j + 1] if lastp else Sn[j]
                d = Dn[(j + 1) // DN] if lastp else Dn[j // DN]
                hidden.append(combine(out[j][p], s, d))
        log.append((k, res, hidden, out))
        Hin, S, D = Hn, Sn, Dn
    return log


ids = [rng.randrange(V) for _ in range(T)]
ref_h, ref_f = reference(ids)
ok = True
for k, res, hidden, out in sweeps(ids, N + 1):
    seg_err = [max(abs(a - b) for p in range(L) for a, b in zip(out[j][p], ref_f[j * L + p])) for j in range(N)]
    wave = max(seg_err[:min(k, N)])
    herr = max(abs(a - b) for x, y in zip(hidden, ref_h) for a, b in zip(x, y))
    print(f"sweep {k}: residual {res:.2e}  hidden_err {herr:.2e}  wavefront_err {wave:.2e}")
    ok &= wave < 1e-12
final_err = max(abs(a - b) for x, y in zip(sweeps(ids, N + 1)[-1][2], ref_h) for a, b in zip(x, y))
broken_err = max(abs(a - b) for x, y in zip(sweeps(ids, N + 1, broken=True)[-1][2], ref_h) for a, b in zip(x, y))
print("wavefront ok:", ok, "| final hidden err:", f"{final_err:.2e}", "| broken-control err:", f"{broken_err:.2e}")
print("TOY_PASS" if ok and final_err < 1e-12 and broken_err > 1e-6 else "TOY_FAIL")
