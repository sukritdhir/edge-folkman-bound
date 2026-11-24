"""Weighted spectral criterion (one weight per triangle orbit, one diagonal entry per edge orbit) for a Cayley graph on
the additive group of a product ring R (exp/ring_dfs.py), connection set S = union of orbits of a subgroup M of R^*.
usage: python3 exp/ring_weighted.py spec |M| rep1,rep2,...     (reps = element indices of orbit representatives)
       python3 exp/ring_weighted.py F729 28 exp:14,20,22,23     (exp: = powers of the primitive element of a single field)
Prints the unweighted ratio and the optimised ratio lambda_min(D^-1/2 M_mu D^-1/2) / (-2/3)   (below 1 = arrows).
"""
import numpy as np, sys
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import cvxpy as cp
from ring_dfs import Ring
from gfq_dfs import Field


def build(R, M, reps):
    n = R.n; mul = R.mul; neg = R.neg; sub = R.sub; Ml = np.array(sorted(M))
    inS = np.zeros(n, dtype=bool)
    for r in reps: inS[mul[Ml, r]] = True
    S = np.nonzero(inS)[0]; assert inS[neg[S]].all() and not (S == neg[S]).any()
    Cp = S[S < neg[S]]; m = len(Cp)
    idx = np.full(n, -1, dtype=np.int64); idx[Cp] = np.arange(m); idx[neg[Cp]] = np.arange(m)
    isrep = np.zeros(n, dtype=bool); isrep[Cp] = True
    pm = np.minimum(np.arange(n), neg)                                   # canonical element of {x, -x}
    # edge orbits: M-orbits of the classes
    eo = {}; eorb = np.zeros(m, dtype=np.int64)
    for j, c in enumerate(Cp):
        k = int(pm[mul[Ml, c]].min()); eorb[j] = eo.setdefault(k, len(eo))
    T = inS[sub[S[None, :], Cp[:, None]]]
    deg = T.sum(axis=1).astype(float)
    jj, vv = np.nonzero(T); v = S[vv]; c = Cp[jj]; w = sub[v, c]
    # triangle orbit: smallest sorted triple of canonical classes over M
    tri = np.sort(np.stack([pm[mul[Ml[:, None], c[None, :]]], pm[mul[Ml[:, None], v[None, :]]], pm[mul[Ml[:, None], w[None, :]]]], axis=2), axis=2)   # (|M|, N, 3)
    code = (tri[:, :, 0] * n + tri[:, :, 1]) * n + tri[:, :, 2]
    key = code.min(axis=0)
    uk, to = np.unique(key, return_inverse=True)
    j2a = idx[v]; x2a = np.where(isrep[v], 0, v)
    j2b = idx[w]; x2b = np.where(isrep[w], c, v)
    ent = np.concatenate([np.stack([jj, j2a, x2a, to], axis=1), np.stack([jj, j2b, x2b, to], axis=1)])
    freqs = [o[0] for o in R.orbits(M, zero=True)]
    return S, Cp, ent, deg, eorb, len(eo), len(uk), np.stack([jj, to], axis=1), freqs


def blocks(R, m, ent, T, f):
    B = np.zeros((T, m, m), dtype=complex)
    np.add.at(B, (ent[:, 3], ent[:, 0], ent[:, 1]), R.char[f, ent[:, 2]])
    return B


def pencil(R, m, ent, T, freqs, mu, Dj):
    dm = 1 / np.sqrt(Dj); lam = np.inf
    for f in freqs:
        Mx = np.tensordot(mu, blocks(R, m, ent, T, f), axes=1) * dm[:, None] * dm[None, :]
        lam = min(lam, np.linalg.eigvalsh((Mx + Mx.conj().T) / 2)[0])
    return lam


def solve(R, M, reps, verbose=True, eps=1e-7):
    S, Cp, ent, deg, eorb, nE, T, tri_of, freqs = build(R, M, reps)
    m = len(Cp)
    if verbose: print("R %s n %d  |M| %d  d %d  edge classes %d  edge orbits %d  triangle orbits %d  tri/edge %d..%d  frequencies %d" % (R.name, R.n, len(M), len(S), m, nE, T, deg.min(), deg.max(), len(freqs)))
    # Deg(mu)_j = sum of mu over the triangles on the class j
    lam0 = pencil(R, m, ent, T, freqs, np.ones(T), deg)
    if verbose: print("unweighted ratio %.4f" % (lam0 / (-2 / 3))); sys.stdout.flush()
    mu = cp.Variable(T, nonneg=True); Dv = cp.Variable(nE, nonneg=True); t = cp.Variable()
    cnt = np.zeros(T); np.add.at(cnt, tri_of[:, 1], 1.0)
    ecount = np.bincount(eorb, minlength=nE).astype(float)
    cons = [cnt @ mu == deg.sum(), ecount @ Dv == cnt @ mu]
    P = np.zeros((m, nE)); P[np.arange(m), eorb] = 1.0
    for f in freqs:
        B = blocks(R, m, ent, T, f)
        Emb = [np.block([[B[o].real, -B[o].imag], [B[o].imag, B[o].real]]) for o in range(T)]
        Mx = sum(mu[o] * Emb[o] for o in range(T))
        dd = P @ Dv
        cons.append(Mx + (2 / 3) * cp.diag(cp.hstack([dd, dd])) - t * np.diag(np.concatenate([deg, deg])) >> 0)
    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver="SCS", eps=eps, max_iters=400000)
    muv = np.maximum(np.array(mu.value), 0); Dval = np.maximum(np.array(Dv.value), 1e-12)
    Dj = Dval[eorb]; Dj = Dj * ((cnt @ muv) / Dj.sum())
    lam = pencil(R, m, ent, T, freqs, muv, Dj)
    if verbose:
        print("SDP value t = %.6f; optimised pencil lambda_min = %.6f; weighted ratio %.5f  %s" % (t.value, lam, lam / (-2 / 3), "ARROWS" if lam > -2 / 3 else ""))
        print("mu:", np.round(muv, 4)); print("triangles per orbit (incidences):", cnt.astype(int))
        print("D per edge orbit:", np.round(Dval, 4), " (unweighted deg per orbit:", [float(deg[eorb == i][0]) for i in range(nE)], ")")
    return lam0 / (-2 / 3), lam / (-2 / 3), muv, Dval


if __name__ == "__main__":
    spec = [(s[0], int(s[1:])) for s in sys.argv[1].split("x")]; R = Ring(spec); mord = int(sys.argv[2])
    a = sys.argv[3]
    if a.startswith("exp:"):
        F = Field(spec[0][1]); reps = [int(F.exp[int(x)]) for x in a[4:].split(",")]
    else: reps = [int(x) for x in a.split(",")]
    Ms = [M for M in R.subgroups() if len(M) == mord]
    print("subgroups of order %d: %d" % (mord, len(Ms)))
    for M in Ms:
        try: solve(R, M, reps)
        except AssertionError: print("  (S not symmetric or contains an involution for this subgroup)")
