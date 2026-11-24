"""Mathon-type graphs over finite commutative rings: Q(R, K), R a product of fields GF(q) and rings Z_m, K a subgroup of
R^* with -1 in K.  Vertices: unimodular vectors of R^2 modulo scalars from K; [x] ~ [y] iff det(x, y) in K.
Degree |R|, |K| triangles on every edge, K4-free iff K is sum-free (K + K misses K).  The graph is vertex-, arc- and
triangle-transitive, so the unweighted ratio is its optimum.
usage: python3 exp/ringq.py lo hi out.log [nmax] [kmin]     all rings with lo <= |R| <= hi (products of at most 3
       components with residue fields of at least 5 elements), all sum-free subgroups K with |K| >= kmin and at most
       nmax vertices;   or  python3 exp/ringq.py spec out.log [nmax] [kmin]   with spec like F5xF17
"""
import sys, time, os, itertools
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np
from ring_dfs import Ring
import gen


def prime_power(q):
    for p in range(2, q + 1):
        if q % p == 0:
            x = q
            while x % p == 0: x //= p
            return (p if x == 1 else None)
    return None


def specs(lo, hi, maxcomp=3):
    """products of fields GF(q), q >= 5, and rings Z_(p^2), p >= 5, with lo <= |R| <= hi, at least 2 components
    or a single Z_(p^2)"""
    comps = []
    for q in range(5, hi + 1):
        p = prime_power(q)
        if p is None: continue
        comps.append(("F", q))
        if q == p * p and p >= 5: comps.append(("Z", q))
    out = []
    def rec(start, cur, size):
        if lo <= size <= hi and (len(cur) >= 2 or (len(cur) == 1 and cur[0][0] == "Z")): out.append(list(cur))
        if len(cur) == maxcomp: return
        for i in range(start, len(comps)):
            if size * comps[i][1] > hi: break
            rec(i, cur + [comps[i]], size * comps[i][1])
    rec(0, [], 1)
    return out


def build(R, K):
    """adjacency matrix of Q(R, K)"""
    n = R.n; mul = R.mul; add = R.add; sub = R.sub; one = R.one
    Kl = np.array(sorted(K)); inK = np.zeros(n, dtype=bool); inK[Kl] = True
    ideal = np.zeros((n, n), dtype=bool)
    ideal[np.repeat(np.arange(n), n), mul.reshape(-1)] = True               # ideal[a] = multiples of a
    reps = []; seen = np.zeros((n, n), dtype=bool)
    for a in range(n):
        ua = sub[one, mul[a]]                                              # the elements 1 - a c
        for b in range(n):
            if seen[a, b]: continue
            if not ideal[b][ua].any(): continue                             # not unimodular
            seen[mul[Kl, a], mul[Kl, b]] = True
            reps.append((a, b))
    Rv = np.array(reps); N = len(Rv)
    D = sub[mul[Rv[:, 0][:, None], Rv[:, 1][None, :]], mul[Rv[:, 1][:, None], Rv[:, 0][None, :]]]
    A = inK[D]
    assert (A == A.T).all() and not A.diagonal().any()
    return A


def run(spec, out, nmax, kmin):
    R = Ring(spec); n = R.n; t0 = time.time()
    U = len(R.units)
    groups = sorted(R.subgroups(), key=lambda M: -len(M))
    inU = np.zeros(n, dtype=bool); inU[R.units] = True
    # number of unimodular vectors
    ideal = np.zeros((n, n), dtype=bool); ideal[np.repeat(np.arange(n), n), R.mul.reshape(-1)] = True
    J2 = 0
    for a in range(n):
        ua = R.sub[R.one, R.mul[a]]
        J2 += int(ideal[:, ua].any(axis=1).sum())
    nsf = 0; best = None; ndone = 0
    for K in groups:
        k = len(K)
        if k < kmin: break
        Kl = np.array(sorted(K)); inK = np.zeros(n, dtype=bool); inK[Kl] = True
        if inK[R.add[np.ix_(Kl, Kl)]].any(): continue                       # not sum-free
        nsf += 1
        nv = J2 // k
        if nv > nmax: continue
        A = build(R, K); assert len(A) == nv, (len(A), nv)
        r = gen.unweighted_ratio(A, tol=1e-6)
        ndone += 1
        line = "Q(%s, K) |R| %d  |K| %d (index %d)  n %d  degree %d  tri/edge %d..%d  ratio %.4f %s" % (R.name, n, k, U // k, nv, int(A[0].sum()), r[2], r[3], r[0], "ARROWS" if r[0] < 1 else "")
        print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
        if best is None or r[0] < best[0]: best = (r[0], k, nv)
    line = "# done %s |R| %d units %d J2 %d  subgroups %d  sum-free with |K| >= %d: %d  evaluated %d  best %s  [%.0fs]" % (
        R.name, n, U, J2, len(groups), kmin, nsf, ndone, ("ratio %.4f |K| %d n %d" % best) if best else "none", time.time() - t0)
    print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()


if __name__ == "__main__":
    if sys.argv[1][0].isdigit():
        lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = open(sys.argv[3], "a")
        nmax = int(sys.argv[4]) if len(sys.argv) > 4 else 900; kmin = int(sys.argv[5]) if len(sys.argv) > 5 else 10
        for sp in sorted(specs(lo, hi), key=lambda s: np.prod([c[1] for c in s])):
            run(sp, out, nmax, kmin)
    else:
        sp = [(c[0], int(c[1:])) for c in sys.argv[1].split("x")]; out = open(sys.argv[2], "a")
        nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 900; kmin = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        run(sp, out, nmax, kmin)
