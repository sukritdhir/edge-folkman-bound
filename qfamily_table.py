#!/usr/bin/env python3
"""
qfamily_table.py -- floating-point survey of the graphs Q(p, e) with H sum-free (not part of any proof).

For every prime p <= PMAX and every divisor e of p - 1 such that the group H of the e-th powers modulo p has
even order k = (p-1)/e and is sum-free, the program builds Q(p, e) from the definition (classes of nonzero
vectors of GF(p)^2 modulo H; adjacency det in H), checks that it is p-regular with k triangles on every edge
and without K4, and computes the smallest eigenvalue lambda of the 0/1 matrix M on the edges ("two edges lie in
a common triangle") by the Lanczos method.  The eigenvalue test with equal weights is  lambda > -2k/3.
The column `ratio` is lambda / (-2k/3); the test holds iff ratio < 1.

Usage: python3 qfamily_table.py [PMAX [NMAX]]      (defaults 300 and 1700; NMAX bounds the number of vertices)
"""
import sys, math, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def graph(p, e):
    k = (p - 1) // e
    H = sorted(set(pow(x, e, p) for x in range(1, p)))
    inH = np.zeros(p, dtype=bool); inH[H] = True
    Ha = np.array(H, dtype=np.int64)
    sumfree = not bool(inH[(Ha[:, None] + Ha[None, :]) % p].any())
    if not (inH[p - 1] and sumfree):
        return None
    a = np.repeat(np.arange(p, dtype=np.int64), p); b = np.tile(np.arange(p, dtype=np.int64), p)
    best = np.full(p * p, p * p, dtype=np.int64)
    for h in H:
        np.minimum(best, ((a * h) % p) * p + (b * h) % p, out=best)
    canon = np.unique(best[1:])
    ra, rb = canon // p, canon % p
    adj = inH[(ra[:, None] * rb[None, :] - rb[:, None] * ra[None, :]) % p]
    return adj, k


def lam_min_edge_graph(adj):
    n = adj.shape[0]
    us, vs = np.nonzero(np.triu(adj, 1))
    eid = -np.ones((n, n), dtype=np.int64)
    eid[us, vs] = np.arange(len(us)); eid[vs, us] = np.arange(len(us))
    rows = []; cols = []
    ntri = 0; k4 = False; cnts = set()
    for i, (u, v) in enumerate(zip(us.tolist(), vs.tolist())):
        c = np.flatnonzero(adj[u] & adj[v])
        cnts.add(len(c))
        if len(c) and adj[np.ix_(c, c)].any():
            k4 = True
        rows.append(np.full(2 * len(c), i)); cols.append(np.concatenate([eid[u, c], eid[v, c]]))
        ntri += int((c > v).sum())
    m = len(us)
    M = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m))
    assert (M != M.T).nnz == 0
    lam = float(eigsh(M, k=1, which="SA", tol=1e-10, ncv=60, maxiter=100000, return_eigenvectors=False)[0])
    return m, ntri, sorted(cnts), k4, lam


def main():
    pmax = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 1700
    print("   p   e    n    k   edges  triangles  lambda_min(M)   -2k/3     ratio   test")
    for p in range(5, pmax + 1):
        if any(p % r == 0 for r in range(2, math.isqrt(p) + 1)):
            continue
        for e in range(2, p - 1):
            if (p - 1) % e or ((p - 1) // e) % 2 or (p + 1) * e > nmax or (p - 1) // e < 2:
                continue
            g = graph(p, e)
            if g is None:
                continue
            adj, k = g
            n = adj.shape[0]
            assert n == (p + 1) * e and bool((adj.sum(axis=1) == p).all())
            m, ntri, cnts, k4, lam = lam_min_edge_graph(adj)
            assert cnts == [k] and not k4
            thr = -2 * k / 3
            print("%4d %3d %4d %4d %7d %10d %14.6f %9.4f %9.6f   %s"
                  % (p, e, n, k, m, ntri, lam, thr, lam / thr, "holds" if lam > thr else "fails"), flush=True)


if __name__ == "__main__":
    main()
