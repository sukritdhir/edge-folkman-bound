#!/usr/bin/env python3
"""
unweighted.py -- the test with equal weights for the graph of a certificate (floating point; not part of any proof).

Usage:  python3 unweighted.py CERTIFICATE [CERTIFICATE ...]

For the target graph G of each certificate, let I be the set of the edges that lie in a triangle, t_e the number of
triangles on e, D = diag(t_e), and M the 0/1 matrix "two edges lie in a common triangle".  With w_T = 1 and
d_e = beta * t_e the matrix A(w, d) = 3 M + 2 beta D is positive semidefinite iff
beta >= beta_0 := -(3/2) lambda_min(D^(-1/2) M D^(-1/2)), and the two sides of the inequality (ii) have the
quotient beta.  So the pair with equal weights is a certificate iff beta_0 < 1.  The program prints beta_0.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import check_independent as ci


def beta0(adj):
    us, vs = np.nonzero(np.triu(adj, 1))
    eid = {}
    rows = []; cols = []
    te = []
    for u, v in zip(us.tolist(), vs.tolist()):
        eid[(u, v)] = len(eid)
    te = np.zeros(len(eid))
    for (u, v), i in eid.items():
        c = np.flatnonzero(adj[u] & adj[v])
        te[i] = len(c)
        for w in c[c > v].tolist():
            i1, i2, i3 = i, eid[(u, w)], eid[(v, w)]
            rows += [i1, i1, i2, i2, i3, i3]; cols += [i2, i3, i1, i3, i1, i2]
    keep = te > 0
    m = len(eid)
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m))
    s = np.where(keep, 1.0 / np.sqrt(np.maximum(te, 1)), 0.0)
    N = sp.diags(s) @ M @ sp.diags(s)
    # largest eigenvalue of -N, so that the tolerance of the Lanczos method is relative to a number of size 1
    lam = -float(eigsh(-N, k=1, which="LA", tol=1e-9, ncv=60, maxiter=100000, return_eigenvectors=False)[0])
    return -1.5 * lam, int(keep.sum()), m


def main():
    cache = {}
    for path in sys.argv[1:]:
        cert = ci.read_certificate(path)
        key = (cert["p"], cert["e0"], cert["nr"], cert["theta"])
        if key not in cache:
            cache[key] = ci.build_q(*key)
        Q = cache[key]
        keep = np.ones(Q["n"], dtype=bool)
        keep[cert["removed"]] = False
        keep[cert["extra"]] = False
        b, ni, m = beta0(Q["adj"][np.ix_(keep, keep)])
        print("%-34s %4d vertices, %6d edges (%d in a triangle): beta_0 = %.6f  (%s)"
              % (path.split("/")[-1], int(keep.sum()), m, ni, b, "passes" if b < 1 else "fails by %.2f per cent" % (100 * (b - 1))), flush=True)


if __name__ == "__main__":
    main()
