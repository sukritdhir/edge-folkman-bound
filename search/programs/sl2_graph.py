"""'Locally cyclotomic' graphs Q(p, e): vertices = nonzero vectors of GF(p)^2 modulo scalars from H (the subgroup of
index e of GF(p)^*, -1 in H), [x] ~ [y] iff det(x, y) in H.  n = (p+1) e, degree p, |H| triangles on every edge, the
link of every vertex is the cyclotomic graph Cay(Z_p, H); K4-free iff H is sum-free.
usage: python3 exp/sl2_graph.py p e          prints the parameters, the K4 check and the unweighted normalised ratio
"""
import sys, numpy as np, itertools, time
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def build(p, e, S_classes=(0,)):
    g = next(g for g in range(2, p) if len(set(pow(g, i, p) for i in range(p - 1))) == p - 1)
    H = sorted(set(pow(g, e * i, p) for i in range((p - 1) // e)))
    assert (p - 1) in H, "-1 must lie in H"
    Hs = set(H)
    cls = {}                                                       # class index of every nonzero residue mod H
    for i in range(p - 1): cls[pow(g, i, p)] = i % e
    # vertices: canonical representative of each H-orbit of nonzero vectors
    vid = {}; reps = []
    for a in range(p):
        for b in range(p):
            if (a, b) == (0, 0) or (a, b) in vid: continue
            for h in H: vid[((a * h) % p, (b * h) % p)] = len(reps)
            reps.append((a, b))
    n = len(reps); assert n == (p + 1) * e
    R = np.array(reps)
    D = (R[:, 0][:, None] * R[:, 1][None, :] - R[:, 1][:, None] * R[:, 0][None, :]) % p
    cl = np.full(p, -1); 
    for x, c in cls.items(): cl[x] = c
    A = np.isin(cl[D], list(S_classes)) & (D != 0)
    assert (A == A.T).all()
    return A, H


def k4_free(A):
    n = len(A); Ai = A.astype(np.int64)
    # a K4 exists iff some edge uv has two adjacent common neighbours
    for u in range(n):
        Nu = np.nonzero(A[u])[0]
        for v in Nu[Nu > u]:
            W = np.nonzero(A[u] & A[v])[0]
            if len(W) > 1 and A[np.ix_(W, W)].any(): return False
    return True


def ratio(A, k=6):
    n = len(A); iu, ju = np.nonzero(np.triu(A, 1)); m = len(iu)
    eid = -np.ones((n, n), dtype=np.int64); eid[iu, ju] = np.arange(m); eid[ju, iu] = np.arange(m)
    rows = []; cols = []
    deg = np.zeros(m)
    for t in range(m):
        u, v = iu[t], ju[t]
        W = np.nonzero(A[u] & A[v])[0]; deg[t] = len(W)
        rows.extend([t] * (2 * len(W))); cols.extend(eid[u, W]); cols.extend(eid[v, W])
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m))
    dm = sp.diags(1 / np.sqrt(deg))
    N = dm @ M @ dm
    lam = eigsh(N, k=k, which="SA", return_eigenvectors=False, tol=1e-9, maxiter=20000)
    return float(lam.min() / (-2 / 3)), m, int(deg.min()), int(deg.max()), np.sort(lam)


if __name__ == "__main__":
    p, e = int(sys.argv[1]), int(sys.argv[2])
    classes = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0,)
    t0 = time.time(); A, H = build(p, e, classes); n = len(A); d = A.sum(axis=1)
    link = np.array([[1.0 if ((s - t) % p) in set(H) else 0.0 for t in range(p)] for s in range(p)])
    ev = np.linalg.eigvalsh(link)
    print("Q(%d,%d) classes %s: n = %d, degree %d..%d, |H| = %d, link Cay(Z_p,H) lambda_min %.4f (ratio to degree %.4f)" % (p, e, classes, n, d.min(), d.max(), len(H), ev[0], ev[0] / len(H)))
    sumfree = not any(((a + b) % p) in set(H) for a in H for b in H)
    print("H sum-free: %s;  K4-free (direct check): %s" % (sumfree, k4_free(A))); sys.stdout.flush()
    r, m, dmin, dmax, lam = ratio(A)
    print("edges %d, triangles per edge %d..%d, unweighted ratio %.5f %s  (lowest eigenvalues %s)  [%.0fs]" % (m, dmin, dmax, r, "ARROWS" if r < 1 else "", np.round(lam, 4), time.time() - t0))
