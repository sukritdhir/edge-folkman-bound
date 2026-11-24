"""Unweighted spectral criterion for a graph given by an adjacency matrix (npy): edges, triangles, K4 count, and
lambda_min(N) for N = D^(-1/2) M D^(-1/2), M_ef = 1 if e, f lie in a common triangle, D_e = number of triangles at e.
The graph arrows (3,3) if lambda_min(N) > -2/3.   usage: python3 exp/spec_crit.py adj.npy"""
import numpy as np, sys, itertools
import scipy.sparse as sp
A = np.load(sys.argv[1]).astype(np.int32); n = len(A)
iu = np.triu_indices(n, 1); m = A[iu] == 1
ends = np.stack([iu[0][m], iu[1][m]], axis=1); E = len(ends)
eid = -np.ones((n, n), dtype=np.int64); eid[ends[:, 0], ends[:, 1]] = np.arange(E); eid[ends[:, 1], ends[:, 0]] = np.arange(E)
tris = []
nb = [np.nonzero(A[v])[0] for v in range(n)]
for (u, v) in ends:
    for w in nb[u]:
        if w > v and A[v, w]: tris.append((eid[u, v], eid[u, w], eid[v, w]))
tris = np.array(tris); T = len(tris)
A2 = A @ A
k4 = 0
for (u, v) in ends:
    cn = np.nonzero(A[u] & A[v])[0]
    k4 += int(A[np.ix_(cn, cn)].sum()) // 2
k4 //= 6
te = np.bincount(tris.ravel(), minlength=E)
print("n %d, edges %d, triangles %d, K4s %d, triangles per edge min %d max %d mean %.2f" % (n, E, T, k4, te.min(), te.max(), te.mean()))
rows = np.concatenate([tris[:, 0], tris[:, 0], tris[:, 1], tris[:, 1], tris[:, 2], tris[:, 2]])
cols = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0], tris[:, 2], tris[:, 0], tris[:, 1]])
M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(E, E)).toarray()
d = 1 / np.sqrt(te.astype(float))
N = M * d[:, None] * d[None, :]
ev = np.linalg.eigvalsh(N)
vals, cnt = np.unique(np.round(ev, 6), return_counts=True)
print("lambda_min(N) = %.6f (threshold -0.666667): %s" % (ev[0], "ARROWS (3,3) by the unweighted criterion" if ev[0] > -2 / 3 + 1e-9 else "criterion fails"))
print("distinct eigenvalues of N:", [(float(v), int(c)) for v, c in zip(vals, cnt)][:12], "..." if len(vals) > 12 else "")
