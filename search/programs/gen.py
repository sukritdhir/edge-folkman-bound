"""Generic helpers for arbitrary graphs given by a boolean adjacency matrix: triangle list, edge index, the
unweighted normalised ratio  lambda_min(Deg^(-1/2) M Deg^(-1/2)) / (-2/3)  (below 1 = the graph arrows (3,3)), K4 test.
Edges that lie in no triangle are ignored (they can be coloured freely)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def triangles(A):
    """all triangles (u < v < w) of the graph, as an array (nt, 3)"""
    n = len(A); out = []
    for u in range(n):
        Nu = np.nonzero(A[u, u + 1:])[0] + u + 1
        if len(Nu) < 2: continue
        sub = np.triu(A[np.ix_(Nu, Nu)], 1)
        i, j = np.nonzero(sub)
        if len(i): out.append(np.stack([np.full(len(i), u), Nu[i], Nu[j]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.int64)


def k4_free(A, T=None):
    """True iff no triangle has a common neighbour of its three vertices"""
    if T is None: T = triangles(A)
    for s in range(0, len(T), 20000):
        t = T[s:s + 20000]
        if (A[t[:, 0]] & A[t[:, 1]] & A[t[:, 2]]).any(): return False
    return True


def edge_system(A, T=None):
    """edges that lie in triangles: (ends (m, 2), tri (nt, 3) of edge ids, deg (m,))"""
    n = len(A)
    if T is None: T = triangles(A)
    code = np.concatenate([T[:, 0] * n + T[:, 1], T[:, 0] * n + T[:, 2], T[:, 1] * n + T[:, 2]])
    uq, inv = np.unique(code, return_inverse=True)
    m = len(uq); nt = len(T)
    tri = inv.reshape(3, nt).T
    ends = np.stack([uq // n, uq % n], axis=1)
    deg = np.bincount(inv, minlength=m)
    return ends, tri, deg


def unweighted_ratio(A, tol=1e-7, k=3, T=None, return_vec=False):
    ends, tri, deg = edge_system(A, T)
    m = len(ends)
    if m == 0: return 9.0, 0, 0, 0
    r = np.concatenate([tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 1], tri[:, 2], tri[:, 2]])
    c = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0], tri[:, 2], tri[:, 0], tri[:, 1]])
    M = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(m, m))
    dm = sp.diags(1 / np.sqrt(deg))
    N = (dm @ M @ dm).tocsr()
    if m < 400:
        lam, vec = np.linalg.eigh(N.toarray()); lam = lam[:1]; vec = vec[:, :1]
    else:
        lam, vec = eigsh(N, k=k, which="SA", tol=tol, maxiter=50000)
    i = int(np.argmin(lam))
    res = (float(lam[i] / (-2 / 3)), m, int(deg.min()), int(deg.max()))
    return res + ((vec[:, i], ends),) if return_vec else res
