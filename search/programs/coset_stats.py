"""Quotient statistics of G_941 by the 20 cosets of K' = <302> (order 47): edges between cosets, and for a set S of
removed cosets (vertex 0 removed too) the degrees and triangle counts of the remaining vertices, per coset.
usage: python3 exp/coset_stats.py set1 set2 ...   (sets like 0,3,7,12,18)"""
import numpy as np, sys
p = 941; g = 2
ind = np.zeros(p, dtype=np.int64); x = 1
for k in range(p - 1):
    ind[x] = k; x = x * g % p
assert x == 1 and len(set(ind[1:])) == p - 1
cos = ind % 20; cos[0] = -1                      # coset index of each nonzero vertex (K' = 20th powers, order 47)
isH = (ind % 5 == 0); isH[0] = False
A = isH[(np.arange(p)[:, None] - np.arange(p)[None, :]) % p]
A = A.astype(np.float32)
A2 = A @ A                                       # common neighbours
def stats(S):
    keep = np.array([v for v in range(1, p) if cos[v] not in S])
    B = A[np.ix_(keep, keep)]
    deg = B.sum(axis=1)
    B2 = B @ B
    tri_v = (B2 * B).sum(axis=1) / 2             # triangles per vertex
    te = (B2 * B)                                # triangles per edge
    tev = te[B > 0]
    cs = cos[keep]
    out = []
    for c in sorted(set(cs)):
        m = cs == c
        out.append((c, deg[m].mean(), tri_v[m].mean(), te[m][B[m] > 0].mean(), te[m][B[m] > 0].min()))
    return len(keep), int(B.sum() / 2), int(tri_v.sum() / 3), tev, out
for s in sys.argv[1:]:
    S = [int(c) for c in s.split(",")]
    nv, ne, nt, tev, out = stats(S)
    d = np.array([o[1] for o in out]); t = np.array([o[2] for o in out])
    print("S = %-16s classes %s  n %d e %d t %d  t/e %.4f | deg per coset: min %.1f max %.1f std %.2f | tri/vertex: min %.0f max %.0f std %.1f | tri per edge: min %d std %.2f"
          % (s, sorted([c % 5 for c in S]), nv, ne, nt, nt / ne, d.min(), d.max(), d.std(), t.min(), t.max(), t.std(), tev.min(), tev.std()))
