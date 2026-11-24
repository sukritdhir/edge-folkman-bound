"""Quotient statistics for the 20 types "whole coset j H of the 5th powers (K'-cosets j, j+5, j+10, j+15) plus one more
K'-coset c" of five removed cosets (vertex 0 removed too): edges, triangles, degree spread, triangles per vertex and per
edge.  Also the 5 four-coset bases (one whole class).   usage: python3 exp/class_plus.py"""
import numpy as np
p = 941; g = 2
ind = np.zeros(p, dtype=np.int64); x = 1
for k in range(p - 1):
    ind[x] = k; x = x * g % p
cos = ind % 20; cos[0] = -1
isH = (ind % 5 == 0); isH[0] = False
A = isH[(np.arange(p)[:, None] - np.arange(p)[None, :]) % p].astype(np.float32)
def stats(S):
    keep = np.array([v for v in range(1, p) if cos[v] not in S])
    B = A[np.ix_(keep, keep)]
    deg = B.sum(axis=1); B2 = B @ B; T = B2 * B
    tri_v = T.sum(axis=1) / 2; tev = T[B > 0]
    return len(keep), int(B.sum() / 2), int(tri_v.sum() / 3), deg.min(), deg.max(), deg.std(), tri_v.min(), tri_v.std(), tev.min(), tev.mean(), tev.std()
print("class j alone (752 vertices):")
for j in range(5):
    S = [j, j + 5, j + 10, j + 15]
    print("  j = %d  %s" % (j, " ".join("%.2f" % v if isinstance(v, float) or hasattr(v, "dtype") else str(v) for v in stats(S))))
print("class j plus coset c:  n, edges, triangles, deg min/max/std, tri per vertex min/std, tri per edge min/mean/std")
for j in range(5):
    for cm in range(5):
        if cm == j: continue
        S = [j, j + 5, j + 10, j + 15, cm]
        st = stats(S)
        print("  j = %d c = %d  set %-14s %d %d %d  deg %d..%d std %.2f  tv min %d std %.1f  te min %d mean %.3f std %.3f" % ((j, cm, ",".join(str(c) for c in sorted(S))) + tuple(st)))
