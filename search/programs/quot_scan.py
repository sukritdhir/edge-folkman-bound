"""Fast quotient statistics for all sets S of removed K'-cosets of G_941 (vertex 0 removed as well).
For every canonical S (up to the shift c -> c+5) of the given size: vertices, edges, triangles, degree and triangles per
vertex by coset (min, std), triangles per edge (min, std).  Output: scan/quot_{size}.npz and a ranked text summary.
usage: python3 exp/quot_scan.py size"""
import numpy as np, sys, itertools, time
p = 941; g = 2
ind = np.zeros(p, dtype=np.int64); x = 1
for k in range(p - 1):
    ind[x] = k; x = x * g % p
cos = ind % 20; cos[0] = -1
isH = (ind % 5 == 0); isH[0] = False
A = isH[(np.arange(p)[:, None] - np.arange(p)[None, :]) % p]
rep = [int(pow(g, c, p)) for c in range(20)]                    # representative vertex of coset c
onehot = np.zeros((p, 20)); onehot[np.arange(1, p), cos[1:]] = 1
rows_a = []; rows_b = []; CN = []
for a in range(20):
    v = rep[a]; nb = np.nonzero(A[v])[0]; nb = nb[nb != 0]
    for u in nb:
        rows_a.append(a); rows_b.append(cos[u]); CN.append((A[v] & A[u]).astype(float) @ onehot)
rows_a = np.array(rows_a); rows_b = np.array(rows_b); CN = np.array(CN)       # (20*188 - #edges to 0, 20)
size = int(sys.argv[1])
def canon(S):
    return min(tuple(sorted((c + 5 * k) % 20 for c in S)) for k in range(4))
sets = sorted(set(canon(S) for S in itertools.combinations(range(20), size)))
print(len(sets), "canonical sets of size", size)
out = []
t0 = time.time()
for S in sets:
    keepc = np.ones(20, bool); keepc[list(S)] = False
    m = keepc[rows_a] & keepc[rows_b]
    te = CN[m][:, keepc].sum(axis=1)                             # triangles on each (doubly counted) edge
    ra = rows_a[m]
    deg = np.bincount(ra, minlength=20)[keepc].astype(float)
    tv = np.bincount(ra, weights=te, minlength=20)[keepc] / 2
    ne = 47 * m.sum() / 2; nt = 47 * te.sum() / 6
    out.append((ne, nt, deg.min(), deg.max(), deg.std(), tv.min(), tv.max(), tv.std(), te.min(), te.mean(), te.std()))
out = np.array(out)
np.savez("scan/quot_%d.npz" % size, sets=np.array(sets), stats=out)
print("done [%.0fs]" % (time.time() - t0))
names = ["edges", "tri", "degmin", "degmax", "degstd", "tvmin", "tvmax", "tvstd", "temin", "temean", "testd"]
def show(key, k=12, rev=True):
    j = names.index(key); order = np.argsort(out[:, j]); order = order[::-1] if rev else order
    print("--- best by", key)
    for i in order[:k]:
        print("  %-18s classes %s  e %d t %d  deg %d..%d std %.2f  tri/v %d..%d std %.1f  tri/e min %d mean %.2f std %.2f"
              % (",".join(map(str, sets[i])), sorted(c % 5 for c in sets[i]), out[i, 0], out[i, 1], out[i, 2], out[i, 3], out[i, 4], out[i, 5], out[i, 6], out[i, 7], out[i, 8], out[i, 9], out[i, 10]))
show("tvmin"); show("degstd", rev=False); show("tvstd", rev=False); show("tri"); show("testd", rev=False)
