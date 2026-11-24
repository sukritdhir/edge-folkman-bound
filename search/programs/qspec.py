"""Full spectrum of the edge-triangle matrix M (two edges adjacent iff they lie in a common triangle) of a graph with a
cyclic automorphism group acting freely on the edges, by Fourier blocks.  For Q(p, e0): shift by e0*step.
usage: python3 exp/qspec.py p e0 [step]     prints the distinct eigenvalues of M with multiplicities (lowest and highest)
"""
import sys
sys.path.insert(0, "exp")
import numpy as np, qfam, gen


def spectrum(A, shift_perm):
    n = len(A); T = gen.triangles(A); ends, tri, deg = gen.edge_system(A, T)
    m = len(ends); code = ends[:, 0] * n + ends[:, 1]
    perm = np.asarray(shift_perm)
    q = 1; x = perm.copy()
    while not (x == np.arange(n)).all(): x = perm[x]; q += 1
    # orbit of every edge
    def img(c):
        u, v = perm[c // n], perm[c % n]
        return np.minimum(u, v) * n + np.maximum(u, v)
    orb = -np.ones(m, dtype=np.int64); off = np.zeros(m, dtype=np.int64); no = 0
    pos = {int(c): i for i, c in enumerate(code)}
    nxt = np.array([pos[int(c)] for c in img(code)])
    for i in range(m):
        if orb[i] >= 0: continue
        j = i
        for k in range(q):
            assert orb[j] < 0, "edge orbit shorter than q"
            orb[j] = no; off[j] = k; j = nxt[j]
        assert j == i
        no += 1
    # blocks: B_r[a, b] = sum over pairs (e in orbit a at offset i, f in orbit b at offset k) in a common triangle, i = 0: w^(r k)
    rows = []; cols = []; dk = []
    for (x, y) in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        e = tri[:, x]; f = tri[:, y]
        sel = off[e] == 0
        rows.append(orb[e][sel]); cols.append(orb[f][sel]); dk.append(off[f][sel])
    rows = np.concatenate(rows); cols = np.concatenate(cols); dk = np.concatenate(dk)
    evs = []
    for r in range(q // 2 + 1):
        ph = np.exp(2j * np.pi * r * dk / q)
        B = np.zeros((no, no), dtype=complex); np.add.at(B, (rows, cols), ph)
        assert np.abs(B - B.conj().T).max() < 1e-9
        w = np.linalg.eigvalsh(B)
        mult = 1 if (r == 0 or 2 * r == q) else 2
        evs.append(np.repeat(w, mult))
    return np.sort(np.concatenate(evs)), int(deg.min()), int(deg.max()), m


def distinct(ev, tol=1e-6):
    out = []; i = 0
    while i < len(ev):
        j = i
        while j < len(ev) and ev[j] - ev[i] < tol: j += 1
        out.append((float(ev[i:j].mean()), j - i)); i = j
    return out


if __name__ == "__main__":
    p, e0 = int(sys.argv[1]), int(sys.argv[2]); step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    Q = qfam.QGraph(p, e0); n = Q.n
    perm = (np.arange(n) + e0 * step) % n
    ev, dmin, dmax, m = spectrum(Q.A, perm)
    k = dmin
    d = distinct(ev)
    print("Q(%d,%d): %d edges, k = %d..%d, %d distinct eigenvalues of M; ratio %.5f" % (p, e0, m, dmin, dmax, len(d), ev[0] / (-2 * k / 3)))
    print("lowest:  " + "  ".join("%.4f x%d" % x for x in d[:14]))
    print("highest: " + "  ".join("%.4f x%d" % x for x in d[-8:]))
