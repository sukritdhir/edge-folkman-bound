"""Does the optimal dual solution of the basic SDP for Q(p, e0) violate triangle inequalities?
The graph is flag-transitive, so the invariant optimal dual solution is Y* = (m / dim) P, P = projector onto the
bottom eigenspace of M (edges sharing a triangle).  For +-1 colourings x the matrix x x^T satisfies
  s1 Y_ef + s2 Y_eg + s3 Y_fg >= -1   for all triples of edges and signs with s1 s2 s3 = +1.
This script computes the row Y*[e0, .] and the Gram matrix on the edges strongly correlated with e0, and reports the
most violated triangle inequalities.
usage: python3 exp/triq.py p e0 step [thr] [tol]
"""
import sys, time
sys.path.insert(0, "exp")
import numpy as np, qfam, gen


class Blocks:
    def __init__(self, A, perm):
        n = len(A); self.n = n
        T = gen.triangles(A); ends, tri, deg = gen.edge_system(A, T)
        self.T = T; self.ends = ends; self.tri = tri; self.deg = deg
        m = len(ends); self.m = m; code = ends[:, 0] * n + ends[:, 1]
        perm = np.asarray(perm); q = 1; x = perm.copy()
        while not (x == np.arange(n)).all(): x = perm[x]; q += 1
        self.q = q
        u, v = perm[ends[:, 0]], perm[ends[:, 1]]
        nxt = np.searchsorted(code, np.minimum(u, v) * n + np.maximum(u, v))
        orb = -np.ones(m, dtype=np.int64); off = np.zeros(m, dtype=np.int64); rep = []
        for i in range(m):
            if orb[i] >= 0: continue
            j = i
            for k in range(q):
                assert orb[j] < 0, "edge orbit shorter than q"
                orb[j] = len(rep); off[j] = k; j = int(nxt[j])
            assert j == i
            rep.append(i)
        self.orb = orb; self.off = off; self.no = len(rep); self.rep = np.array(rep)
        rows = []; cols = []; dk = []
        for (x_, y_) in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
            e = tri[:, x_]; f = tri[:, y_]
            sel = off[e] == 0
            rows.append(orb[e][sel]); cols.append(orb[f][sel]); dk.append(off[f][sel])
        self.rows = np.concatenate(rows); self.cols = np.concatenate(cols); self.dk = np.concatenate(dk)

    def block(self, r):
        ph = np.exp(2j * np.pi * r * self.dk / self.q)
        B = np.zeros((self.no, self.no), dtype=complex); np.add.at(B, (self.rows, self.cols), ph)
        return B


def bottom(Bk, tol):
    """eigen-decomposition of all blocks; returns lam_min, dim, list of projectors C_r (r = 0..q-1) onto the eigenvalues
    within tol of the minimum"""
    q = Bk.q; ev = []; vecs = []
    for r in range(q // 2 + 1):
        w, V = np.linalg.eigh(Bk.block(r)); ev.append(w); vecs.append(V)
    lam = min(w[0] for w in ev)
    C = [None] * q; dim = 0
    for r in range(q // 2 + 1):
        sel = ev[r] < lam + tol
        V = vecs[r][:, sel]
        P = V @ V.conj().T
        C[r] = P; dim += int(sel.sum())
        if r != 0 and 2 * r != q:
            C[q - r] = P.conj(); dim += int(sel.sum())
    return lam, dim, C, ev


def gram(Bk, C, dim, E1, E2):
    """Y*[E1, E2] for arrays of edge ids"""
    q = Bk.q; a = Bk.orb[E1]; i = Bk.off[E1]; b = Bk.orb[E2]; k = Bk.off[E2]
    out = np.zeros((len(E1), len(E2)), dtype=complex)
    for r in range(q):
        if C[r] is None: continue
        ph1 = np.exp(2j * np.pi * r * i / q); ph2 = np.exp(-2j * np.pi * r * k / q)
        out += (C[r][np.ix_(a, b)] * ph1[:, None]) * ph2[None, :]
    assert np.abs(out.imag).max() < 1e-8
    return out.real / q * (Bk.m / dim)


if __name__ == "__main__":
    p, e0, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
    Q = qfam.QGraph(p, e0); n = Q.n; t0 = time.time()
    perm = (np.arange(n) + e0 * step) % n
    Bk = Blocks(Q.A, perm); m = Bk.m; k = int(Bk.deg[0])
    lam, dim, C, ev = bottom(Bk, tol)
    print("Q(%d,%d): n %d, m %d, k %d, q %d, %d edge orbits; lam_min %.5f (ratio %.5f), bottom dimension %d  [%.0fs]" % (
        p, e0, n, m, k, Bk.q, Bk.no, lam, lam / (-2 * k / 3), dim, time.time() - t0))
    allev = np.sort(np.concatenate([np.repeat(w, 1 if (r == 0 or 2 * r == Bk.q) else 2) for r, w in enumerate(ev)]))
    print("lowest eigenvalues:", np.round(allev[:6], 4), "...", np.round(allev[dim - 2: dim + 4], 4))
    e_ref = int(Bk.rep[0])
    y = gram(Bk, C, dim, np.array([e_ref]), np.arange(m))[0]
    print("Y[e0,e0] = %.6f; row: min %.4f max (off-diagonal) %.4f" % (y[e_ref], y.min(), np.sort(y)[-2]))
    # values on the triangle neighbours of e0
    tri = Bk.tri
    nb = set()
    for t in tri[(tri == e_ref).any(axis=1)]:
        nb |= set(int(x) for x in t)
    nb.discard(e_ref); nb = np.array(sorted(nb))
    print("triangle neighbours of e0: %d, Y values %.5f .. %.5f (expected %.5f)" % (len(nb), y[nb].min(), y[nb].max(), lam / (2 * k)))
    hist, edges_ = np.histogram(y, bins=[-1.01, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.99, 1.01])
    print("histogram of the row:", list(zip(np.round(edges_[:-1], 2), hist)))
    F = np.nonzero(np.abs(y) >= thr)[0]; F = F[F != e_ref]
    print("edges with |Y| >= %.2f: %d" % (thr, len(F)))
    if len(F) == 0: sys.exit()
    Gm = gram(Bk, C, dim, F, F)
    yF = y[F]
    best = []
    # triples (e0, f, g)
    for (s1, s2, s3) in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        V = s1 * yF[:, None] + s2 * yF[None, :] + s3 * Gm
        np.fill_diagonal(V, 9)
        i, j = np.unravel_index(np.argmin(V), V.shape)
        best.append((float(V[i, j]), (s1, s2, s3), int(F[i]), int(F[j]), float(yF[i]), float(yF[j]), float(Gm[i, j])))
        print("signs %s: min of s1 Y_e0f + s2 Y_e0g + s3 Y_fg = %.4f (f %d, g %d: Y_e0f %.4f, Y_e0g %.4f, Y_fg %.4f); number of pairs below -1: %d" % (
            (s1, s2, s3), V[i, j], F[i], F[j], yF[i], yF[j], Gm[i, j], int((V < -1 - 1e-9).sum()) // (2 if s1 == 1 and s2 == s3 else 1)))
    b = min(best)
    ends = Bk.ends
    print("most violated: value %.4f signs %s  e0 = %s, f = %s, g = %s" % (b[0], b[1], tuple(ends[e_ref]), tuple(ends[b[2]]), tuple(ends[b[3]])))
