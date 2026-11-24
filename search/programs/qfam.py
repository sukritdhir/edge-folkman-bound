"""The family Q(p; V0, S): vertices = nonzero vectors of GF(p)^2 modulo scalars from the subgroup V0 of GF(p)^* (-1 in
V0), [x] ~ [y] iff det(x, y) in S, where S is a union of cosets of V0 with S = -S.  For S = V0 = H (index e) this is
Q(p, e): n = (p+1) e, degree p, |H| triangles on every edge, every link is the cyclotomic graph Cay(Z_p, H), and the graph
is K4-free iff H is sum-free.
The vectors are identified with GF(p^2); the vertex [theta^i] (theta a primitive element) gets the number i mod n,
n = (p+1) e0, e0 = index of V0.  The map i -> i + e0 (multiplication by theta^e0, whose norm lies in V0) is an
automorphism; it generates a cyclic group of order p+1 that acts freely on the vertices and on the edges.  The matrix
M (edges that share a triangle) is block-diagonalised by the characters of this group.
usage: python3 exp/qfam.py p e0 [classes]      classes = comma list of coset numbers c (coset g^c V0), default 0
"""
import sys, time
import numpy as np


def prim_root(p):
    fs = set(); x = p - 1; d = 2
    while d * d <= x:
        while x % d == 0: fs.add(d); x //= d
        d += 1
    if x > 1: fs.add(x)
    return next(g for g in range(2, p) if all(pow(g, (p - 1) // f, p) != 1 for f in fs))


class QGraph:
    def __init__(self, p, e0, classes=(0,)):
        self.p = p; self.e0 = e0; self.classes = tuple(sorted(classes))
        assert (p - 1) % e0 == 0 and ((p - 1) // e0) % 2 == 0, "V0 must contain -1"
        n = (p + 1) * e0; self.n = n
        # GF(p^2) = GF(p)[t] / (t^2 - nr)
        nr = next(a for a in range(2, p) if pow(a, (p - 1) // 2, p) == p - 1)
        def mul(x, y): return ((x[0] * y[0] + nr * x[1] * y[1]) % p, (x[0] * y[1] + x[1] * y[0]) % p)
        def powr(x, k):
            r = (1, 0)
            while k:
                if k & 1: r = mul(r, x)
                x = mul(x, x); k >>= 1
            return r
        N2 = p * p - 1; fs = set(); x = N2; d = 2
        while d * d <= x:
            while x % d == 0: fs.add(d); x //= d
            d += 1
        if x > 1: fs.add(x)
        th = None
        for a in range(p):
            for b in range(1, p):
                if all(powr((a, b), N2 // f) != (1, 0) for f in fs): th = (a, b); break
            if th: break
        self.theta = th
        # representatives theta^i, i = 0..n-1
        X = np.zeros((n, 2), dtype=np.int64); cur = (1, 0)
        for i in range(n): X[i] = cur; cur = mul(cur, th)
        self.X = X
        # g = norm of theta = theta^(p+1) generates GF(p)^*; class of a nonzero residue = its log base g modulo e0
        g = powr(th, p + 1); assert g[1] == 0; g = g[0]
        cl = np.full(p, -1, dtype=np.int64); y = 1
        for i in range(p - 1): cl[y] = i % e0; y = (y * g) % p
        self.g = g; self.cl = cl
        D = (X[:, 0][:, None] * X[:, 1][None, :] - X[:, 1][:, None] * X[:, 0][None, :]) % p
        A = np.isin(cl[D], list(self.classes)) & (D != 0)
        assert (A == A.T).all(), "S must be symmetric"
        assert (A == np.roll(np.roll(A, e0, axis=0), e0, axis=1)).all(), "the shift by e0 must be an automorphism"
        self.A = A; self.deg = int(A[0].sum()); assert (A.sum(axis=1) == self.deg).all()

    def link_triangle_free(self):
        """K4-free iff the link of one vertex of each orbit of the cyclic group is triangle-free (checked for e0 vertices)"""
        for v in range(self.e0):
            Nv = np.nonzero(self.A[v])[0]; L = self.A[np.ix_(Nv, Nv)].astype(np.int64)
            if np.trace(L @ L @ L) != 0: return False
        return True

    def edge_orbits(self):
        n = self.n; e0 = self.e0; m = self.p + 1; A = self.A
        iu, ju = np.nonzero(np.triu(A, 1)); ne = len(iu)
        eid = -np.ones((n, n), dtype=np.int64); orb = np.zeros(ne, dtype=np.int64); pos = np.zeros(ne, dtype=np.int64)
        eid[iu, ju] = np.arange(ne); eid[ju, iu] = np.arange(ne)
        seen = np.zeros(ne, dtype=bool); reps = []
        for t in range(ne):
            if seen[t]: continue
            u, v = iu[t], ju[t]; o = len(reps); reps.append((int(u), int(v)))
            s = np.arange(m); ids = eid[(u + e0 * s) % n, (v + e0 * s) % n]
            assert (ids >= 0).all() and len(set(ids.tolist())) == m, "the cyclic group must act freely on the edges"
            seen[ids] = True; orb[ids] = o; pos[ids] = s
        return eid, orb, pos, reps

    def blocks_data(self):
        """for each orbit representative (u, v): the edges (u, w), (v, w) over the common neighbours w, as (orbit, position)"""
        eid, orb, pos, reps = self.edge_orbits(); A = self.A
        rows = []; cols = []; ps = []; tri = []
        for o, (u, v) in enumerate(reps):
            W = np.nonzero(A[u] & A[v])[0]; tri.append(len(W))
            for f in (eid[u, W], eid[v, W]):
                rows.extend([o] * len(W)); cols.extend(orb[f].tolist()); ps.extend(pos[f].tolist())
        return np.array(rows), np.array(cols), np.array(ps), np.array(tri, dtype=float), len(reps)

    def ratio(self, verbose=False):
        rows, cols, ps, tri, no = self.blocks_data(); m = self.p + 1
        if (tri == 0).any(): return 9.0, tri.min(), tri.max(), no
        dm = 1 / np.sqrt(tri); lam = np.inf; flat = rows * no + cols
        for r in range(m // 2 + 1):
            ph = np.exp(2j * np.pi * r * ps / m)
            B = (np.bincount(flat, weights=ph.real, minlength=no * no) + 1j * np.bincount(flat, weights=ph.imag, minlength=no * no)).reshape(no, no)
            assert np.abs(B - B.conj().T).max() < 1e-9
            ev = np.linalg.eigvalsh(B * dm[:, None] * dm[None, :])
            if verbose: print("   character %3d: lambda_min %.6f" % (r, ev[0]))
            lam = min(lam, ev[0])
        return float(lam / (-2 / 3)), int(tri.min()), int(tri.max()), no


if __name__ == "__main__":
    p, e0 = int(sys.argv[1]), int(sys.argv[2])
    classes = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0,)
    t0 = time.time(); Q = QGraph(p, e0, classes)
    print("Q(%d; index %d, classes %s): n = %d, degree %d, K4-free: %s" % (p, e0, classes, Q.n, Q.deg, Q.link_triangle_free())); sys.stdout.flush()
    r = Q.ratio(verbose="-v" in sys.argv)
    print("edge orbits %d (blocks of this size, %d characters), triangles per edge %d..%d, unweighted ratio %.6f %s  [%.0fs]" % (r[3], p + 1, r[1], r[2], r[0], "ARROWS" if r[0] < 1 else "", time.time() - t0))
