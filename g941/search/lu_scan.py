"""Scan Lu-type circulants  G = Cay(Z_n, S),  S a cyclic subgroup of Z_n^* containing -1,  K4-free.

For each graph: k = triangles per edge, and two spectral tests for arrowing (3,3):
  basic     lambda_min(A_H) > -2k/3                 (Dudek-Rodl / Lu eigenvalue test; H = edge-triangle graph)
  weighted  max over weights mu on the triangle orbits (sum over a base edge normalised to k) of
            lambda_min(sum_o mu_o A_o) > -2k/3       (new: one multiplier per orbit of triangles)
Both are rigorous in exact arithmetic: for a colouring x without monochromatic triangle,
  x^T M(mu) x = -2 sum_T mu_T  while  x^T M x >= lambda_min(M) |E|.
The ratio printed is  -lambda_min / (2k/3);  a value below 1 proves G -> (3,3).
"""
import numpy as np, sys, time, math
from math import gcd


def cyclic_subgroups(n):
    units = [a for a in range(1, n) if gcd(a, n) == 1]
    seen = set(); out = []
    for s in units:
        S = []; x = 1
        while True:
            S.append(x); x = (x * s) % n
            if x == 1: break
        fs = frozenset(S)
        if fs in seen: continue
        seen.add(fs)
        if (n - 1) in fs: out.append(sorted(S))
    return out


class LuGraph:
    def __init__(self, n, S):
        self.n = n; self.S = S; self.Sset = set(S); self.d = len(S)
        self.W = sorted(w for w in S if (w - 1) % n in self.Sset)       # common neighbours of 0 and 1
        self.k = len(self.W)

    def k4free(self):
        W = self.W; Sset = self.Sset; n = self.n
        Ws = set(W)
        for i, a in enumerate(W):
            for b in W[i + 1:]:
                if (b - a) % n in Sset: return False
        return True

    def setup(self):
        n = self.n; S = self.S; Sset = self.Sset
        self.Cp = [c for c in S if 2 * c < n]
        self.m = len(self.Cp); self.cidx = {c: j for j, c in enumerate(self.Cp)}
        # triangle orbits = orbits of the anharmonic group on W
        inv = lambda a: pow(a, -1, n)
        typ = {}; self.orbits = []
        for w in self.W:
            if w in typ: continue
            orb = set(); stack = [w]
            while stack:
                u = stack.pop()
                if u in orb: continue
                orb.add(u)
                stack.append((1 - u) % n); stack.append(inv(u))
            assert all(u in set(self.W) for u in orb), "anharmonic orbit leaves W"
            for u in orb: typ[u] = len(self.orbits)
            self.orbits.append(sorted(orb))
        self.typ = typ; self.T = len(self.orbits)
        # frequency representatives under multiplication by S
        seen = set(); self.freqs = []
        for f in range(n):
            if f in seen: continue
            orb = set((a * f) % n for a in S); seen |= orb
            self.freqs.append(f)
        # entries: for base edge (0,j) and w in W: neighbours f = {0, c w}, g = {c, c w}
        ent = []            # (j, j2, x2, type)
        for j, c in enumerate(self.Cp):
            for w in self.W:
                v = (c * w) % n
                for (a, b) in ((0, v), (c, v)):
                    dd = (b - a) % n
                    if dd in self.cidx: x2, j2 = a, self.cidx[dd]
                    else: x2, j2 = b, self.cidx[(a - b) % n]
                    ent.append((j, j2, x2, typ[w]))
        self.ent = np.array(ent)

    def blocks(self, f):
        """A_o(f) for each triangle orbit o: array (T, m, m) complex."""
        m = self.m; e = self.ent
        ph = np.exp(2j * np.pi * ((f * e[:, 2]) % self.n) / self.n)
        B = np.zeros((self.T, m, m), dtype=complex)
        np.add.at(B, (e[:, 3], e[:, 0], e[:, 1]), ph)
        return B

    def basic(self):
        lam = np.inf; arg = None
        self.B = {}
        for f in self.freqs:
            B = self.blocks(f); self.B[f] = B
            M = B.sum(axis=0)
            assert np.abs(M - M.conj().T).max() < 1e-9
            ev = np.linalg.eigvalsh(M)
            if ev[0] < lam: lam = ev[0]; arg = f
        return lam, arg

    def weighted(self, solver="SCS", eps=1e-7):
        import cvxpy as cp
        m = self.m
        mu = cp.Variable(self.T); tau = cp.Variable()
        cons = [sum(len(self.orbits[o]) * mu[o] for o in range(self.T)) == self.k]
        for f in self.freqs:
            B = self.B[f]
            real = np.abs(B.imag).max() < 1e-9
            if real:
                M = sum(mu[o] * B[o].real for o in range(self.T))
                cons.append(M - tau * np.eye(m) >> 0)
            else:
                Emb = [np.block([[B[o].real, -B[o].imag], [B[o].imag, B[o].real]]) for o in range(self.T)]
                M = sum(mu[o] * Emb[o] for o in range(self.T))
                cons.append(M - tau * np.eye(2 * m) >> 0)
        prob = cp.Problem(cp.Maximize(tau), cons)
        if solver == "SCS": prob.solve(solver="SCS", eps=eps, max_iters=100000)
        else: prob.solve(solver=solver)
        muv = np.array(mu.value)
        # recompute the true minimum eigenvalue for the returned mu (after exact renormalisation)
        muv = muv * self.k / sum(len(self.orbits[o]) * muv[o] for o in range(self.T))
        lam = min(np.linalg.eigvalsh(np.tensordot(muv, self.B[f], axes=1))[0] for f in self.freqs)
        return lam, muv


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    kmin = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    near = float(sys.argv[4]) if len(sys.argv) > 4 else 1.25
    for n in range(lo, hi + 1):
        for S in cyclic_subgroups(n):
            if len(S) < 12: continue
            g = LuGraph(n, S)
            if g.k < kmin or not g.k4free(): continue
            t0 = time.time()
            g.setup()
            lam, arg = g.basic()
            thr = 2 * g.k / 3
            r0 = -lam / thr
            line = "n %4d  d %3d  k %3d  tri-orbits %2d  lam_min %.4f  need > %.4f  ratio %.4f" % (n, g.d, g.k, g.T, lam, -thr, r0)
            if r0 < near and g.T > 1:
                lw, mu = g.weighted()
                line += "  | weighted lam %.4f ratio %.4f" % (lw, -lw / thr)
                if -lw / thr < 1: line += "  ARROWS(weighted)"
            if r0 < 1: line += "  ARROWS(basic)"
            print(line + "  [%.1fs]" % (time.time() - t0)); sys.stdout.flush()
