"""Weighted spectral arrowing certificates for subgraphs of a Lu-type circulant that are invariant under a
multiplier subgroup K' = <h> (so everything block-diagonalises over the characters of K').

Certificate: nonnegative weights mu_T on the triangles of a K4-free graph such that
     N(mu) = Deg^(-1/2) M(mu) Deg^(-1/2)   has   lambda_min > -2/3,
where M(mu) = sum_T mu_T (adjacency of T on its 3 edges), Deg_e = sum_{T containing e} mu_T (edges with
Deg_e = 0 are dropped: triangles with weight 0 are simply not used).  Then the graph arrows (3,3).
"""
import numpy as np, sys, time, pickle
from scipy.linalg import eigh
from subgraph_bound import Sub
from lu_scan import cyclic_subgroups


class SymSub:
    def __init__(self, G, h, removed, verbose=True):
        t0 = time.time()
        n, m = G.n, G.m; g = G.g
        self.G = G; self.h = h
        q = 1; x = h
        while x != 1: x = (x * h) % n; q += 1
        self.q = q
        keep = np.ones(n, dtype=bool); keep[list(removed)] = False
        assert all(keep[(h * v) % n] == keep[v] for v in range(n)), "vertex set not invariant"
        self.keep = keep
        # edge permutation induced by h
        xs = np.repeat(np.arange(n), m); js = np.tile(np.arange(m), n)
        Cp = np.array(g.Cp)
        u = (h * xs) % n; dv = (h * Cp[js]) % n
        cid = -np.ones(n, dtype=np.int64)
        for c, j in g.cidx.items(): cid[c] = j
        flip = cid[dv] < 0
        base = np.where(flip, (u + dv) % n, u); jj = np.where(flip, cid[(n - dv) % n], cid[dv])
        P = base * m + jj
        ekeep = keep[G.ends[:, 0]] & keep[G.ends[:, 1]]
        rep = -np.ones(G.E, dtype=np.int64); off = np.zeros(G.E, dtype=np.int64)
        nrep = 0
        for e in np.nonzero(ekeep)[0]:
            if rep[e] >= 0: continue
            f = e; i = 0
            while rep[f] < 0:
                rep[f] = nrep; off[f] = i; f = P[f]; i += 1
            assert f == e and i == q, "edge orbit of unexpected size %d" % i
            nrep += 1
        self.nrep = nrep
        self.edge_rep = rep; self.edge_off = off
        tk = ekeep[G.tri[:, 0]] & ekeep[G.tri[:, 1]] & ekeep[G.tri[:, 2]]
        tri = G.tri[tk]; ttype = G.tri_type[tk]
        A = rep[tri]; O = off[tri]                                    # (nt, 3)
        best = None
        for p in range(3):
            rel = (O - O[:, [p]]) % q
            code = A * q + rel                                         # (nt,3) codes
            code = np.sort(code, axis=1)
            key = (code[:, 0] * (nrep * q) + code[:, 1]) * (nrep * q) + code[:, 2]
            best = key if best is None else np.minimum(best, key)
        uniq, first, inv = np.unique(best, return_index=True, return_inverse=True)
        cnt = np.bincount(inv)
        assert (cnt == q).all(), "triangle orbits of unexpected size"
        self.no = len(uniq)
        self.tri_kept = tri; self.tri_orbit = inv
        self.oa = A[first]; self.oi = O[first]; self.otype = ttype[first]
        self.nv = int(keep.sum()); self.ne = int(ekeep.sum()); self.nt = len(tri)
        self.pairs = [(0, 1), (0, 2), (1, 2)]
        if verbose:
            print("subgraph: %d vertices, %d edges, %d triangles; |K'| = %d; edge orbits %d, triangle orbits %d  [%.1fs]"
                  % (self.nv, self.ne, self.nt, q, nrep, self.no, time.time() - t0)); sys.stdout.flush()

    def degrees(self, mu):
        D = np.zeros(self.nrep)
        for p in range(3): np.add.at(D, self.oa[:, p], mu)
        return D

    def block(self, mu, r, D=None):
        if D is None: D = self.degrees(mu)
        live = D > 1e-12
        B = np.zeros((self.nrep, self.nrep), dtype=complex)
        w = np.exp(2j * np.pi * r / self.q)
        for (p, s) in self.pairs:
            ph = mu * w ** ((self.oi[:, s] - self.oi[:, p]) % self.q)
            np.add.at(B, (self.oa[:, p], self.oa[:, s]), ph)
            np.add.at(B, (self.oa[:, s], self.oa[:, p]), np.conj(ph))
        idx = np.nonzero(live)[0]
        dm = 1.0 / np.sqrt(D[idx])
        Nr = B[np.ix_(idx, idx)] * dm[:, None] * dm[None, :]
        return Nr, idx, dm

    def eig(self, mu, r, kk=6, D=None):
        Nr, idx, dm = self.block(mu, r, D)
        kk = min(kk, Nr.shape[0])
        lam, U = eigh(Nr, subset_by_index=[0, kk - 1])
        return lam, U, idx, dm

    def lam_all(self, mu, rs=None, kk=6):
        D = self.degrees(mu)
        out = {}
        for r in (range(self.q // 2 + 1) if rs is None else rs):
            out[r] = self.eig(mu, r, kk, D)
        return out

    def gradient(self, mu, eigs, delta=0.002):
        """Soft-min gradient of lambda_min(N(mu)) with respect to mu (per triangle orbit)."""
        lmin = min(v[0][0] for v in eigs.values())
        g = np.zeros(self.no); wsum = 0.0
        for r, (lam, U, idx, dm) in eigs.items():
            mult = 1 if (2 * r) % self.q == 0 else 2
            for i in range(len(lam)):
                wt = mult * np.exp(-(lam[i] - lmin) / delta)
                if wt < 1e-4: continue
                v = np.zeros(self.nrep, dtype=complex); v[idx] = U[:, i] * dm
                gi = np.zeros(self.no)
                w = np.exp(2j * np.pi * r / self.q)
                for (p, s) in self.pairs:
                    gi += 2 * np.real(np.conj(v[self.oa[:, p]]) * v[self.oa[:, s]] * w ** ((self.oi[:, s] - self.oi[:, p]) % self.q))
                for p in range(3): gi -= lam[i] * np.abs(v[self.oa[:, p]]) ** 2
                g += wt * gi; wsum += wt
        return g / wsum, lmin


def optimise(Sx, mu, iters=40, eta0=0.3, kk=6, window=0.03, log=None, tag=""):
    best = (-9, None)
    active = None
    for it in range(iters):
        t0 = time.time()
        if it % 8 == 0 or active is None:
            eigs = Sx.lam_all(mu, kk=kk)
            lm = {r: v[0][0] for r, v in eigs.items()}
            lmin = min(lm.values())
            active = [r for r in lm if lm[r] < lmin + window]
        else:
            eigs = Sx.lam_all(mu, rs=active, kk=kk)
        g, lmin = Sx.gradient(mu, eigs)
        full = (it % 8 == 0)
        if full and lmin > best[0]: best = (lmin, mu.copy())
        msg = "%s it %3d  lambda_min(N) = %.5f %s (active blocks %d, zero weights %d)  [%.1fs]" % (tag, it, lmin, "(all blocks)" if full else "", len(active), int((mu < 1e-9).sum()), time.time() - t0)
        print(msg); sys.stdout.flush()
        if log: log.write(msg + "\n"); log.flush()
        eta = eta0 / (1 + it / 15.0)
        mu = mu * np.exp(eta * g / (np.abs(g).max() + 1e-300))
        mu[mu < 1e-6 * mu.max()] = 0.0
        mu = mu / mu.mean()
    eigs = Sx.lam_all(mu, kk=2)
    lmin = min(v[0][0] for v in eigs.values())
    if lmin > best[0]: best = (lmin, mu.copy())
    return best


if __name__ == "__main__":
    n = 941
    S = [s for s in cyclic_subgroups(n) if len(s) == 188][0]
    G = Sub(n, S)
    def order(a):
        x = a; k = 1
        while x != 1: x = (x * a) % n; k += 1
        return k
    gen = next(a for a in range(2, n) if order(a) == n - 1)
    qk = int(sys.argv[1])                       # order of K' (47, 94 or 188)
    cosets = [int(c) for c in sys.argv[2].split(",")] if sys.argv[2] != "none" else []
    rm0 = int(sys.argv[3]); iters = int(sys.argv[4])
    idx = (n - 1) // qk
    h = pow(gen, idx, n)
    Kp = [pow(h, t, n) for t in range(qk)]
    R = set((pow(gen, c, n) * s) % n for c in cosets for s in Kp) | ({0} if rm0 else set())
    Sx = SymSub(G, h, R)
    mu = np.ones(Sx.no)
    tag = "K%d cosets %s%s" % (qk, sys.argv[2], "+0" if rm0 else "")
    best = optimise(Sx, mu, iters=iters, tag=tag)
    print("%s: n' = %d  best lambda_min(N) = %.5f  (need > -0.66667)  %s" % (tag, Sx.nv, best[0], "ARROWS" if best[0] > -2 / 3 else ""))
    pickle.dump({"n": n, "S": S, "h": h, "removed": sorted(R), "mu": best[1], "lam": best[0], "oa": Sx.oa, "oi": Sx.oi},
                open("scan/symsub_K%d_%s_%d.pkl" % (qk, sys.argv[2].replace(",", "_"), rm0), "wb"))
